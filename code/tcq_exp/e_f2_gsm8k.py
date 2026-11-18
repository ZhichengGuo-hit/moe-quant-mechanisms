"""E-F2: GSM8K 全量 1319 题 x 2 配置 (iq2xxs vs tcmoe_nl_global), 单 server 16 线程 + 分块交替。

纪律落地:
  - 两个 llama-server 常驻, 各自 -t 16 且亲和性 0-15; 同一时刻只有一个 server 在生成,
    另一个空闲 (不抢 CPU/内存带宽), 因此每段都是满 16 线程速度;
  - 按 50 题一块交替: 偶数块 iq 先跑完 50 题再 nl, 奇数块反之, 相邻块顺序翻转;
  - 每题落 predicted_n 与完整输出, JSONL 断点续跑 (按 (tag, task_id) 去重);
  - 每块打印进度 + ETA。

用法:
    python e_f2_gsm8k.py                 # 全量 (可反复执行, 自动续跑)
    python e_f2_gsm8k.py --limit 8       # 冒烟 (只跑前 8 题)
    python e_f2_gsm8k.py --block 50      # 每块题数 (默认 50)
输出: results/e_f2_gsm8k_full.jsonl
"""
import argparse
import ctypes
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

import e17_task_eval as E17  # noqa: E402  复用 build_prompt / gen / extract_number / start_server

BIN_DIR = os.path.abspath(os.path.join(HERE, "..", "llama.cpp", "build-tcmoe", "bin", "Release"))
OUT = os.path.join(RES, "e_f2_gsm8k_full.jsonl")

CFGS = [
    {"tag": "iq2xxs", "model": r"E:\CUDA11.8\model\baselines\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
     "port": 18091, "mask": 0xFFFF, "domain": None},
    {"tag": "tcmoe_nl_global", "model": r"E:\CUDA11.8\train\tcmoe\out\model-tcmoe-q4k-nl.gguf",
     "port": 18092, "mask": 0xFFFF, "domain": "global"},
]

PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_INFORMATION = 0x0400
KERNEL32 = ctypes.windll.kernel32


def set_affinity(proc, mask):
    h = KERNEL32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, proc.pid)
    if not h:
        return False
    try:
        return bool(KERNEL32.SetProcessAffinityMask(h, ctypes.c_size_t(mask)))
    finally:
        KERNEL32.CloseHandle(h)


def load_tasks():
    """GSM8K test 1319 题, 首次运行时落盘为 data/gsm8k_full.jsonl。"""
    p = os.path.join(DATA, "gsm8k_full.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(DATA, "gsm8k_test.parquet")).to_pylist()
    tasks = []
    for i, r in enumerate(rows):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        tasks.append({"task_id": f"gsm8k-{i + 1:04d}", "question": r["question"],
                      "gold": gold, "answer": r["answer"]})
    with open(p, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[data] {len(tasks)} questions -> {p}", flush=True)
    return tasks


def done_set():
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                done.add((r["tag"], r["task_id"]))
    return done


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--block", type=int, default=50)
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)

    tasks = load_tasks()
    if a.limit:
        tasks = tasks[: a.limit]
    done = done_set()
    if done:
        print(f"[resume] 已有 {len(done)} 条 (tag, task) 记录", flush=True)

    procs, logs = {}, []
    for c in CFGS:
        log_path = os.path.join(RES, f"e_f2_{c['tag']}.server.log")
        t0 = time.time()
        proc, lg = E17.start_server(BIN_DIR, os.path.abspath(c["model"]), c["port"],
                                    a.threads, 4096, log_path)
        ok = set_affinity(proc, c["mask"])
        print(f"[server] {c['tag']} port={c['port']} threads={a.threads} "
              f"affinity=0x{c['mask']:04X} set={ok} ({time.time() - t0:.0f}s)", flush=True)
        procs[c["tag"]] = proc
        logs.append(lg)

    n_todo = sum(1 for t in tasks for c in CFGS if (c["tag"], t["task_id"]) not in done)
    blocks = [tasks[i: i + a.block] for i in range(0, len(tasks), a.block)]
    print(f"[plan] 待跑 {n_todo} 条请求 / {len(tasks)} 题 / {len(blocks)} 块 "
          f"(每块 {a.block} 题, 交替顺序)", flush=True)
    t_all = time.time()
    n_done = 0
    n_pass = {c["tag"]: 0 for c in CFGS}
    n_cnt = {c["tag"]: 0 for c in CFGS}
    try:
        with open(OUT, "a", encoding="utf-8") as f:
            for bi, blk in enumerate(blocks):
                order = CFGS if bi % 2 == 0 else list(reversed(CFGS))
                for c in order:
                    for row in blk:
                        if (c["tag"], row["task_id"]) in done:
                            continue
                        prompt, n_pred, stops = E17.build_prompt("gsm8k", row, a.n_predict)
                        ts = time.time()
                        try:
                            content, tim = E17.gen(c["port"], prompt, n_pred, stops, c["domain"])
                        except Exception as e:  # noqa: BLE001  单题异常不中断, 留待续跑
                            print(f"  [skip] {c['tag']} {row['task_id']}: {e!r}", flush=True)
                            continue
                        pred = E17.extract_number(content)
                        passed = pred is not None and abs(pred - float(row["gold"])) < 1e-4
                        rec = {"tag": c["tag"], "task_id": row["task_id"], "pred": pred,
                               "gold": row["gold"], "passed": passed,
                               "predicted_n": (tim or {}).get("predicted_n"),
                               "gen_ms": round((time.time() - ts) * 1000),
                               "block": bi, "order_pos": order.index(c), "completion": content}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        n_done += 1
                        n_cnt[c["tag"]] += 1
                        n_pass[c["tag"]] += int(passed)
                        done.add((c["tag"], row["task_id"]))
                el = time.time() - t_all
                rate = n_done / el if el else 0
                eta = (n_todo - n_done) / rate / 3600 if rate else 0
                print(f"  块 {bi + 1}/{len(blocks)} 完 | 累计 {n_done}/{n_todo} 请求 "
                      f"| iq={n_pass['iq2xxs']}/{n_cnt['iq2xxs']} "
                      f"nl={n_pass['tcmoe_nl_global']}/{n_cnt['tcmoe_nl_global']} "
                      f"| {el / 3600:.2f}h elapsed, ETA {eta:.1f}h", flush=True)
    finally:
        for p in procs.values():
            p.terminate()
            try:
                p.wait(timeout=30)
            except Exception:  # noqa: BLE001
                p.kill()
        for lg in logs:
            lg.close()
    print(f"[e_f2] 本段新增 {n_done} 条, 用时 {(time.time() - t_all) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
