"""E-F8: 难题基准小探针 (AIME 30 / GPQA-100), 验证"饱和带"的边界条件。

四个配置各跑同一批难题: iq2xxs (基线) / ud_q2kxl / nl_math / nl_global。
单进程顺序起停 llama-server (一次只加载一个模型), -t 16 + 亲和性 0xff。

    # 只看配置与题量, 不加载模型
    python e_f8_hard_probe.py --dry-run

    # 正式跑 (可反复执行, 按 (tag, task_id) 断点续跑)
    python e_f8_hard_probe.py --bench aime30 --configs iq2xxs,ud_q2kxl,nl_math,nl_global \
        --n-predict 4096 --limit 30 --port 18093

输出:
    results/e_f8_hard_probe.jsonl   每题 predicted_n / finish_reason / completion / passed
    results/e_f8_summary.json       每配置 acc / avg predicted_n / capped / McNemar(vs iq2xxs)
"""
import argparse
import ctypes
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

import e17_task_eval as E17  # noqa: E402  复用 post_json / start_server

# CUDA 构建 (上游 b8746): 只有它没有 tcmoe 域补丁, 只用来跑 iq2xxs / ud_q2kxl
BIN_CUDA = os.path.abspath(os.path.join(HERE, "..", "llama.cpp", "build-cuda", "bin"))
# CPU 构建 (tcmoe 域补丁): 23 GiB 的 tcmoe 文件超过 16 GB 显存, 只能纯 CPU
BIN_CPU = os.path.abspath(os.path.join(HERE, "..", "llama.cpp", "build-tcmoe", "bin", "Release"))
TCMOE = r"E:\CUDA11.8\train\tcmoe\out\model-tcmoe-q4k-nl.gguf"
OUT = os.path.join(RES, "e_f8_hard_probe.jsonl")
SUMMARY = os.path.join(RES, "e_f8_summary.json")

CFGS = {
    "iq2xxs": {"model": r"E:\CUDA11.8\model\baselines\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
               "domain": None, "bin": BIN_CUDA, "ngl": 99},
    "ud_q2kxl": {"model": r"E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",
                 "domain": None, "bin": BIN_CUDA, "ngl": 99},
    "nl_math": {"model": TCMOE,
                "domain": "math", "bin": BIN_CPU, "ngl": None},
    "nl_global": {"model": TCMOE,
                  "domain": "global", "bin": BIN_CPU, "ngl": None},
}
ALIASES = {"tcmoe_nl_math": "nl_math", "tcmoe_nl_global": "nl_global"}

SYSTEM_AIME = ("You are a math problem solver. Solve the problem step by step. "
               "The final answer is an integer. You MUST end your response with a final line "
               "of exactly the form <answer>INTEGER</answer>, and write nothing after it.")
SYSTEM_GPQA = ("You are an expert scientist. Answer the multiple-choice question. "
               "Answer with a single letter A/B/C/D.")

PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_INFORMATION = 0x0400
KERNEL32 = ctypes.windll.kernel32


def set_affinity(proc, mask):
    h = KERNEL32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION,
                             False, proc.pid)
    if not h:
        return False
    try:
        return bool(KERNEL32.SetProcessAffinityMask(h, ctypes.c_size_t(mask)))
    finally:
        KERNEL32.CloseHandle(h)


def data_path(bench):
    return os.path.join(DATA, "aime30.jsonl" if bench == "aime30" else "gpqa100.jsonl")


def load_tasks(bench):
    p = data_path(bench)
    if not os.path.exists(p):
        raise SystemExit(f"缺少数据: {p} (先按实验记录下载并落盘)")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def format_options(opts):
    if isinstance(opts, dict):
        items = sorted(opts.items())
    else:
        items = list(zip("ABCD", opts))
    return "\n".join(f"{k}. {v}" for k, v in items)


def build_prompt(bench, row):
    if bench == "aime30":
        system, user = SYSTEM_AIME, row["question"].strip()
    else:
        system = SYSTEM_GPQA
        user = row["question"].strip() + "\n\n" + format_options(row.get("options") or [])
    stops = ["\n\nQuestion:", "\n\nQ:", "\n\nProblem:", "</answer>"]
    # Qwen3.6 模板 (enable_thinking=false) 的等价纯文本形式
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"), stops


def gen_raw(port, prompt, n_predict, stops, domain=None, seed=1234, repeat_penalty=0.0):
    """同 e17.gen 的请求, 但保留 stop 状态以便记录 finish_reason。"""
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
               "top_k": 1, "seed": seed, "cache_prompt": False,
               "stop": stops + ["<|im_end|>", "<|endoftext|>"], "stream": False}
    if repeat_penalty > 0:
        payload["repeat_penalty"] = repeat_penalty
    if domain:
        payload["domain"] = domain
    r = E17.post_json(f"http://127.0.0.1:{port}/completion", payload, timeout=1800)
    content = r.get("content", "")
    tim = r.get("timings") or {}
    pn = int(tim.get("predicted_n") or r.get("tokens_predicted") or 0)
    if r.get("stopped_limit") or r.get("truncated"):
        finish = "length"
    elif r.get("stopped_eos"):
        finish = "eos"
    elif r.get("stopped_word") or r.get("stopping_word"):
        finish = "stop_word"
    elif pn >= n_predict:
        finish = "length"
    else:
        finish = "stop"
    return content, tim, finish, pn


def last_boxed(text):
    idx = text.rfind("\\boxed")
    while idx != -1:
        j = text.find("{", idx + len("\\boxed"))
        if j != -1:
            depth = 0
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[j + 1:k]
        idx = text.rfind("\\boxed", 0, idx)
    return None


NUM_RE = re.compile(r"[-+]?\d[\d,]*")


def extract_int(text):
    m = re.findall(r"<answer>\s*([-+]?\d[\d,]*)", text)
    if m:
        return int(m[-1].replace(",", ""))
    box = last_boxed(text)
    target = box if box is not None else text
    nums = NUM_RE.findall(target)
    if not nums and box is not None:
        nums = NUM_RE.findall(text)
    if not nums:
        return None
    s = nums[-1].replace(",", "")
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def extract_letter(text):
    m = re.findall(r"\\boxed\s*\{\s*([A-D])\s*\}", text)
    if m:
        return m[-1]
    letters = re.findall(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])", text)
    return letters[-1] if letters else None


def gold_of(bench, row):
    """gpqa100.jsonl 用 answer_letter/answer_index, aime30.jsonl 用 answer。"""
    if bench == "aime30":
        return row["answer"]
    if row.get("answer_letter"):
        return row["answer_letter"]
    if row.get("answer") is not None:
        return row["answer"]
    return "ABCD"[int(row["answer_index"])]


def grade(bench, row, content):
    if bench == "aime30":
        pred = extract_int(content)
        gold = extract_int(str(row["answer"]))
        return pred, pred is not None and gold is not None and pred == gold
    pred = extract_letter(content)
    gold = str(gold_of(bench, row)).strip().upper()[:1]
    return pred, pred is not None and pred == gold


def done_set():
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["tag"], r["task_id"]))
                except Exception:  # noqa: BLE001
                    continue
    return done


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def summarize(bench, n_predict, tags):
    recs = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("bench", "aime30") == bench:
                    recs.append(r)
    by_tag = {t: [r for r in recs if r["tag"] == t] for t in tags}
    out = {"bench": bench, "n_predict": n_predict, "configs": {}, "mcnemar": {}}
    p = data_path(bench)
    if os.path.exists(p):
        import hashlib
        out["data_path"] = p
        out["data_sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        out["n_questions"] = len(load_tasks(bench))
    for t in tags:
        rs = by_tag[t]
        n = len(rs)
        passed = sum(1 for r in rs if r.get("passed"))
        pn = [r.get("predicted_n") or 0 for r in rs]
        capped = sum(1 for r in rs if (r.get("predicted_n") or 0) >= n_predict
                     or r.get("finish_reason") == "length")
        fins = {}
        for r in rs:
            fins[r.get("finish_reason")] = fins.get(r.get("finish_reason"), 0) + 1
        out["configs"][t] = {
            "model": CFGS[t]["model"], "domain": CFGS[t]["domain"],
            "n": n, "passed": passed, "acc": round(passed / n, 4) if n else None,
            "avg_predicted_n": round(sum(pn) / n, 1) if n else None,
            "avg_gen_s": round(sum(r.get("gen_s") or 0 for r in rs) / n, 1) if n else None,
            "capped": capped, "capped_rate": round(capped / n, 4) if n else None,
            "finish_reasons": fins,
        }
    if "iq2xxs" in by_tag:
        base = {r["task_id"]: bool(r.get("passed")) for r in by_tag["iq2xxs"]}
        for t in tags:
            if t == "iq2xxs":
                continue
            other = {r["task_id"]: bool(r.get("passed")) for r in by_tag[t]}
            common = sorted(set(base) & set(other))
            b = sum(1 for k in common if base[k] and not other[k])
            c = sum(1 for k in common if not base[k] and other[k])
            out["mcnemar"][t] = {"paired": len(common), "b_baseline_pass_other_fail": b,
                                 "c_baseline_fail_other_pass": c,
                                 "p_exact": round(mcnemar_exact(b, c), 6)}
    os.makedirs(RES, exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["aime30", "gpqa100"], default="aime30")
    ap.add_argument("--configs", default="iq2xxs,ud_q2kxl,nl_math,nl_global")
    ap.add_argument("--n-predict", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--mask", type=lambda s: int(s, 0), default=0xFF)
    ap.add_argument("--ctx", type=int, default=0, help="0 表示自动: max(4096, n_predict+2048)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repeat-penalty", type=float, default=0.0,
                    help="0 表示不下发该字段 (保持原贪心协议)")
    ap.add_argument("--out", default="", help="改写输出 jsonl (诊断用, 不污染主结果)")
    ap.add_argument("--ngl", type=int, default=-1,
                    help="-1 表示按配置 (CUDA 臂 99, CPU 臂不下发 -ngl)")
    a = ap.parse_args(argv)

    global OUT, SUMMARY
    if a.out:
        OUT = a.out if os.path.isabs(a.out) else os.path.join(RES, a.out)
        SUMMARY = OUT.replace('.jsonl', '_summary.json').replace('.json', '_summary.json')

    tags = []
    for s in a.configs.split(","):
        t = ALIASES.get(s.strip(), s.strip())
        if t:
            if t not in CFGS:
                raise SystemExit(f"未知配置: {t}")
            tags.append(t)
    tasks = load_tasks(a.bench)
    if a.limit:
        tasks = tasks[: a.limit]
    ctx = a.ctx or max(4096, a.n_predict + 2048)

    if a.dry_run:
        p = data_path(a.bench)
        print(f"[dry-run] bench={a.bench} data={p} questions={len(tasks)} "
              f"limit={a.limit} planned={len(tasks) * len(tags)} requests", flush=True)
        print(f"[dry-run] server=cpu:{BIN_CPU}\\llama-server.exe exists={os.path.exists(os.path.join(BIN_CPU, 'llama-server.exe'))} "
              f"cuda:{BIN_CUDA}\\llama-server.exe exists={os.path.exists(os.path.join(BIN_CUDA, 'llama-server.exe'))} "
              f"threads={a.threads} mask=0x{a.mask:X} ctx={ctx} n_predict={a.n_predict} port={a.port}",
              flush=True)
        for t in tags:
            c = CFGS[t]
            ngl = a.ngl if a.ngl >= 0 else c["ngl"]
            print(f"  {t:10s} domain={c['domain'] or '-':7s} ngl={ngl if ngl is not None else '-':3} "
                  f"exists={os.path.exists(c['model'])} {c['model']}", flush=True)
        return

    os.makedirs(RES, exist_ok=True)
    done = done_set()
    if done:
        print(f"[resume] 已有 {len(done)} 条 (tag, task) 记录", flush=True)

    for tag in tags:
        c = CFGS[tag]
        todo = [r for r in tasks if (tag, r["task_id"]) not in done]
        print(f"[{tag}] 待跑 {len(todo)} / {len(tasks)} 题, domain={c['domain']}", flush=True)
        if not todo:
            continue
        log_path = os.path.join(RES, f"e_f8_{tag}.server.log")
        ngl = a.ngl if a.ngl >= 0 else c["ngl"]
        extra = ("-ngl", str(ngl)) if ngl is not None else ()
        t0 = time.time()
        proc, lg = E17.start_server(c["bin"], os.path.abspath(c["model"]), a.port,
                                    a.threads, ctx, log_path, extra=extra)
        ok = set_affinity(proc, a.mask)
        print(f"[server] {tag} bin={c['bin']} ngl={ngl} port={a.port} threads={a.threads} ctx={ctx} "
              f"affinity=0x{a.mask:X} set={ok} ({time.time() - t0:.0f}s)", flush=True)
        n_done = n_pass = 0
        try:
            with open(OUT, "a", encoding="utf-8") as f:
                for i, row in enumerate(todo):
                    prompt, stops = build_prompt(a.bench, row)
                    ts = time.time()
                    try:
                        content, tim, finish, pn = gen_raw(a.port, prompt, a.n_predict,
                                                           stops, c["domain"],
                                                           repeat_penalty=a.repeat_penalty)
                    except Exception as e:  # noqa: BLE001  单题异常不中断, 留待续跑
                        print(f"  [skip] {tag} {row['task_id']}: {e!r}", flush=True)
                        continue
                    pred, passed = grade(a.bench, row, content)
                    rec = {"bench": a.bench, "tag": tag, "task_id": row["task_id"],
                           "predicted": pred, "gold": gold_of(a.bench, row), "passed": passed,
                           "n_predict": a.n_predict, "predicted_n": pn,
                           "repeat_penalty": a.repeat_penalty,
                           "finish_reason": finish, "timings": tim,
                           "gen_s": round(time.time() - ts, 2), "completion": content}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_done += 1
                    n_pass += int(passed)
                    if (i + 1) % 5 == 0 or i + 1 == len(todo):
                        el = time.time() - t0
                        rate = n_done / el if el else 0
                        eta = (len(todo) - n_done) / rate / 60 if rate else 0
                        print(f"  {tag} {i + 1}/{len(todo)} acc={n_pass / n_done:.3f} "
                              f"{rec['gen_s']:.1f}s pn={pn} {finish} "
                              f"elapsed={el / 60:.0f}m ETA={eta:.0f}m", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                proc.kill()
            lg.close()
        print(f"[{tag}] 本段新增 {n_done} 条, acc={n_pass / n_done:.3f}" if n_done
              else f"[{tag}] 本段无新增", flush=True)
        time.sleep(2)

    s = summarize(a.bench, a.n_predict, tags)
    print(f"\n{'tag':10s} {'n':>3s} {'acc':>7s} {'avg_n':>7s} {'capped':>7s} {'avg_s':>7s}",
          flush=True)
    for t in tags:
        v = s["configs"][t]
        print(f"{t:10s} {v['n']:3d} {v['acc'] if v['acc'] is not None else 0:7.3f} "
              f"{v['avg_predicted_n'] or 0:7.1f} {v['capped']:7d} {v['avg_gen_s'] or 0:7.1f}",
              flush=True)
    for t, v in s["mcnemar"].items():
        print(f"[mcnemar] {t} vs iq2xxs: b={v['b_baseline_pass_other_fail']} "
              f"c={v['c_baseline_fail_other_pass']} p={v['p_exact']}", flush=True)
    print(f"[e_f8] -> {OUT} / {SUMMARY}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
