"""E-G1 跨模型评测: gemma-4-26B-A4B (MoE) 三任务 (gsm8k / humaneval / gpqa)。

协议与 e17_task_eval.py 保持一致, 差别只在推理接口:
gemma4 的 chat template 由 GGUF 自带, 因此走 /v1/chat/completions 的 messages,
不手写 <|im_start|> 前缀; system 指令沿用 e17 的 SYSTEM_MATH / SYSTEM_CODE。

    # 例: 一个配置/一个任务 (可断点续跑, 按 (tag, task_id) 跳过)
    python e_g1_eval.py run --tag B_ud_q4kxl --model E:\\CUDA11.8\\model\\gemma4\\ud_q4kxl\\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
        --task gsm8k --port 18101 --threads 16 --cpus 0-15 --n-predict 1024

    # 汇总 (所有 e_g1_gemma4_*.jsonl)
    python e_g1_eval.py score

判分: GSM8K 复用 e17.extract_number; HumanEval 复用 e17.extract_code + 隔离子进程跑单测;
GPQA 从输出里抽最后一个可靠字母 (A/B/C/D)。
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

import e17_task_eval as E17  # noqa: E402  复用 post_json / start_server / 抽取逻辑
from pin import pin_self  # noqa: E402

BIN_DIR = os.environ.get("E_G_BIN") or os.path.abspath(
    os.path.join(HERE, "..", "llama.cpp", "build-tcmoe", "bin", "Release"))
FILE_PREFIX = os.environ.get("E_G_PREFIX", "e_g1_gemma4_")
BASELINE = os.environ.get("E_G_BASELINE", "B_ud_q4kxl")

SYSTEM_GPQA = ("You are an expert scientist. Answer the multiple-choice question. "
               "Reply with ONLY the single letter of the correct option (A, B, C or D). "
               "Do not explain, do not show reasoning, do not restate the question.")
SYSTEM = {"gsm8k": E17.SYSTEM_MATH, "humaneval": E17.SYSTEM_CODE,
          "gpqa": SYSTEM_GPQA}
STOPS = {"gsm8k": ["\n\nQuestion:", "\n\nQ:"],
         "humaneval": ["\nif __name__", "\nprint("],
         "gpqa": ["\n\nQuestion:", "\n\nQ:", "\n\nProblem:"]}


def load_tasks(task):
    if task == "gpqa":
        path = os.path.join(DATA, "gpqa100.jsonl")
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    return E17.load_tasks(task)


def out_path(tag, task):
    return os.path.join(RES, f"{FILE_PREFIX}{tag}_{task}.jsonl")


def build_messages(task, row):
    if task == "humaneval":
        user = row["prompt"].strip()
    elif task == "gsm8k":
        user = row["question"].strip()
    else:
        user = row["question"].strip() + "\n\n" + "\n".join(
            f"{c}. {o}" for c, o in zip("ABCD", row["options"]))
    return [{"role": "system", "content": SYSTEM[task]},
            {"role": "user", "content": user}]


def post_json(url, payload, timeout=1200):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def finish_of(resp, choice, pn, n_predict):
    fr = choice.get("finish_reason")
    if fr:
        return fr
    if resp.get("stopped_limit") or resp.get("truncated"):
        return "length"
    if resp.get("stopped_eos"):
        return "eos"
    if resp.get("stopped_word") or resp.get("stopping_word"):
        return "stop_word"
    return "length" if pn >= n_predict else "stop"


def gen_chat(port, task, row, n_predict, seed=1234):
    # gemma4 模板默认 enable_thinking=false (= 直接输出 <|channel|> 关闭思维通道);
    # 不显式传参时 llama-server 会打开 thinking, 导致 1024 上限内大量记录在思维通道里截断
    # (content 为空)。这里显式关闭, 与 e17(Qwen3.6) 的评测协议保持一致。
    payload = {"messages": build_messages(task, row), "max_tokens": n_predict,
               "temperature": 0.0, "top_k": 1, "seed": seed,
               "cache_prompt": False, "stop": STOPS[task], "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    resp = post_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if not content and reasoning:
        content = reasoning  # 兜底: 万一模板仍产出思维通道
    tim = resp.get("timings") or {}
    pn = int(tim.get("predicted_n")
             or (resp.get("usage") or {}).get("completion_tokens") or 0)
    return content, pn, finish_of(resp, choice, pn, n_predict), reasoning


# ---------------- 判分 ----------------

P_ANSWER_PAREN = re.compile(
    r"(?:answer|choice|option|correct\s+answer|correct\s+option)"
    r"\s*(?:is|would\s+be|:|=|-)?\s*\(\s*([A-D])\s*\)", re.I)
P_ANSWER_BARE = re.compile(
    r"(?:answer|choice|option|correct\s+answer|correct\s+option)"
    r"\s*(?:is|would\s+be|:|=|-)\s*(?:\*\*)?([A-D])(?![A-Za-z0-9])", re.I)
P_PAREN = re.compile(r"\(\s*([A-D])\s*\)")
P_BOXED = re.compile(r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?([A-D])")
P_LINE = re.compile(r"^\s*(?:\*\*)?([A-D])(?:\*\*)?\s*[.)]?\s*$", re.M)

CANDIDATES = [(P_ANSWER_PAREN, 0), (P_ANSWER_BARE, 0), (P_PAREN, 1),
              (P_BOXED, 1), (P_LINE, 2)]


def extract_letter(text):
    """按出现顺序取最后一个可靠匹配 (同位置时优先 answer/option 显式句式)。"""
    cands = []
    for pat, prio in CANDIDATES:
        for m in pat.finditer(text):
            cands.append((m.start(), prio, m.group(1).upper()))
    if not cands:
        return None
    return max(cands, key=lambda c: (c[0], -c[1]))[2]


def check_humaneval(row, content, timeout=8):
    """照抄 e17.run_humaneval 的代码抽取 + 隔离单测执行。"""
    code = E17.extract_code(row, content)
    prog = code + "\n\n" + row["test"] + f"\n\ncheck({row['entry_point']})\n"
    ok, err = False, ""
    td = tempfile.mkdtemp(prefix="he_")  # 不用 TemporaryDirectory: 清理失败会中断整批
    try:
        p = os.path.join(td, "sol.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(prog)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, timeout=timeout,
                               capture_output=True, text=True)
            ok = r.returncode == 0
            if not ok:
                lines = (r.stderr or "").strip().splitlines()
                err = lines[-1][:200] if lines else "fail"
        except subprocess.TimeoutExpired:
            err = "timeout"
        except Exception as e:  # noqa: BLE001
            err = f"run-fail: {e}"
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return ok, err


def grade(task, row, content):
    if task == "gsm8k":
        pred = E17.extract_number(content)
        gold = float(row["gold"])
        return pred is not None and abs(pred - gold) < 1e-4
    if task == "humaneval":
        return check_humaneval(row, content)[0]
    pred = extract_letter(content)
    return pred is not None and pred == str(row["answer_letter"]).upper()


# ---------------- run ----------------

def done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["task_id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


def run(args):
    os.makedirs(RES, exist_ok=True)
    path = out_path(args.tag, args.task)
    done = done_ids(path)
    tasks = load_tasks(args.task)
    if args.limit:
        tasks = tasks[: args.limit]
    todo = [t for t in tasks if t["task_id"] not in done]
    if args.max_new:
        todo = todo[: args.max_new]  # 块交错: 每轮只推进 N 题
    print(f"[{args.tag}/{args.task}] 共 {len(tasks)} 题, 已完成 {len(done)}, 待跑 {len(todo)}",
          flush=True)
    if not todo:
        return

    cpus = pin_self(cpus=args.cpus) if args.cpus else pin_self(numa=0)
    print(f"[{args.tag}/{args.task}] 绑定 CPU {cpus}", flush=True)

    log_path = os.path.join(RES, f"{FILE_PREFIX}{args.tag}_{args.task}.server.log")
    proc, log = E17.start_server(args.bin_dir or BIN_DIR, os.path.abspath(args.model), args.port,
                                 args.threads, args.ctx, log_path,
                                 extra=("-ngl", str(args.ngl)))
    n_ok = 0
    t0 = time.time()
    try:
        with open(path, "a", encoding="utf-8") as f:
            for i, row in enumerate(todo):
                ts = time.time()
                try:
                    content, pn, finish, reasoning = gen_chat(args.port, args.task, row,
                                                             args.n_predict)
                except Exception as e:  # noqa: BLE001  单题异常不中断, 留待续跑
                    print(f"  [skip] {row['task_id']}: {e!r}", flush=True)
                    continue
                ok = grade(args.task, row, content)
                rec = {"task_id": row["task_id"], "tag": args.tag, "task": args.task,
                       "passed": bool(ok), "predicted_n": pn,
                       "gen_ms": round((time.time() - ts) * 1000),
                       "finish_reason": finish, "text": content[:4000],
                       "reasoning": reasoning[:2000],
                       "block": args.block}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += int(ok)
                if i < 3 or (i + 1) % 20 == 0 or i + 1 == len(todo):
                    el = time.time() - t0
                    print(f"  {i + 1}/{len(todo)} {row['task_id']} pass={ok} "
                          f"acc={n_ok / (i + 1):.3f} {rec['gen_ms'] / 1000:.1f}s "
                          f"pn={pn} {finish} elapsed={el / 60:.0f}m", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    print(f"[{args.tag}/{args.task}] 新增 {len(todo)} 条, 本批 acc={n_ok / len(todo):.3f}, "
          f"用时 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------- score ----------------

def wilson(p, n, z=1.959963984540054):
    if n <= 0:
        return (0.0, 1.0)
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def load_records(res_dir):
    """返回 {(tag, task): {task_id: rec}}; 重复 task_id 保留首次。"""
    per = {}
    files = sorted(x for x in os.listdir(res_dir)
                   if x.startswith(FILE_PREFIX) and x.endswith(".jsonl"))
    for fn in files:
        stem = fn[len(FILE_PREFIX):-len(".jsonl")]
        tag, task = stem.rsplit("_", 1)
        recs = per.setdefault((tag, task), {})
        with open(os.path.join(res_dir, fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                recs.setdefault(r["task_id"], r)
    return per


def score(args):
    res_dir = os.path.abspath(args.results)
    per = load_records(res_dir)
    if not per:
        print(f"[score] {res_dir} 下没有 {FILE_PREFIX}*.jsonl")
        return

    rows = []
    for (tag, task), recs in sorted(per.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n = len(recs)
        ok = sum(int(bool(r.get("passed"))) for r in recs.values())
        p = ok / n if n else 0.0
        lo, hi = wilson(p, n)
        rows.append({"tag": tag, "task": task, "n": n, "passed": ok, "acc": p,
                     "wilson_lo": lo, "wilson_hi": hi,
                     "avg_gen_s": round(sum(r.get("gen_ms", 0) or 0
                                            for r in recs.values()) / n / 1000, 1) if n else 0.0,
                     "avg_predicted_n": round(sum(r.get("predicted_n", 0) or 0
                                                  for r in recs.values()) / n, 1) if n else 0.0})

    rng = random.Random(20261104)
    pairs = []
    tasks_seen = sorted({t for _, t in per})
    for task in tasks_seen:
        base_recs = per.get((args.baseline, task))
        if not base_recs:
            continue
        base = {tid: bool(r.get("passed")) for tid, r in base_recs.items()}
        for (tag, t), recs in sorted(per.items()):
            if t != task or tag == args.baseline:
                continue
            other = {tid: bool(r.get("passed")) for tid, r in recs.items()}
            common = sorted(set(base) & set(other))
            if not common:
                continue
            b = sum(1 for k in common if base[k] and not other[k])
            c = sum(1 for k in common if not base[k] and other[k])
            acc_b = sum(base[k] for k in common) / len(common)
            acc_o = sum(other[k] for k in common) / len(common)
            diffs = []
            for _ in range(args.bootstrap):
                samp = [common[rng.randrange(len(common))] for _ in common]
                diffs.append(sum(other[k] for k in samp) / len(samp)
                             - sum(base[k] for k in samp) / len(samp))
            diffs.sort()
            k_lo = int(math.floor(0.025 * len(diffs)))
            k_hi = max(int(math.ceil(0.975 * len(diffs))) - 1, k_lo)
            pairs.append({"task": task, "tag": tag, "baseline": args.baseline,
                          "paired_n": len(common),
                          "b_baseline_pass_tag_fail": b,
                          "c_baseline_fail_tag_pass": c,
                          "mcnemar_p_exact": mcnemar_exact(b, c),
                          "acc_tag": acc_o, "acc_baseline": acc_b,
                          "acc_diff": acc_o - acc_b,
                          "boot_ci_lo": diffs[k_lo], "boot_ci_hi": diffs[k_hi]})

    summary = {"baseline": args.baseline, "bootstrap": args.bootstrap,
               "bootstrap_seed": 20261104,
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "per_tag_task": rows, "vs_baseline": pairs}

    out = os.path.join(res_dir, args.out or f"{FILE_PREFIX}summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"{'task':10s} {'tag':16s} {'n':>4s} {'acc':>7s} {'wilson 95%':>17s} "
          f"{'avg_s':>7s} {'avg_n':>7s}")
    for r in rows:
        print(f"{r['task']:10s} {r['tag']:16s} {r['n']:4d} {r['acc']:7.3f} "
              f"[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] {r['avg_gen_s']:7.1f} "
              f"{r['avg_predicted_n']:7.1f}")
    for v in pairs:
        print(f"[mcnemar] {v['task']}/{v['tag']} vs {v['baseline']}: "
              f"paired={v['paired_n']} b={v['b_baseline_pass_tag_fail']} "
              f"c={v['c_baseline_fail_tag_pass']} p={v['mcnemar_p_exact']:.4f} | "
              f"dacc={v['acc_diff']:+.3f} boot95%=[{v['boot_ci_lo']:+.3f},{v['boot_ci_hi']:+.3f}]")
    print(f"[score] -> {out}")


def main(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tag", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=["gsm8k", "humaneval", "gpqa"])
    p.add_argument("--port", type=int, default=18101)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--cpus", default="0-15")
    p.add_argument("--n-predict", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new", type=int, default=0,
                   help="本次最多推进多少题 (块交错用; 0=全部剩余)")
    p.add_argument("--block", default="g1", help="运行块标签, 写入每条记录")
    p.add_argument("--bin-dir", default="", help="llama.cpp 可执行目录 (默认 CPU 构建)")
    p.add_argument("--ngl", type=int, default=99,
                   help="上 GPU 的层数; 不给 -ngl 时 llama.cpp 按加载时空闲显存自动分裂, "
                        "同一模型在 16 GiB 卡上会时快时慢 (14.4 GiB 权重最坏掉到 1.6 t/s)")
    s = sub.add_parser("score")
    s.add_argument("--results", default=RES, help="结果目录 (默认 results/)")
    s.add_argument("--baseline", default=BASELINE, help="McNemar 主基线 tag")
    s.add_argument("--bootstrap", type=int, default=1000, help="序列级 bootstrap 次数")
    s.add_argument("--out", default="", help="summary 输出路径 (默认 <prefix>summary.json)")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        score(a)
    else:
        run(a)


if __name__ == "__main__":
    main(sys.argv[1:])
