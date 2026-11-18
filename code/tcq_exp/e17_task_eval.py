"""E17 任务级评测: HumanEval pass@1 + GSM8K accuracy。

配置由外部 runner 逐个启动 llama-server, 本脚本负责出题、生成、判分与汇总。

    # 1) 准备数据 (只需一次)
    python e17_task_eval.py prepare

    # 2) 跑一个配置/一个任务 (可断点续跑)
    python e17_task_eval.py run --tag tcmoe_code --model ..\tcmoe\out\model-tcmoe-q4k.gguf \
        --domain code --task humaneval --port 18081 --threads 16 --cpus 0-15
    python e17_task_eval.py run --tag iq3xxs --model E:\CUDA11.8\model\baselines\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
        --task gsm8k --port 18081

    # 3) 汇总
    python e17_task_eval.py score

判分: HumanEval 在隔离子进程里跑单测(8s 超时); GSM8K 取 #### 后的数字。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")

SYSTEM_CODE = ("You are an expert Python programmer. Complete the function. "
               "Output only code, without explanations or tests.")
SYSTEM_MATH = ("You are a math problem solver. Think step by step, then output the final "
               "numeric answer on a new line as: #### <number>")


def prepare():
    import pyarrow.parquet as pq
    os.makedirs(DATA, exist_ok=True)
    he = pq.read_table(os.path.join(DATA, "humaneval.parquet")).to_pylist()
    rows = [{"task_id": r["task_id"], "prompt": r["prompt"], "test": r["test"],
             "entry_point": r["entry_point"]} for r in he]
    with open(os.path.join(DATA, "humaneval_164.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gs = pq.read_table(os.path.join(DATA, "gsm8k_test.parquet")).to_pylist()[:200]
    with open(os.path.join(DATA, "gsm8k_200.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(gs):
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"task_id": f"gsm8k-{i}", "question": r["question"],
                                "gold": gold, "answer": r["answer"]},
                               ensure_ascii=False) + "\n")
    print(f"humaneval {len(rows)}  gsm8k {len(gs)} -> {DATA}")


def load_tasks(task):
    path = os.path.join(DATA, "humaneval_164.jsonl" if task == "humaneval" else "gsm8k_200.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_prompt(task, row, n_predict_cap=512):
    if task == "humaneval":
        system, user = SYSTEM_CODE, row["prompt"].strip()
        # 不能停 "\ndef " / "\n# ": 模型会重写函数签名或先写注释, 会被腰斩
        n_predict, stops = n_predict_cap, ["\nif __name__", "\nprint("]
    else:
        system, user = SYSTEM_MATH, row["question"].strip()
        n_predict, stops = n_predict_cap, ["\n\nQuestion:", "\n\nQ:"]
    # Qwen3.6 模板 (enable_thinking=false) 的等价纯文本形式
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"), n_predict, stops


def post_json(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(port, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server 退出, code={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("llama-server 启动超时")


def start_server(bin_dir, model, port, threads, ctx, log_path, extra=()):
    cmd = [os.path.join(bin_dir, "llama-server.exe"), "-m", model,
           "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx), "-np", "1",
           "-t", str(threads), "--no-webui", "--verbosity", "1", *extra]
    log = open(log_path, "w", encoding="utf-8")
    log.write(" ".join(cmd) + "\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_health(port, proc)
    except BaseException:
        # 启动失败必须回收, 否则残留的 server 会继续加载模型并拖慢后续任务
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        raise
    return proc, log


def gen(port, prompt, n_predict, stops, domain=None, seed=1234):
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
               "top_k": 1, "seed": seed, "cache_prompt": False,
               "stop": stops + ["<|im_end|>", "<|endoftext|>"],
               "stream": False}
    if domain:
        payload["domain"] = domain
    r = post_json(f"http://127.0.0.1:{port}/completion", payload)
    return r.get("content", ""), r.get("timings", {})


FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.S)


def extract_code(row, content):
    m = FENCE.search(content)
    code = m.group(1) if m else content
    code = code.replace("```", "").strip("\n")
    if code.endswith("</think>"):
        code = code[: -len("</think>")].rstrip()
    if f"def {row['entry_point']}" not in code:
        code = row["prompt"].rstrip("\n") + "\n" + code
    else:
        # 模型重写了函数: 补回 prompt 里的 import (如 List/Optional)
        imports = re.findall(r"^(?:from [\w.]+ import .+|import [\w.]+)$", row["prompt"], re.M)
        missing = [i for i in imports if i not in code]
        if missing:
            code = "\n".join(missing) + "\n\n" + code
    return code


def run_humaneval(port, row, domain, timeout=8, n_predict_cap=512):
    prompt, n_predict, stops = build_prompt("humaneval", row, n_predict_cap)
    content, tim = gen(port, prompt, n_predict, stops, domain)
    code = extract_code(row, content)
    prog = code + "\n\n" + row["test"] + f"\n\ncheck({row['entry_point']})\n"
    ok = False
    err = ""
    # 不用 TemporaryDirectory: 其 __exit__ 清理失败 (WinError 145) 会中断整批
    td = tempfile.mkdtemp(prefix="he_")
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
    return content, code, ok, err, tim


NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_number(text):
    tail = text.split("####")[-1] if "####" in text else text
    nums = NUM.findall(tail.replace(",", ""))
    if not nums:
        nums = NUM.findall(text.replace(",", ""))
    if not nums:
        return None
    s = nums[-1].rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def run_gsm8k(port, row, domain, n_predict_cap=512):
    prompt, n_predict, stops = build_prompt("gsm8k", row, n_predict_cap)
    content, tim = gen(port, prompt, n_predict, stops, domain)
    pred = extract_number(content)
    gold = float(row["gold"])
    ok = pred is not None and abs(pred - gold) < 1e-4
    return content, pred, ok, tim


def run(args):
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, f"e17_{args.tag}_{args.task}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for l in f:
                try:
                    done.add(json.loads(l)["task_id"])
                except Exception:
                    pass
    tasks = load_tasks(args.task)
    if args.limit:
        tasks = tasks[: args.limit]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        tasks = [t for t in tasks if t["task_id"] in want]
    if args.trunc_from:
        ref = os.path.join(RES, f"e17_{args.trunc_from}_{args.task}.jsonl")
        cut = set()
        with open(ref, encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                if not r.get("passed") and ((r.get("timings") or {}).get("predicted_n") or 0) >= 512:
                    cut.add(r["task_id"])
        tasks = [t for t in tasks if t["task_id"] in cut]
        print(f"[{args.tag}/{args.task}] 截断补跑 {len(tasks)} 条 (来自 {args.trunc_from})", flush=True)
    todo = [t for t in tasks if t["task_id"] not in done]
    if not todo:
        print(f"[{args.tag}/{args.task}] 已完成 {len(done)} 条, 无需运行")
        return

    sys.path.insert(0, HERE)
    from pin import pin_self
    cpus = pin_self(cpus=args.cpus) if args.cpus else pin_self(numa=0)
    print(f"[{args.tag}/{args.task}] 绑定 CPU {cpus}", flush=True)

    bin_dir = os.path.abspath(os.path.join(HERE, "..", "llama.cpp", "build-tcmoe",
                                           "bin", "Release"))
    log_path = os.path.join(RES, f"e17_{args.tag}_{args.task}.server.log")
    t0 = time.time()
    proc, log = start_server(bin_dir, os.path.abspath(args.model), args.port,
                             args.threads, args.ctx, log_path, args.extra)
    n_ok = 0
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            for i, row in enumerate(todo):
                ts = time.time()
                try:
                    if args.task == "humaneval":
                        content, code, ok, err, tim = run_humaneval(
                            args.port, row, args.domain, n_predict_cap=args.n_predict)
                        rec = {"task_id": row["task_id"], "passed": ok, "error": err,
                               "completion": content, "code": code}
                    else:
                        content, pred, ok, tim = run_gsm8k(
                            args.port, row, args.domain, n_predict_cap=args.n_predict)
                        rec = {"task_id": row["task_id"], "passed": ok, "pred": pred,
                               "gold": row["gold"], "completion": content}
                except Exception as e:  # noqa: BLE001  单题异常不中断整批, 留待续跑
                    print(f"  [skip] {row['task_id']}: {e!r}", flush=True)
                    continue
                rec.update({"domain": args.domain, "gen_ms": round((time.time() - ts) * 1000),
                            "timings": tim})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += int(ok)
                if i < 3 or (i + 1) % 20 == 0 or i + 1 == len(todo):
                    el = time.time() - t0
                    print(f"  {i+1}/{len(todo)} {row['task_id']} pass={ok} "
                          f"acc={n_ok/(i+1):.3f} {rec['gen_ms']/1000:.1f}s elapsed={el/60:.0f}m",
                          flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    print(f"[{args.tag}/{args.task}] 新增 {len(todo)} 条, 本批 acc={n_ok/len(todo):.3f}, "
          f"用时 {(time.time()-t0)/60:.1f} min", flush=True)


def score():
    files = sorted(x for x in os.listdir(RES) if x.startswith("e17_") and x.endswith(".jsonl")
                   and "server" not in x)
    print(f"{'tag':22s} {'task':10s} {'n':>4s} {'acc':>7s} {'avg_s':>7s}")
    rows = []
    for fn in files:
        tag, task = fn[len("e17_"):-len(".jsonl")].rsplit("_", 1)
        n = ok = 0
        tot = 0.0
        with open(os.path.join(RES, fn), encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                n += 1
                ok += int(bool(r.get("passed")))
                tot += r.get("gen_ms", 0) / 1000
        if n:
            print(f"{tag:22s} {task:10s} {n:4d} {ok/n:7.3f} {tot/n:7.1f}")
            rows.append({"tag": tag, "task": task, "n": n, "acc": ok / n,
                         "avg_s": tot / n})
    if rows:
        with open(os.path.join(RES, "e17_summary.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


def main(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("score")
    p = sub.add_parser("run")
    p.add_argument("--tag", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=["humaneval", "gsm8k"])
    p.add_argument("--domain", default=None)
    p.add_argument("--port", type=int, default=18081)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--cpus", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", default=None, help="逗号分隔的 task_id 子集, 用于补跑")
    p.add_argument("--trunc-from", default=None, help="自动选取该 tag 中 512 上限截断的失败条目")
    p.add_argument("--n-predict", type=int, default=512, help="生成 token 上限, 默认 512")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "score":
        score()
    else:
        run(a)


if __name__ == "__main__":
    main(sys.argv[1:])
