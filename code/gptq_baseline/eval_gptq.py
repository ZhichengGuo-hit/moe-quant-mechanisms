"""E-GPTQ 评测：GPTQ-3bit（HF 格式）在 GSM8K 200 + HumanEval 164 上。

    python eval_gptq.py --model out/qwen36-3bit-gptq --task gsm8k
    python eval_gptq.py --model out/qwen36-3bit-gptq --task humaneval

复用 E17 的 prompt 模板与判分（extract_number / extract_code / 子进程执行测试），
greedy、1024 上限、逐题落 jsonl、断点续跑。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "tcq_exp")))
from e17_task_eval import build_prompt, extract_code, extract_number, load_tasks  # noqa: E402

N_PREDICT = 1024
STOPS = ["\n\nQuestion:", "\n\nQ:"]


def clean(content, stops):
    for s in stops:
        i = content.find(s)
        if i >= 0:
            content = content[:i]
    return content.strip()


def run_he(row, code, timeout=8):
    prog = code + "\n\n" + row["test"] + f"\n\ncheck({row['entry_point']})\n"
    td = tempfile.mkdtemp(prefix="he_")
    try:
        p = os.path.join(td, "sol.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(prog)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, timeout=timeout,
                               capture_output=True, text=True)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default=r"E:\CUDA11.8\model\Qwen3.6-35B-A3B")
    ap.add_argument("--task", choices=["gsm8k", "humaneval"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    try:
        from gptqmodel import GPTQModel
        from transformers import AutoTokenizer
    except Exception as e:  # noqa: BLE001
        print("依赖不可用: %r（见 PREREG.md §4）" % (e,))
        return 2

    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, "gptq_%s.jsonl" % args.task)
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["task_id"])
                except Exception:  # noqa: BLE001
                    pass
    rows = load_tasks(args.task)
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = GPTQModel.load(args.model, device_map="auto")
    model.eval()
    n_ok = 0
    n = 0
    with open(out, "a", encoding="utf-8") as f:
        for row in rows:
            if row["task_id"] in done:
                continue
            prompt, mx, stops = build_prompt(args.task, row, N_PREDICT)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            gen = model.generate(**ids, max_new_tokens=min(mx, N_PREDICT), do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            text = clean(text, stops)
            if args.task == "gsm8k":
                pred = extract_number(text)
                ok = pred is not None and abs(pred - float(row["gold"])) < 1e-4
                rec = {"task_id": row["task_id"], "predicted_n": int(gen.shape[1] - ids["input_ids"].shape[1]),
                       "pred": pred, "gold": row["gold"], "passed": bool(ok), "content": text}
            else:
                code = extract_code(row, text)
                ok = run_he(row, code)
                rec = {"task_id": row["task_id"], "passed": bool(ok), "content": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            n_ok += int(ok)
            if n % 10 == 0:
                print("[%s] %d done, acc=%.3f" % (args.task, n, n_ok / n), flush=True)
    print("total %d, acc %.4f -> %s" % (n, n_ok / max(n, 1), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
