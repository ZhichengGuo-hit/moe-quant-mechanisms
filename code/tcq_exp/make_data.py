"""构建三域校准/评测数据: math(GSM8K), code(HumanEval), general(Alpaca)。
输出 tcq_exp/data/{domain}_{calib,eval}.jsonl, 每行 {"text":..., "n_tok":...}
"""
import os
import json
import gzip

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "data_extract")
OUT = os.path.join(HERE, "data")

MODEL_PATH = r"E:\CUDA11.8\model\Qwen3.6-35B-A3B"
N_CALIB = 24
N_EVAL = 16
N_EVAL_LARGE = 64
MAX_CALIB = 96
MAX_EVAL = 64


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_math():
    d = os.path.join(EX, "grade-school-math-master", "grade_school_math", "data")
    calib = [f"Question: {r['question']}\nAnswer: {r['answer']}"
             for r in read_jsonl(os.path.join(d, "train.jsonl"))]
    ev = [f"Question: {r['question']}\nAnswer: {r['answer']}"
          for r in read_jsonl(os.path.join(d, "test.jsonl"))]
    return calib, ev


def build_code():
    p = os.path.join(EX, "human-eval-master", "data", "HumanEval.jsonl.gz")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    texts = [r["prompt"] + r["canonical_solution"] for r in rows]
    # HumanEval 仅 164 条: calib 与 eval 池必须互斥, eval=texts[24:40], eval64=texts[40:104]
    return texts[:N_CALIB], texts[N_CALIB:]


def build_general():
    p = os.path.join(EX, "stanford_alpaca-main", "alpaca_data.json")
    rows = json.load(open(p, encoding="utf-8"))

    def fmt(r):
        s = r["instruction"]
        if r.get("input"):
            s += "\n" + r["input"]
        return s + "\n" + r["output"]

    texts = [fmt(r) for r in rows]
    # 过滤过短样本, 保证域内文本量
    texts = [t for t in texts if len(t) > 80]
    # calib 与 eval 池互斥, 避免校准/评测泄漏
    return texts[:N_CALIB], texts[N_CALIB:]


def main():
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def trim(text, n):
        ids = tok(text, add_special_tokens=False).input_ids[:n]
        return tok.decode(ids, skip_special_tokens=True), len(ids)

    for dom, fn in (("math", build_math), ("code", build_code),
                    ("general", build_general)):
        calib_raw, ev_raw = fn()
        assert (len(calib_raw) >= N_CALIB and
                len(ev_raw) >= N_EVAL + N_EVAL_LARGE), dom
        splits = {
            "calib": (calib_raw[:N_CALIB], MAX_CALIB),
            "eval": (ev_raw[:N_EVAL], MAX_EVAL),
            "eval64": (ev_raw[N_EVAL:N_EVAL + N_EVAL_LARGE], MAX_EVAL),
        }
        for name, (raw, max_len) in splits.items():
            rows_out = []
            for t in raw:
                txt, n = trim(t, max_len)
                rows_out.append({"text": txt, "n_tok": n})
            p = os.path.join(OUT, f"{dom}_{name}.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for r in rows_out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{dom}/{name}] n={len(rows_out)} "
                  f"avg_tok={sum(r['n_tok'] for r in rows_out)/len(rows_out):.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
