"""阶段 2 数据构建:
  B4: 翻译域 trans / 摘要域 summ (来自 Alpaca)
  C2: 同源题库对 (math_alpaca vs prose_alpaca; code vs code_desc, 同一 HumanEval 题面)
  C3: 每域 eval128 (math/general 取未用过的样本; code 用 HumanEval 36:164)
输出 data/*.jsonl, 每行 {"text":..., "n_tok":..., 可选 "subtype"/"label"}
"""
import os
import json
import gzip
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "data_extract")
OUT = os.path.join(HERE, "data")
MODEL_PATH = r"E:\CUDA11.8\model\Qwen3.6-35B-A3B"

N_CALIB = 24
N_EVAL = 16
N_TEST32 = 32
N128 = 128
MAX_TRAIN = 96
MAX_EVAL = 64


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def alpaca_rows():
    return json.load(open(os.path.join(EX, "stanford_alpaca-main", "alpaca_data.json"),
                          encoding="utf-8"))


def fmt(r):
    s = r["instruction"]
    if r.get("input"):
        s += "\n" + r["input"]
    return s + "\n" + r["output"]


def humaneval_rows():
    p = os.path.join(EX, "human-eval-master", "data", "HumanEval.jsonl.gz")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def subtype(instruction):
    s = instruction.lower()
    if any(k in s for k in ("translate", "translation")):
        return "translate"
    if any(k in s for k in ("summariz", "summary", "tl;dr")):
        return "summarize"
    if any(k in s for k in ("explain", "why", "what is", "difference between")):
        return "explain"
    if any(k in s for k in ("write", "compose", "create", "generate", "draft")):
        return "generate"
    if any(k in s for k in ("classify", "identify", "extract", "label", "determine")):
        return "extract"
    return "other"


def main():
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def trim(text, n):
        ids = tok(text, add_special_tokens=False).input_ids[:n]
        return tok.decode(ids, skip_special_tokens=True), len(ids)

    def dump(name, rows, max_len, extra=None):
        out = []
        for i, t in enumerate(rows):
            txt, n = trim(t, max_len)
            r = {"text": txt, "n_tok": n}
            if extra:
                r.update(extra(i))
            out.append(r)
        p = os.path.join(OUT, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{name}] n={len(out)} avg_tok={sum(r['n_tok'] for r in out)/len(out):.1f}",
              flush=True)

    al = alpaca_rows()
    trans = [fmt(r) for r in al
             if any(k in r["instruction"].lower() for k in ("translate", "translation"))]
    summ = [fmt(r) for r in al
            if any(k in r["instruction"].lower() for k in ("summariz", "summary"))]
    trans = [t for t in trans if len(t) > 80]
    summ = [t for t in summ if len(t) > 80]
    assert len(trans) >= N_CALIB + N_EVAL and len(summ) >= N_CALIB + N_EVAL
    dump("trans_calib", trans[:N_CALIB], MAX_TRAIN)
    dump("trans_eval", trans[N_CALIB:N_CALIB + N_EVAL], MAX_EVAL)
    dump("summ_calib", summ[:N_CALIB], MAX_TRAIN)
    dump("summ_eval", summ[N_CALIB:N_CALIB + N_EVAL], MAX_EVAL)

    # ---- C2: 同源对 A: Alpaca 数学题 vs 非数学散文 ----
    math_re = re.compile(r"[0-9]")
    math_kw = ("calculate", "how many", "total", "average", "cost", "price",
               "percent", "sum", "add", "subtract", "multiply", "divide")
    is_math = [bool(math_re.search(r["instruction"]) and
                    any(k in r["instruction"].lower() for k in math_kw)) for r in al]
    math_rows = [fmt(al[i]) for i in range(len(al)) if is_math[i]]
    prose_rows = [fmt(al[i]) for i in range(len(al))
                  if not is_math[i] and len(al[i]["instruction"]) > 40]
    math_rows = [t for t in math_rows if len(t) > 80]
    prose_rows = [t for t in prose_rows if len(t) > 80]
    need = N_CALIB + N_TEST32
    assert len(math_rows) >= need and len(prose_rows) >= need
    dump("samemath_calib", math_rows[:N_CALIB], MAX_TRAIN)
    dump("samemath_test", math_rows[N_CALIB:need], MAX_EVAL)
    dump("prose_calib", prose_rows[:N_CALIB], MAX_TRAIN)
    dump("prose_test", prose_rows[N_CALIB:need], MAX_EVAL)

    # ---- C2: 同源对 B: HumanEval 代码 vs 同一题的题面描述 ----
    he = humaneval_rows()
    code_all = [r["prompt"] + r["canonical_solution"] for r in he]
    desc_all = [r["prompt"] for r in he]
    dump("hecode_calib", code_all[:N_CALIB], MAX_TRAIN)
    dump("hecode_test", code_all[N_CALIB:need], MAX_EVAL)
    dump("codespec_calib", desc_all[:N_CALIB], MAX_TRAIN)
    dump("codespec_test", desc_all[N_CALIB:need], MAX_EVAL)

    # ---- C3: eval128 ----
    # math / general: 取 eval 池 80:208 (与 calib/eval/eval64 不重叠)
    math_raw = read_jsonl(os.path.join(EX, "grade-school-math-master", "grade_school_math",
                                       "data", "test.jsonl"))
    math_texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in math_raw]
    dump("math_eval128", math_texts[80:80 + N128], MAX_EVAL)

    gen_pairs = [(fmt(r), subtype(r["instruction"])) for r in al]
    gen_pairs = [p for p in gen_pairs if len(p[0]) > 80]
    gp = gen_pairs[80:80 + N128]
    gt = [p[0] for p in gp]
    gsub = [p[1] for p in gp]
    dump("general_eval128", gt, MAX_EVAL, extra=lambda i: {"subtype": gsub[i]})
    from collections import Counter
    print("[general subtypes]", Counter(gsub), flush=True)

    dump("code_eval128", code_all[36:36 + N128], MAX_EVAL)


if __name__ == "__main__":
    main()
