"""E20: 构建 per-domain imatrix 校准文本（math/code/general/global）。

约定：与 E17 评测集不重叠。
- math：GSM8K train（评测用 test 0:200，天然不重叠）
- code：本地 Python 源码（kt_env site-packages），不用 HumanEval，避免与评测题泄漏
- general：Alpaca 3000:3200（评测未用 Alpaca）
- global：三域等量交错，作为同流水线的对照
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "data_extract")
OUT = os.path.join(HERE, "data_e20")


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_math():
    d = os.path.join(EX, "grade-school-math-master", "grade_school_math", "data")
    rows = list(read_jsonl(os.path.join(d, "train.jsonl")))[:400]
    return [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows]


def build_code():
    root = r"E:\CUDA11.8\train\kt_env\Lib\site-packages"
    files = []
    for dp, _, fns in os.walk(root):
        if os.sep + "pip" + os.sep in dp + os.sep:
            continue
        for fn in fns:
            if fn.endswith(".py"):
                p = os.path.join(dp, fn)
                try:
                    if os.path.getsize(p) > 6000:
                        files.append(p)
                except OSError:
                    pass
    random.Random(0).shuffle(files)
    texts, tot = [], 0
    for p in files:
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()[:8000]
        except OSError:
            continue
        texts.append(f"# file: {os.path.basename(p)}\n{t}")
        tot += len(t)
        if tot > 90000:
            break
    return texts


def build_general():
    rows = json.load(open(os.path.join(EX, "stanford_alpaca-main", "alpaca_data.json"),
                          encoding="utf-8"))[3000:3200]

    def fmt(r):
        s = r["instruction"]
        if r.get("input"):
            s += "\n" + r["input"]
        return s + "\n" + r["output"]

    return [fmt(r) for r in rows]


def write(name, texts):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))
    n_chars = os.path.getsize(p)
    print(f"{name}: {len(texts)} chunks, {n_chars} chars, ~{n_chars // 4} tokens")


def main():
    os.makedirs(OUT, exist_ok=True)
    m, c, g = build_math(), build_code(), build_general()
    write("math.txt", m)
    write("code.txt", c)
    write("general.txt", g)
    gl = []
    for i in range(min(len(m), len(c), len(g))):
        gl += [m[i], c[i], g[i]]
    write("global.txt", gl)


main()
