#!/usr/bin/env python
"""E-H3a: Xing C vs D 逐题配对（McNemar 精确检验）+ 配对 bootstrap 置信区间。

用法:
  python e_h3_mcnemar.py --a results/e_g3_xing_C_roleq4_gpqa.jsonl \
                         --b results/e_g3_xing_D_roleq4_im_gpqa.jsonl \
                         --out results/e_h3_xing_mcnemar.json
"""
import argparse
import json
import math
import os
import random


def load(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            d[r["task_id"]] = r
    return d


def binom_two_side(k, n, p=0.5):
    if n == 0:
        return 1.0
    lp = sum(math.lgamma(i + 1) for i in range(n + 1))
    probs = [math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                      + i * math.log(p) + (n - i) * math.log(1 - p)) for i in range(n + 1)]
    p0 = probs[k]
    return min(1.0, sum(x for x in probs if x <= p0 + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-a", default="C")
    ap.add_argument("--label-b", default="D")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))
    a = [1 if A[i].get("passed") else 0 for i in ids]
    b = [1 if B[i].get("passed") else 0 for i in ids]
    acc_a = sum(a) / len(ids)
    acc_b = sum(b) / len(ids)
    b01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)   # A 对 B 错
    b10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)   # A 错 B 对
    n_dis = b01 + b10
    p = binom_two_side(min(b01, b10), n_dis) if n_dis else 1.0

    rng = random.Random(args.seed)
    diffs = []
    n = len(ids)
    for _ in range(args.nboot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append((sum(b[i] for i in idx) - sum(a[i] for i in idx)) / n)
    diffs.sort()
    lo, hi = diffs[int(0.05 * args.nboot)], diffs[int(0.95 * args.nboot)]
    out = {"n_pairs": n, "label_a": args.label_a, "label_b": args.label_b,
           "acc_a": acc_a, "acc_b": acc_b,
           "diff_pt": (acc_b - acc_a) * 100,
           "ci90_pt": [lo * 100, hi * 100],
           "b_a_only": b01, "c_b_only": b10, "n_disagree": n_dis,
           "mcnemar_exact_p": p,
           "significant_0.05": bool(p < 0.05),
           "files": {"a": args.a, "b": args.b}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
