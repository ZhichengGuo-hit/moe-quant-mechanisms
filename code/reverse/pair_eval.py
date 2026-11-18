"""E-R1 配对统计: McNemar 精确 + 1000 次序列级 bootstrap (seed 20261104)。

    python pair_eval.py --target results/c2_reverse_tasks.jsonl \
        --baseline g2=../tcq_exp/results/c1_g2_tasks.jsonl \
        --baseline grand=../tcq_exp/results/c1_grand_tasks.jsonl \
        --baseline g1=../tcq_exp/results/e_h1_g1_im_{task}.jsonl \
        --out results/c2_reverse_pairwise.json

基线可写 {task} 占位, 按任务分别加载 (G1 的逐任务文件)。
判定只看 McNemar p 与 bootstrap CI, 不看点估计方向。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

SEED = 20261104
N_BOOT = 1000
TASKS = ("gsm8k", "humaneval")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def load(path):
    """-> {(task, task_id): bool}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            task = r.get("task")
            if not task:
                base = os.path.basename(path)
                task = "humaneval" if "humaneval" in base else "gsm8k"
            ok = r.get("acc", r.get("passed"))
            out[(task, str(r["task_id"]))] = bool(ok)
    return out


def load_spec(spec):
    label, path = spec.split("=", 1)
    if "{task}" in path:
        merged = {}
        for t in TASKS:
            p = path.replace("{task}", t)
            if os.path.exists(p):
                merged.update(load(p))
        return label, merged
    return label, load(path)


def compare(base, tgt, rng):
    per = {}
    allb = allc = alln = 0
    for task in TASKS:
        ids = sorted(k for k in tgt if k[0] == task and k in base)
        if not ids:
            continue
        b = sum(1 for k in ids if base[k] and not tgt[k])
        c = sum(1 for k in ids if tgt[k] and not base[k])
        acc_b = sum(base[k] for k in ids) / len(ids)
        acc_t = sum(tgt[k] for k in ids) / len(ids)
        arr_b = np.array([base[k] for k in ids], dtype=np.float64)
        arr_t = np.array([tgt[k] for k in ids], dtype=np.float64)
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, len(ids), len(ids))
            diffs[i] = arr_t[idx].mean() - arr_b[idx].mean()
        per[task] = {
            "paired_n": len(ids), "b": b, "c": c, "p": mcnemar_exact(b, c),
            "acc_baseline": acc_b, "acc_target": acc_t, "acc_diff": acc_t - acc_b,
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "significant": bool(mcnemar_exact(b, c) < 0.05
                                and (np.percentile(diffs, 2.5) > 0
                                     or np.percentile(diffs, 97.5) < 0)),
        }
        allb += b
        allc += c
        alln += len(ids)
    out = {"per_task": per}
    if alln:
        out["combined"] = {"paired_n": alln, "b": allb, "c": allc,
                           "p": mcnemar_exact(allb, allc)}
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--baseline", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tgt = load(a.target)
    rng = np.random.default_rng(SEED)
    res = {"target": a.target, "n_target": len(tgt),
           "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "item"},
           "test": "McNemar exact (two-sided binomial) + bootstrap CI for acc diff"}
    for spec in a.baseline:
        label, base = load_spec(spec)
        res["vs_" + label] = compare(base, tgt, rng)
    p = a.out if os.path.isabs(a.out) else os.path.join(os.path.dirname(__file__), a.out)
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in res.items():
        if not isinstance(v, dict) or "per_task" not in v:
            continue
        print("[%s]" % k)
        for task, r in v["per_task"].items():
            print("  %-9s n=%3d b=%2d c=%2d p=%.4f acc %.3f -> %.3f (%+.3f) "
                  "CI[%+.3f,%+.3f] %s" % (
                      task, r["paired_n"], r["b"], r["c"], r["p"],
                      r["acc_baseline"], r["acc_target"], r["acc_diff"],
                      r["ci95"][0], r["ci95"][1],
                      "SIG" if r["significant"] else "ns"))
    print("[out]", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
