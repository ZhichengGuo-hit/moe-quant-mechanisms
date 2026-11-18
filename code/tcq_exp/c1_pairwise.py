#!/usr/bin/env python
"""C1: 逐题配对 McNemar (精确二项, 与 e_g1_eval/e_h1_summary 同口径) 汇总。

    python c1_pairwise.py --target results/c1_g2_tasks.jsonl \\
        --baseline-prefix e_h1_g1_im --out results/c1_g2_pairwise.json \\
        --extra g2_im=results/c1_g2_tasks.jsonl

--target / --extra 为含 task 字段的逐题 jsonl; --baseline-prefix 为 results/ 下按任务
拆分的 e_h1/e17 结果前缀 (读 {prefix}_{task}.jsonl)。
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
from e_g1_eval import mcnemar_exact  # noqa: E402  同一实现 (2*sum(C(n,i), i<=min(b,c))/2^n)

TASKS = ("gsm8k", "humaneval")


def load_any(spec):
    """返回 {task: {task_id: bool}}。spec = 逐题 jsonl 路径 或 results/ 下的前缀。"""
    out = {}
    if spec.endswith(".jsonl"):
        path = spec if os.path.isabs(spec) else os.path.join(HERE, spec)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                task = r.get("task") or r["task_id"].split("-")[0]
                if task.startswith("HumanEval"):
                    task = "humaneval"
                val = r.get("acc")
                if val is None:
                    val = r.get("passed")
                out.setdefault(task, {})[r["task_id"]] = bool(val)
        return out
    for task in TASKS:
        path = os.path.join(RES, "%s_%s.jsonl" % (spec, task))
        if not os.path.exists(path):
            continue
        d = out.setdefault(task, {})
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                d.setdefault(r["task_id"], bool(r.get("passed") or r.get("acc")))
    return out


def compare(base, other):
    res = {"per_task": {}, "combined": {}}
    all_b, all_c, all_n = 0, 0, 0
    for task in TASKS:
        B = base.get(task, {})
        O = other.get(task, {})
        ids = sorted(set(B) & set(O))
        if not ids:
            continue
        b = sum(1 for i in ids if B[i] and not O[i])   # baseline 对 / 目标 错
        c = sum(1 for i in ids if not B[i] and O[i])   # baseline 错 / 目标 对
        acc_b = sum(B[i] for i in ids) / len(ids)
        acc_o = sum(O[i] for i in ids) / len(ids)
        res["per_task"][task] = {
            "paired_n": len(ids), "b": b, "c": c,
            "p": mcnemar_exact(b, c),
            "acc_baseline": acc_b, "acc_target": acc_o, "acc_diff": acc_o - acc_b,
        }
        all_b += b
        all_c += c
        all_n += len(ids)
    if all_n:
        res["combined"] = {
            "paired_n": all_n, "b": all_b, "c": all_c,
            "p": mcnemar_exact(all_b, all_c),
            "acc_baseline": (sum(sum(v.values()) for v in base.values())
                             / sum(len(v) for v in base.values())),
            "acc_target": (sum(sum(v.values()) for v in other.values())
                           / sum(len(v) for v in other.values())),
        }
        res["combined"]["acc_diff"] = (res["combined"]["acc_target"]
                                       - res["combined"]["acc_baseline"])
    return res


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--baseline-prefix", default="e_h1_g1_im")
    ap.add_argument("--label-target", default="")
    ap.add_argument("--label-baseline", default="g1_im")
    ap.add_argument("--extra", action="append", default=[],
                    help="label=jsonl, 追加对照 (例如 g2_im=results/c1_g2_tasks.jsonl)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    base = load_any(a.baseline_prefix)
    target = load_any(a.target)
    out = {"target": a.label_target or os.path.basename(a.target),
           "baseline": a.label_baseline,
           "test": "McNemar exact (two-sided binomial), same-task pairing",
           "baseline_source": a.baseline_prefix,
           "target_source": a.target,
           "vs_baseline": compare(base, target)}
    for spec in a.extra:
        label, path = spec.split("=", 1)
        other = load_any(path)
        out.setdefault("supplementary_vs_" + label, compare(other, target))
        out["supplementary_vs_" + label]["label"] = label

    path = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in out.items():
        if not isinstance(v, dict) or ("per_task" not in v):
            continue
        print("[%s] %s vs %s" % (k, out["target"], v.get("label", a.label_baseline)))
        for task, r in v["per_task"].items():
            print("  %-10s n=%3d b=%3d c=%3d p=%.4f acc %.3f -> %.3f (%+.3f)" % (
                task, r["paired_n"], r["b"], r["c"], r["p"],
                r["acc_baseline"], r["acc_target"], r["acc_diff"]))
        r = v["combined"]
        if r:
            print("  %-10s n=%3d b=%3d c=%3d p=%.4f acc %.3f -> %.3f (%+.3f)" % (
                "combined", r["paired_n"], r["b"], r["c"], r["p"],
                r["acc_baseline"], r["acc_target"], r["acc_diff"]))
    print("[write]", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
