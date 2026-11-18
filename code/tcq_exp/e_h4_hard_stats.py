#!/usr/bin/env python
"""E-H4: 困难任务（AIME）诊断统计 —— acc / 终止率 / 输出长度 / 重复率。

数据: results/e_f8_*.jsonl（每行含 tag, passed, finish_reason, predicted_n, completion, n_predict）。
输出: results/e_h4_hard_stats.json
"""
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e_h4_hard_stats.json")


def repeat_metrics(text, n=4):
    words = re.findall(r"\w+", text.lower())
    if len(words) < n * 2:
        return {"max_ngram_rep": 0, "rep_frac": 0.0}
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    cnt = {}
    for g in grams:
        cnt[g] = cnt.get(g, 0) + 1
    mx = max(cnt.values())
    return {"max_ngram_rep": mx, "rep_frac": round(1 - len(cnt) / len(grams), 4)}


def main():
    out = {}
    for path in sorted(glob.glob(os.path.join(RES, "e_f8_*.jsonl"))):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if not rows:
            continue
        tags = sorted({r.get("tag", "?") for r in rows})
        for tag in tags:
            rs = [r for r in rows if r.get("tag", "?") == tag]
            n = len(rs)
            passed = sum(1 for r in rs if r.get("passed"))
            fin = {}
            for r in rs:
                fin[r.get("finish_reason")] = fin.get(r.get("finish_reason"), 0) + 1
            nat = sum(1 for r in rs if r.get("finish_reason") in ("stop", "stop_word", "eos"))
            lens = [r.get("predicted_n", 0) for r in rs]
            comps = [r.get("completion", "") for r in rs]
            reps = [repeat_metrics(c)["max_ngram_rep"] for c in comps]
            repfrac = [repeat_metrics(c)["rep_frac"] for c in comps]
            out["%s|%s" % (os.path.basename(path), tag)] = {
                "n": n, "acc": passed / n, "passed": passed,
                "natural_stop_rate": nat / n,
                "finish_reason_counts": fin,
                "len_mean": statistics.mean(lens), "len_median": statistics.median(lens),
                "len_max": max(lens),
                "rep_max_ngram_mean": statistics.mean(reps),
                "rep_max_ngram_max": max(reps),
                "rep_frac_mean": statistics.mean(repfrac),
                "rep_frac_max": max(repfrac),
            }
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT)
    for k, v in out.items():
        print("%-45s n=%-4d acc=%.3f stop=%.3f len=%.0f repmax=%.1f" % (
            k, v["n"], v["acc"], v["natural_stop_rate"], v["len_mean"], v["rep_max_ngram_mean"]))


if __name__ == "__main__":
    main()
