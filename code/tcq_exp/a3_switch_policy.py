"""A3: lift 自适应换装决策表 (依赖 B6/B4 的域对 lift 与 CI)。

规则 (以 E13/E15 的可用结论为校准锚点):
  - 换装能带来显著域内增益的前提是该域对"确实可分离": lift 低 (<1.6) 且 Spearman <0.35
  - lift 高 (>3.0) 表示两域 top-20% 专家高度混同 -> 换装无意义, 直接共享配额
  - 中间区间 -> 条件换装 (仅当 CI 上界 < 阈值且预测增益可覆盖切换成本)
输出: results/a3_decision.json + 决策表
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T

LIFT_SWITCH = 1.6
LIFT_SHARE = 3.0


def load_pairs():
    rows = []
    with open(os.path.join(T.RES_DIR, "b6_lift.json"), encoding="utf-8") as f:
        b6 = json.load(f)
    for tag, blob in b6["models"].items():
        for pair, v in blob["pairs"].items():
            rows.append({"model": tag, "signal": blob["signal"], "pair": pair,
                         "jaccard": v["jaccard_pooled"], "lift": v["lift"],
                         "lift_ci95": v["lift_ci95"],
                         "spearman": v["spearman_pooled"],
                         "spearman_ci95": v["spearman_ci95"]})
    p4 = os.path.join(T.RES_DIR, "b4_domain_pairs.json")
    if os.path.exists(p4):
        with open(p4, encoding="utf-8") as f:
            b4 = json.load(f)
        for pair, v in b4["h1"].items():
            rows.append({"model": b4["model"], "signal": "route", "pair": pair,
                         "jaccard": v["jaccard"], "lift": v["lift"],
                         "lift_ci95": [v["jaccard_ci95"][0] / 0.111111,
                                       v["jaccard_ci95"][1] / 0.111111],
                         "spearman": v["spearman"],
                         "spearman_ci95": v["spearman_ci95"]})
    return rows


def decide(r):
    # 以 lift 为主, 结合 H1 的 Jaccard/Spearman 阈值
    lo, hi = r["lift_ci95"]
    if hi <= LIFT_SWITCH and r["spearman"] < 0.35:
        return "switch"
    if r["lift"] >= LIFT_SHARE or lo >= LIFT_SWITCH:
        return "share"
    return "conditional"


def main():
    rows = load_pairs()
    table = []
    for r in rows:
        d = decide(r)
        table.append({**r, "policy": d})
    n = {k: sum(1 for t in table if t["policy"] == k)
         for k in ("switch", "conditional", "share")}
    out = {"thresholds": {"lift_switch": LIFT_SWITCH, "lift_share": LIFT_SHARE,
                          "spearman_max_switch": 0.35},
           "counts": n, "rows": table}
    p = os.path.join(T.RES_DIR, "a3_decision.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[a3] saved {p} counts={n}", flush=True)
    for t in table:
        print(f"  {t['model']:<18} {t['signal']:<6} {t['pair']:<18} "
              f"J={t['jaccard']:.3f} lift={t['lift']:.2f} "
              f"CI[{t['lift_ci95'][0]:.2f},{t['lift_ci95'][1]:.2f}] "
              f"rho={t['spearman']:.3f} -> {t['policy']}", flush=True)


if __name__ == "__main__":
    main()
