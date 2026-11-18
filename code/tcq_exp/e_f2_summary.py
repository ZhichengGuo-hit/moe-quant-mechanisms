"""E-F2 汇总: 配对 accuracy + 1000 次序列级 bootstrap + McNemar + 功效。

输入: results/e_f2_gsm8k_full.jsonl (e_f2_gsm8k.py 产物)
输出: results/e_f2_summary.json
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SRC = os.path.join(RES, "e_f2_gsm8k_full.jsonl")
DST = os.path.join(RES, "e_f2_summary.json")
TAGS = ("iq2xxs", "tcmoe_nl_global")
CAP = 1024
N_BOOT = 1000
SEED = 20261104


def load():
    per = {t: {} for t in TAGS}
    if not os.path.exists(SRC):
        return per
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r["tag"] in per:
                per[r["tag"]][r["task_id"]] = r
    return per


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def binom_two_sided(b, c):
    """McNemar 精确二项检验 (H0: 不一致对半)。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    per = load()
    ids = sorted(set(per["iq2xxs"]) & set(per["tcmoe_nl_global"]))
    if not ids:
        print("[e_f2_summary] 没有配对数据")
        return

    a = [int(bool(per["iq2xxs"][i]["passed"])) for i in ids]
    b_ = [int(bool(per["tcmoe_nl_global"][i]["passed"])) for i in ids]
    n = len(ids)
    acc_a = sum(a) / n
    acc_b = sum(b_) / n
    b01 = sum(1 for x, y in zip(a, b_) if x and not y)   # iq 对 / nl 错
    b10 = sum(1 for x, y in zip(a, b_) if not x and y)   # iq 错 / nl 对
    p_mcnemar = binom_two_sided(b01, b10)

    rnd = random.Random(SEED)
    d_acc, d_diff = [], []
    for _ in range(N_BOOT):
        idx = [rnd.randrange(n) for _ in range(n)]
        sa = sum(a[i] for i in idx) / n
        sb = sum(b_[i] for i in idx) / n
        d_acc.append(sa)
        d_diff.append(sa - sb)

    psi = (b01 + b10) / n
    mde_pt = (1.959964 + 0.841621) * math.sqrt(psi / n) * 100 if psi > 0 else None

    def cap_stats(tag):
        recs = per[tag]
        pn = [r.get("predicted_n") or 0 for r in recs.values()]
        hit = sum(1 for v in pn if v >= CAP)
        return {"n": len(pn), "predicted_n_mean": round(sum(pn) / len(pn), 1) if pn else None,
                "predicted_n_median": pct(pn, 0.5),
                "capped_at": CAP, "capped_frac": round(hit / len(pn), 4) if pn else None}

    out = {
        "protocol": {"task": "gsm8k", "items": n, "n_predict": CAP, "greedy": True,
                     "decode": "two llama-servers (-t 8 each, affinity 0-7 / 8-15), "
                               "per-question interleaved, order alternates by question parity",
                     "bootstrap": {"n": N_BOOT, "unit": "question (sequence-level)", "seed": SEED}},
        "tags": list(TAGS),
        "accuracy": {"iq2xxs": round(acc_a, 4), "tcmoe_nl_global": round(acc_b, 4),
                     "diff_pt": round((acc_a - acc_b) * 100, 2)},
        "bootstrap_ci95": {},
        "mcnemar": {"iq_only_pass": b01, "nl_only_pass": b10, "p_value": round(p_mcnemar, 6),
                    "discordant_rate": round(psi, 4)},
        "power": {"mde_pt_approx": round(mde_pt, 2) if mde_pt else None,
                  "formula": "delta ~ (z_.975 + z_.80) * sqrt(psi/n), psi = discordant rate"},
        "capped_record": {t: cap_stats(t) for t in TAGS},
    }

    # 补齐 nl 的 acc CI 与配对数
    d_b = []
    rnd2 = random.Random(SEED + 1)
    for _ in range(N_BOOT):
        idx = [rnd2.randrange(n) for _ in range(n)]
        d_b.append(sum(b_[i] for i in idx) / n)
    out["bootstrap_ci95"] = {
        "iq2xxs": [round(pct(d_acc, 0.025), 4), round(pct(d_acc, 0.975), 4)],
        "tcmoe_nl_global": [round(pct(d_b, 0.025), 4), round(pct(d_b, 0.975), 4)],
        "diff_pt": [round(pct(d_diff, 0.025) * 100, 2), round(pct(d_diff, 0.975) * 100, 2)],
    }

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"pairs n={n}")
    print(f"  iq2xxs           {acc_a:.4f}  CI {out['bootstrap_ci95']['iq2xxs']}")
    print(f"  tcmoe_nl_global  {acc_b:.4f}  CI {out['bootstrap_ci95']['tcmoe_nl_global']}")
    print(f"  diff             {(acc_a - acc_b) * 100:+.2f} pt  "
          f"CI {out['bootstrap_ci95']['diff_pt']}")
    print(f"  McNemar b={b01} c={b10} p={p_mcnemar:.4f}  MDE~{mde_pt:.2f} pt")
    print(f"[saved] {DST}")


if __name__ == "__main__":
    main()
