"""E-H2: TOST 等价检验 (零机时, 复用已有配对数据)。

预设等价界值 (投稿前写死):
  GSM8K ±1.5pt, HumanEval ±3pt, GPQA ±6pt

判定口径:
  90% 配对 bootstrap CI 与 [-m, +m] 的关系
    CI 完全落在界内            -> equivalent   ("统计等价"成立)
    CI 完整落在界外            -> not_equivalent (差异超出界值)
    跨越边界                    -> inconclusive (未检测到差异, 但功效不足以宣称等价)

  GSM8K 用 F2 全量 1319 题 (iq2xxs vs tcmoe_nl_global);
  HE   用 e17 的 iq2xxs vs tcmoe_nl_math (164 题);
  GPQA 本机无 Qwen3.6 配对数据 -> 记为 no_data (功效不足, 不作等价声明)。

用法:
    python e_h2_tost.py
输出:
    results/e_h2_tost.json
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
N_BOOT = 10000
SEED = 20261104

PAIRS = [
    {
        "task": "gsm8k",
        "items": 1319,
        "bound_pt": 1.5,
        "a": {"tag": "iq2xxs", "file": "e_f2_gsm8k_full.jsonl"},
        "b": {"tag": "tcmoe_nl_global", "file": "e_f2_gsm8k_full.jsonl"},
        "note": "F2 全量 1319 题, 逐题交错生成",
    },
    {
        "task": "humaneval",
        "items": 164,
        "bound_pt": 3.0,
        "a": {"tag": None, "file": "e17_iq2xxs_humaneval.jsonl"},
        "b": {"tag": None, "file": "e17_tcmoe_nl_math_humaneval.jsonl"},
        "note": "e17 三域对照中的 nl_math 档",
    },
    {
        "task": "gpqa",
        "items": 100,
        "bound_pt": 6.0,
        "a": {"tag": "iq2xxs", "file": ""},
        "b": {"tag": "tcmoe_nl_global", "file": ""},
        "note": "本机无 Qwen3.6 GPQA 配对产物; 100 题下 90%CI 约 ±9pt, 已超 ±6pt 界值",
    },
]


def load_pairs(a_file, a_tag, b_file, b_tag):
    recs = {}
    for which, path, tag in (("a", a_file, a_tag), ("b", b_file, b_tag)):
        with open(os.path.join(RES, path), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                # e17 时代的 jsonl 没有 tag 字段, 文件本身即配置标识
                if tag is not None and r.get("tag") != tag:
                    continue
                recs.setdefault(r["task_id"], {})[which] = bool(r["passed"])
    ids = sorted(k for k, v in recs.items() if "a" in v and "b" in v)
    return ids, recs


def pct(x):
    return 100.0 * x


def boot_ci(diffs, lo=5, hi=95, n=N_BOOT, seed=SEED):
    rnd = random.Random(seed)
    n_ids = len(diffs)
    vals = []
    for _ in range(n):
        acc = 0.0
        for _ in range(n_ids):
            acc += diffs[rnd.randrange(n_ids)]
        vals.append(acc / n_ids)
    vals.sort()
    return pct(vals[int(lo / 100.0 * n)]), pct(vals[int(hi / 100.0 * n) - 1])


def mcnemar(b, c):
    """双侧精确二项 McNemar (b, c 为不一致题数)。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
    return min(1.0, p)


def verdict(ci_lo, ci_hi, bound):
    if ci_lo > -bound and ci_hi < bound:
        return "equivalent"
    if ci_lo > bound or ci_hi < -bound:
        return "not_equivalent"
    return "inconclusive"


def run():
    out = {"experiment": "E-H2", "bound_note": "界值在投稿前写死, 不随数据调整",
           "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "question"},
           "pairs": []}

    for p in PAIRS:
        row = {"task": p["task"], "bound_pt": p["bound_pt"], "note": p["note"],
               "a_tag": p["a"]["tag"], "b_tag": p["b"]["tag"]}
        if not p["a"]["file"] or not p["b"]["file"]:
            row.update({"n": 0, "status": "no_data",
                        "verdict": "inconclusive",
                        "statement": "无配对数据; 100 题量级下 90%CI 约 ±9pt, 远超 ±6pt 界值, "
                                     "只能写 '未检测到差异', 不能宣称等价"})
            out["pairs"].append(row)
            continue

        ids, recs = load_pairs(p["a"]["file"], p["a"]["tag"],
                               p["b"]["file"], p["b"]["tag"])
        n = len(ids)
        acc_a = sum(recs[i]["a"] for i in ids) / n
        acc_b = sum(recs[i]["b"] for i in ids) / n
        diffs = [int(recs[i]["b"]) - int(recs[i]["a"]) for i in ids]
        lo, hi = boot_ci(diffs)
        diff = pct(acc_b - acc_a)
        b = sum(1 for i in ids if recs[i]["a"] and not recs[i]["b"])
        c = sum(1 for i in ids if recs[i]["b"] and not recs[i]["a"])
        pval = mcnemar(b, c)
        v = verdict(lo, hi, p["bound_pt"])
        row.update({
            "n": n, "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
            "diff_pt": round(diff, 2), "ci90_pt": [round(lo, 2), round(hi, 2)],
            "discordant": {"a_only": b, "b_only": c}, "mcnemar_p": round(pval, 4),
            "agreement": {"n_same": n - b - c, "frac": round((n - b - c) / n, 4)},
            "verdict": v,
        })
        out["pairs"].append(row)

    g = [r for r in out["pairs"] if r["task"] == "gsm8k"][0]
    if g.get("verdict") == "equivalent":
        out["headline"] = ("GSM8K 全量 1319: diff %+.2fpt, 90%%CI [%+.2f, %+.2f] ⊂ ±%.1f -> "
                           "TOST 通过, '统计等价'可写入论文" %
                           (g["diff_pt"], g["ci90_pt"][0], g["ci90_pt"][1], g["bound_pt"]))

    path = os.path.join(RES, "e_h2_tost.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for r in out["pairs"]:
        print("%-10s n=%-5s diff=%+6.2f CI90=[%+6.2f,%+6.2f] bound=±%.1f -> %s"
              % (r["task"], r.get("n", 0), r.get("diff_pt", 0.0),
                 r.get("ci90_pt", [0, 0])[0], r.get("ci90_pt", [0, 0])[1],
                 r["bound_pt"], r["verdict"]))
    print("saved", path)
    return out


if __name__ == "__main__":
    run()
