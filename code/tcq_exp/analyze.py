"""实验分析: H1(Jaccard/Spearman)、代理有效性(ΔL相关)、H4(配额收益) -> results/summary.json"""
import os
import sys
import json
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import tcq_lib as T

FRAC = 0.2


def rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(np.asarray(a, float)), rankdata(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def top_sets(score, frac=FRAC):
    L, E = score.shape
    k = max(1, int(E * frac))
    out = []
    for l in range(L):
        idx = np.argsort(-score[l])[:k]
        out.append(set(int(i) for i in idx))
    return out


def jaccard(setA, setB):
    return len(setA & setB) / max(len(setA | setB), 1)


def main():
    sm = json.load(open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8"))
    s = np.array(sm["s"])
    u = np.array(sm["usage"])
    doms = sm["domains"]

    # ---------- H1: 域间专家集合重叠 ----------
    h1 = {"impact": {}, "intrinsic": {}, "usage": {}}
    sets_imp = {d: top_sets(s[:, :, i]) for i, d in enumerate(doms)}
    eff = np.where(u > 0, s / np.maximum(u, 1e-9), 0.0)
    sets_int = {d: top_sets(eff[:, :, i]) for i, d in enumerate(doms)}
    sets_use = {d: top_sets(u[:, :, i]) for i, d in enumerate(doms)}

    for a, b in itertools.combinations(doms, 2):
        js_imp = [jaccard(sets_imp[a][l], sets_imp[b][l]) for l in range(s.shape[0])]
        js_int = [jaccard(sets_int[a][l], sets_int[b][l]) for l in range(s.shape[0])]
        js_use = [jaccard(sets_use[a][l], sets_use[b][l]) for l in range(s.shape[0])]
        # 全矩阵 Spearman (仅两域都有使用的专家)
        common = (u[:, :, doms.index(a)] > 0) & (u[:, :, doms.index(b)] > 0)
        rho_imp = spearman(s[:, :, doms.index(a)][common], s[:, :, doms.index(b)][common])
        h1["impact"][f"{a}-{b}"] = {"jaccard": float(np.mean(js_imp)),
                                    "jaccard_std": float(np.std(js_imp)),
                                    "spearman": rho_imp, "n_common": int(common.sum())}
        h1["intrinsic"][f"{a}-{b}"] = {"jaccard": float(np.mean(js_int))}
        h1["usage"][f"{a}-{b}"] = {"jaccard": float(np.mean(js_use))}

    # ---------- 代理有效性: s vs 直接 ΔL ----------
    dm = json.load(open(os.path.join(T.RES_DIR, "delta_matrix.json"), encoding="utf-8"))
    proxy = {"per_domain": {}, "pooled": None, "table": []}
    for d in doms:
        xs, ys = [], []
        for row in dm["experts"]:
            l, e = row["layer"], row["expert"]
            xs.append(s[l, e, doms.index(d)])
            ys.append(row["delta"][d])
            proxy["table"].append({"layer": l, "expert": e, "tag": row["tag"],
                                   "domain": d, "s": float(s[l, e, doms.index(d)]),
                                   "delta_nll": float(row["delta"][d])})
        proxy["per_domain"][d] = {"spearman": spearman(xs, ys),
                                  "n": len(xs),
                                  "s_range": [float(min(xs)), float(max(xs))],
                                  "dNLL_range": [float(min(ys)), float(max(ys))]}
    allx = [r["s"] for r in proxy["table"]]
    ally = [r["delta_nll"] for r in proxy["table"]]
    proxy["pooled"] = {"spearman": spearman(allx, ally), "n": len(allx)}

    # ---------- H4: 配额配置 ----------
    al = json.load(open(os.path.join(T.RES_DIR, "allocation.json"), encoding="utf-8"))
    base = next(c for c in al["configs"] if c["name"] == "base_int4")
    glob = next(c for c in al["configs"] if c["name"] == "global")
    h4 = {"domains": doms, "configs": {}, "verdict": {}}
    for c in al["configs"]:
        gains = {d: glob["nll"][d] - c["nll"][d] for d in doms}
        h4["configs"][c["name"]] = {"nll": c["nll"], "gain_vs_global": gains,
                                    "vs_base_int4": {d: c["nll"][d] - base["nll"][d]
                                                     for d in doms}}
    for d in doms:
        tc = h4["configs"].get(f"tc_{d}")
        if not tc:
            continue
        others = [x for x in doms if x != d]
        h4["verdict"][d] = {
            "in_domain_gain": tc["gain_vs_global"][d],
            "cross_domain_cost": float(np.mean([-tc["gain_vs_global"][o] for o in others])),
            "net_task_weighted": tc["gain_vs_global"][d] - float(np.mean(
                [-tc["gain_vs_global"][o] for o in others])),
        }

    summary = {
        "signal": "s = sum_t route_weight * relative_expert_output_error(INT4 vs BF16)",
        "h1_jaccard_threshold": 0.5, "h1_spearman_threshold": 0.6,
        "h1": h1, "proxy": proxy, "h4": h4,
    }
    p = os.path.join(T.RES_DIR, "summary.json")
    json.dump(summary, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
