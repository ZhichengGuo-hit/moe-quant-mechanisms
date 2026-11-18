"""B3: 路由集中度画像 (exp(entropy)/Gini/top20 质量) 与逐层分离度 (1 - Jaccard) 的跨层相关。

对每个模型:
  - 逐层逐域: 有效专家数 exp(H)、Gini、top-20% 质量、max 权重
  - 逐层分离度: 逐层 top-20% Jaccard 的 1 - J
  - 跨层相关: Pearson / Spearman(有效专家数 vs 分离度; Gini vs 分离度), 置换检验 p 值
输出: results/b3_concentration.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T

FRAC = 0.2
N_PERM = 500
SEED = 31


def gini(p):
    p = np.sort(np.asarray(p, dtype=np.float64))
    n = p.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(p) / (n * p.sum() + 1e-12))


def metrics(p):
    p = np.maximum(p, 1e-12)
    p = p / p.sum()
    eff = float(np.exp(-(p * np.log(p)).sum()))
    k = max(1, int(round(len(p) * FRAC)))
    top20 = float(np.sort(p)[::-1][:k].sum())
    return {"eff_experts": eff, "gini": gini(p), "top20_mass": top20,
            "max_weight": float(p.max())}


def top_mask(h, frac=FRAC):
    L, E = h.shape
    k = max(1, int(round(E * frac)))
    m = np.zeros((L, E), dtype=bool)
    for l in range(L):
        m[l, np.argsort(-h[l])[:k]] = True
    return m


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def perm_p(x, y, rng, stat):
    obs = stat(x, y)
    cnt = 0
    y2 = np.array(y, float)
    for _ in range(N_PERM):
        rng.shuffle(y2)
        if abs(stat(x, y2)) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (N_PERM + 1)


def load_inputs():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    u = np.array(sm["usage"], dtype=np.float64)
    route = u / np.maximum(u.sum(axis=1, keepdims=True), 1e-12)
    doms = list(sm["domains"])
    inputs = {"qwen3.6-35b-a3b": {
        "L": int(sm["n_layers"]), "E": int(sm["n_experts"]), "domains": doms,
        "route": {d: route[:, :, i] for i, d in enumerate(doms)}}}
    with open(os.path.join(T.RES_DIR, "cross_model.json"), encoding="utf-8") as f:
        cm = json.load(f)
    for tag, blob in cm.items():
        if not isinstance(blob, dict) or "mean_hist" not in blob:
            continue
        inputs[tag] = {"L": int(blob["n_layers"]), "E": int(blob["n_experts"]),
                       "domains": list(T.DOMAINS),
                       "route": {d: np.array(blob["mean_hist"][d], dtype=np.float64)
                                 for d in T.DOMAINS}}
    return inputs, doms


def main():
    inputs, doms = load_inputs()
    rng = np.random.default_rng(SEED)
    out = {"frac": FRAC, "n_perm": N_PERM, "models": {}}

    for tag, blob in inputs.items():
        hs = blob["route"]
        L, E = blob["L"], blob["E"]
        per_dom = {}
        for d in doms:
            per_dom[d] = {k: [] for k in ("eff_experts", "gini", "top20_mass", "max_weight")}
            for l in range(L):
                m = metrics(hs[d][l])
                for k in per_dom[d]:
                    per_dom[d][k].append(m[k])
        # 跨域平均集中度
        avg = {k: np.mean([per_dom[d][k] for d in doms], axis=0)
               for k in ("eff_experts", "gini", "top20_mass", "max_weight")}
        for d in per_dom:
            for k in per_dom[d]:
                per_dom[d][k] = [float(x) for x in per_dom[d][k]]
        for k in avg:
            avg[k] = [float(x) for x in avg[k]]

        pairs = {}
        for i, a in enumerate(doms):
            for b in doms[i + 1:]:
                ma, mb = top_mask(hs[a]), top_mask(hs[b])
                jl = np.array([1.0 - float((ma[l] & mb[l]).sum() /
                                           max((ma[l] | mb[l]).sum(), 1)) for l in range(L)])
                pairs[f"{a}-{b}"] = {
                    "sep_mean": float(jl.mean()), "sep_std": float(jl.std()),
                    "sep_per_layer": jl.tolist(),
                    "corr": {}}
                for mname in ("eff_experts", "gini", "top20_mass", "max_weight"):
                    x = avg[mname]
                    for sname, fn in (("pearson", pearson), ("spearman", spearman)):
                        r = fn(x, jl)
                        pairs[f"{a}-{b}"]["corr"][f"{mname}/{sname}"] = {
                            "r": r, "p": perm_p(x, jl, rng, fn)}

        # 域间集中度差 vs 分离度
        gap = {}
        for i, a in enumerate(doms):
            for b in doms[i + 1:]:
                dg = np.abs(np.array(per_dom[a]["gini"]) - np.array(per_dom[b]["gini"]))
                jl = np.array(pairs[f"{a}-{b}"]["sep_per_layer"])
                gap[f"{a}-{b}"] = {"pearson": {"r": pearson(dg, jl),
                                              "p": perm_p(dg, jl, rng, pearson)},
                                   "spearman": {"r": spearman(dg, jl),
                                                "p": perm_p(dg, jl, rng, spearman)}}

        out["models"][tag] = {"L": L, "E": E, "per_domain": per_dom, "avg": avg,
                              "pairs": pairs, "gini_gap_vs_sep": gap}
        print(f"[{tag}] avg eff={np.mean(avg['eff_experts']):.1f}/{E} "
              f"gini={np.mean(avg['gini']):.3f} " +
              " | ".join(f"{k}: sep={v['sep_mean']:.3f} "
                         f"r(eff)={v['corr']['eff_experts/pearson']['r']:+.3f}"
                         f"(p{v['corr']['eff_experts/pearson']['p']:.3f})"
                         for k, v in pairs.items()), flush=True)

    p = os.path.join(T.RES_DIR, "b3_concentration.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[b3] saved {p}", flush=True)


if __name__ == "__main__":
    main()
