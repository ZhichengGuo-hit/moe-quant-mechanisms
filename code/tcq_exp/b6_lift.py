"""B6: 三模型逐域 top-20% Jaccard 的 lift 与 200 次单元 bootstrap 95% CI（统一口径）。

口径 R（route signature, 三模型均可）: 每域每层路由权重直方图, 逐层取 top-20% 专家;
口径 S（sensitivity, 仅主模型）: E1 的 s(l,e,D) 逐层 top-20%。
合并池化 Jaccard = |A∩B| / |A∪B|, 单元 = (层, 专家); 随机基线 0.111 = 0.04/0.36。
lift = J / 0.111; bootstrap 以 (层, 专家) 为单元有放回重采样 200 次。

输出: results/b6_lift.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T

N_BOOT = 200
SEED = 23
FRAC = 0.2
RAND_BASE = (FRAC * FRAC) / (2 * FRAC - FRAC * FRAC)  # 0.1111


def top_mask(h, frac=FRAC):
    L, E = h.shape
    k = max(1, int(round(E * frac)))
    m = np.zeros((L, E), dtype=bool)
    for l in range(L):
        m[l, np.argsort(-h[l])[:k]] = True
    return m


def pooled_jacc(a, b):
    return float((a & b).sum()) / max(float((a | b).sum()), 1.0)


def rank_corr(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load_inputs():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"], dtype=np.float64)
    u = np.array(sm["usage"], dtype=np.float64)
    route = u / np.maximum(u.sum(axis=1, keepdims=True), 1e-12)
    doms = list(sm["domains"])
    inputs = {"qwen3.6-35b-a3b": {
        "L": int(sm["n_layers"]), "E": int(sm["n_experts"]), "K": 8, "domains": doms,
        "route": {d: route[:, :, i] for i, d in enumerate(doms)},
        "sens": {d: s[:, :, i] for i, d in enumerate(doms)}}}
    with open(os.path.join(T.RES_DIR, "cross_model.json"), encoding="utf-8") as f:
        cm = json.load(f)
    for tag, blob in cm.items():
        if not isinstance(blob, dict) or "mean_hist" not in blob:
            continue
        inputs[tag] = {"L": int(blob["n_layers"]), "E": int(blob["n_experts"]),
                       "K": int(blob.get("top_k", 8)), "domains": list(T.DOMAINS),
                       "route": {d: np.array(blob["mean_hist"][d], dtype=np.float64)
                                 for d in T.DOMAINS}}
    return inputs, doms


def main():
    inputs, doms = load_inputs()
    rng = np.random.default_rng(SEED)
    out = {"n_boot": N_BOOT, "frac": FRAC, "random_baseline": RAND_BASE,
           "unit": "(layer, expert)", "models": {}}

    for tag, blob in inputs.items():
        signals = [("route", blob["route"])]
        if "sens" in blob:
            signals.append(("sens", blob["sens"]))
        for sig, hs in signals:
            res = {}
            for i, a in enumerate(doms):
                for b in doms[i + 1:]:
                    ma, mb = top_mask(hs[a]), top_mask(hs[b])
                    va, vb = hs[a].reshape(-1), hs[b].reshape(-1)
                    fa, fb = ma.reshape(-1), mb.reshape(-1)
                    n = fa.size
                    j = pooled_jacc(ma, mb)
                    r = rank_corr(va, vb)
                    js, rs = np.empty(N_BOOT), np.empty(N_BOOT)
                    for bi in range(N_BOOT):
                        idx = rng.integers(0, n, size=n)
                        js[bi] = pooled_jacc(fa[idx], fb[idx])
                        # 保持层内顺序, 逐层 Spearman 再合并
                        rs[bi] = rank_corr(va[idx], vb[idx])
                    jci = ci(js)
                    res[f"{a}-{b}"] = {
                        "jaccard_pooled": j, "lift": j / RAND_BASE,
                        "jaccard_ci95": jci,
                        "lift_ci95": [jci[0] / RAND_BASE, jci[1] / RAND_BASE],
                        "spearman_pooled": r, "spearman_ci95": ci(rs),
                        "n_units": int(n)}
            out["models"][tag] = {"L": blob["L"], "E": blob["E"], "K": blob["K"],
                                  "signal": sig, "pairs": res}
            print(f"[{tag}/{sig}] " + json.dumps(
                {k: [round(v["jaccard_pooled"], 3), round(v["lift"], 2),
                     [round(x, 3) for x in v["jaccard_ci95"]]] for k, v in res.items()},
                ensure_ascii=False), flush=True)

    rnd = []
    for _ in range(200):
        a = np.zeros(256, dtype=bool)
        b = np.zeros(256, dtype=bool)
        a[rng.choice(256, 51, replace=False)] = True
        b[rng.choice(256, 51, replace=False)] = True
        rnd.append(pooled_jacc(a, b))
    out["random_check"] = {"mean": float(np.mean(rnd)), "std": float(np.std(rnd)),
                           "analytic": RAND_BASE}

    p = os.path.join(T.RES_DIR, "b6_lift.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[b6] saved {p} random_check={out['random_check']}", flush=True)


if __name__ == "__main__":
    main()
