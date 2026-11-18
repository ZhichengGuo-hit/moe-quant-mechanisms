"""B2: 逐层分离曲线 (Jaccard / Spearman 随层深) 与层条件化补丁可行性。

对每个模型 / 口径 / 域对:
  - 逐层 top-20% Jaccard 与层内 Spearman
  - 分离质量 sep(l) = 1 - J(l); 集中度 = 最少的层数占 80% Σsep
  - 若 80% 分离由 <= 25% 的层承担 -> 判定"集中", 层条件化补丁可行
输出: results/b2_layer_curve.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T

FRAC = 0.2


def top_mask(h, frac=FRAC):
    L, E = h.shape
    k = max(1, int(round(E * frac)))
    m = np.zeros((L, E), dtype=bool)
    for l in range(L):
        m[l, np.argsort(-h[l])[:k]] = True
    return m


def rank_corr(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def smooth(x, w=3):
    x = np.asarray(x, dtype=np.float64)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid").tolist()


def analyze(hs, doms, tag, sig):
    L = hs[doms[0]].shape[0]
    out = {}
    for i, a in enumerate(doms):
        for b in doms[i + 1:]:
            ma, mb = top_mask(hs[a]), top_mask(hs[b])
            js = [1.0 - float((ma[l] & mb[l]).sum() / max((ma[l] | mb[l]).sum(), 1))
                  for l in range(L)]
            # 逐层 Spearman -> 转成"不相似"方向: 1 - (1+rho)/2 ∈ [0,1]
            rs = [1.0 - (1.0 + rank_corr(hs[a][l], hs[b][l])) / 2.0 for l in range(L)]
            sep = np.array(js)                       # 1 - J 已是分离强度
            order = np.argsort(-sep)
            cum = np.cumsum(sep[order]) / max(sep.sum(), 1e-12)
            n80 = int(np.searchsorted(cum, 0.8) + 1)
            n25 = max(1, int(np.ceil(0.25 * L)))
            top_quarter_share = float(sep[order][:n25].sum() / max(sep.sum(), 1e-12))
            out[f"{a}-{b}"] = {
                "jaccard_per_layer": js,
                "separation_per_layer": sep.tolist(),
                "spearman_per_layer": rs,
                "jaccard_smooth3": smooth(js),
                "separation_smooth3": smooth(sep),
                "min_jaccard": float(np.min(js)),
                "min_jaccard_layer": int(np.argmin(js)),
                "max_jaccard": float(np.max(js)),
                "jaccard_std": float(np.std(js)),
                "n_layers_80pct_sep": n80,
                "n_layers_80pct_frac": n80 / L,
                "top_quarter_sep_share": top_quarter_share,
                "concentrated": bool(top_quarter_share >= 0.5 and n80 <= n25),
                "best_layers": [int(x) for x in order[:n80]],
            }
    return out


def load_inputs():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"], dtype=np.float64)
    u = np.array(sm["usage"], dtype=np.float64)
    route_main = u / np.maximum(u.sum(axis=1, keepdims=True), 1e-12)
    inputs = {
        "qwen3.6-35b-a3b": {
            "L": int(sm["n_layers"]), "E": int(sm["n_experts"]),
            "domains": list(sm["domains"]),
            "route": {d: route_main[:, :, i] for i, d in enumerate(sm["domains"])},
            "sens": {d: s[:, :, i] for i, d in enumerate(sm["domains"])},
        }
    }
    with open(os.path.join(T.RES_DIR, "cross_model.json"), encoding="utf-8") as f:
        cm = json.load(f)
    for tag, blob in cm.items():
        if not isinstance(blob, dict) or "mean_hist" not in blob:
            continue
        inputs[tag] = {"L": int(blob["n_layers"]), "E": int(blob["n_experts"]),
                       "domains": list(T.DOMAINS),
                       "route": {d: np.array(blob["mean_hist"][d], dtype=np.float64)
                                 for d in T.DOMAINS}}
    return inputs, list(T.DOMAINS)


def main():
    inputs, doms = load_inputs()
    out = {"frac": FRAC, "models": {}}
    for tag, blob in inputs.items():
        signals = [("route", blob["route"])]
        if "sens" in blob:
            signals.append(("sens", blob["sens"]))
        out["models"][tag] = {"L": blob["L"], "E": blob["E"], "signals": {}}
        for sig, hs in signals:
            res = analyze(hs, doms, tag, sig)
            out["models"][tag]["signals"][sig] = res
            print(f"[{tag}/{sig}] " + " | ".join(
                f"{k}: Jmin={v['min_jaccard']:.3f}@L{v['min_jaccard_layer']} "
                f"n80={v['n_layers_80pct_sep']}/{blob['L']} "
                f"top25%share={v['top_quarter_sep_share']:.2f} "
                f"{'CONC' if v['concentrated'] else 'DIFF'}"
                for k, v in res.items()), flush=True)

    p = os.path.join(T.RES_DIR, "b2_layer_curve.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[b2] saved {p}", flush=True)


if __name__ == "__main__":
    main()
