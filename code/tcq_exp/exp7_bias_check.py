"""实验7: 3.5(a) 中 7232 个"三域都使用"单元的选择偏差自查。

要查的问题: 方差分解只用了三域都被路由到的专家, 这批专家天然是"通用专家",
排除掉的恰恰是最任务的专家(只在一两域出现)。结论"(条件性来自 usage)"是否
只是这个子集造成的?

自查内容:
 1) 单元覆盖统计(0/1/2/3 域) + 被排除单元占用的 usage 质量 + 层分布
 2) 无插补版本: 只用"恰好 2 域"的单元, 在它被使用的两域内做同样的分解
 3) 全网格版本(缺失域 usage 用各域中位数的 floor 填补, q 用域中位数填补) —— 明确标注为插补敏感
 4) 按 usage 四分位分层看 share
 5) 单元 bootstrap(200 次)给 share 的 90% 区间
 6) 分量相关: 跨域 Pearson corr(s), corr(u), corr(q) —— 不依赖 log 分解

输出: results/bias_check.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T


def load():
    sm = json.load(open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8"))
    s = np.array(sm["s"])
    u = np.array(sm["usage"])
    doms = sm["domains"]
    return s, u, doms


def logs(x):
    return np.log(np.where(x > 0, x, np.nan))


def decomp(s, u, q, masked_doms):
    """masked_doms: bool [L,E,D], 只有 True 的域参与跨域方差。
    返回 per-cell (share_u, share_q, share_cov, n_dom)"""
    L, E, D = s.shape
    ok = masked_doms.any(-1) & (masked_doms.sum(-1) >= 2)
    ls, lu, lq = logs(s), logs(u), logs(q)
    out = {}
    vals_u, vals_q, vals_c, vals_s, ns = [], [], [], [], []
    for l, e in zip(*np.nonzero(ok)):
        m = masked_doms[l, e]
        if m.sum() < 2:
            continue
        a, b, c = ls[l, e][m], lu[l, e][m], lq[l, e][m]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                and np.all(np.isfinite(c))):
            continue
        vs = np.var(a)
        if vs <= 0:
            continue
        vu, vq = np.var(b), np.var(c)
        cv = np.mean((b - b.mean()) * (c - c.mean()))
        vals_s.append(vs)
        vals_u.append(vu)
        vals_q.append(vq)
        vals_c.append(2 * cv)
        ns.append(int(m.sum()))
    out["n_cells"] = len(vals_s)
    out["mean_var_log_s"] = float(np.mean(vals_s))
    out["ratio_of_means_usage"] = float(np.mean(vals_u) / np.mean(vals_s))
    out["ratio_of_means_q"] = float(np.mean(vals_q) / np.mean(vals_s))
    out["ratio_of_means_cov"] = float(np.mean(vals_c) / np.mean(vals_s))
    out["share_usage"] = float(np.mean(np.array(vals_u) / np.array(vals_s)))
    out["share_intrinsic_q"] = float(np.mean(np.array(vals_q) / np.array(vals_s)))
    out["share_cov"] = float(np.mean(np.array(vals_c) / np.array(vals_s)))
    out["mean_var_log_u"] = float(np.mean(vals_u))
    out["mean_var_log_q"] = float(np.mean(vals_q))
    return out, (np.array(vals_u), np.array(vals_q), np.array(vals_s))


def part_stats(s, u, doms):
    act = u > 0
    k = act.sum(-1)
    mass = u.sum(axis=(0, 1))           # 每域总路由权重
    by_k = {}
    for kk in (0, 1, 2, 3):
        sel = k == kk
        row = {"n_cells": int(sel.sum())}
        for di, d in enumerate(doms):
            row[f"usage_mass_{d}"] = float(u[..., di][sel].sum())
            row[f"mass_frac_{d}"] = float(u[..., di][sel].sum() / max(mass[di], 1e-12))
        by_k[str(kk)] = row
    per_layer = [int((k == 3)[l].sum()) for l in range(k.shape[0])]
    mean_usage = {}
    for di, d in enumerate(doms):
        v = u[..., di]
        mean_usage[d] = {
            "k3_mean": float(v[k == 3].mean()),
            "k2_mean": float(v[k == 2].mean()),
            "k1_mean": float(v[k == 1].mean()),
            "median_active": float(np.median(v[v > 0])),
        }
    return {"by_domain_coverage": by_k,
            "k3_count_per_layer": per_layer,
            "k3_min_max_layer": [int(min(per_layer)), int(max(per_layer))],
            "usage_scale": mean_usage}


def part_subsets(s, u, q, doms):
    act = u > 0
    k = act.sum(-1)
    res = {}
    res["baseline_k3"] = decomp(s, u, q, np.broadcast_to(k[..., None] == 3, s.shape)
                                & act)[0]
    res["check_k2"] = decomp(s, u, q, np.broadcast_to(k[..., None] == 2, s.shape)
                             & act)[0]

    # 全网格 + floor 插补 (标注敏感)
    med = np.array([np.median(u[..., di][u[..., di] > 0]) for di in range(len(doms))])
    med_q = np.array([np.nanmedian(q[..., di]) for di in range(len(doms))])
    floored = {}
    for ff in (1e-3, 1e-2, 1e-1):
        uu = u.copy()
        qq = q.copy()
        for di in range(len(doms)):
            miss = uu[..., di] == 0
            uu[..., di][miss] = ff * med[di]
            qq[..., di][miss] = med_q[di]
        ss = uu * qq   # 逐单元重建 s = u*q
        floored[str(ff)] = decomp(ss, uu, qq, np.ones_like(u, dtype=bool))[0]
        floored[str(ff)]["floor_value"] = {d: float(ff * med[i])
                                           for i, d in enumerate(doms)}
        floored[str(ff)]["note"] = "缺失域 usage/q 用中位数*floor 插补, 结论对插补敏感"
    res["floored_full_grid"] = floored

    # usage 四分位分层 (baseline k3 单元)
    k3 = np.all(act, axis=-1)
    lmean = np.nanmean(np.log(np.where(act, u, np.nan)), axis=-1)
    v = lmean[k3]
    qs = np.quantile(v, [0.25, 0.5, 0.75])
    layers = {}
    for name, lo, hi in (("q1_lowest", -np.inf, qs[0]), ("q2", qs[0], qs[1]),
                         ("q3", qs[1], qs[2]), ("q4_highest", qs[2], np.inf)):
        mask = k3 & (lmean >= lo) & (lmean < hi)
        layers[name] = decomp(s, u, q, np.broadcast_to(k[..., None] == 3, s.shape)
                              & act & mask[..., None])[0]
    res["by_usage_quartile"] = layers
    return res


def part_bootstrap(s, u, q, doms, n_boot=200):
    k3 = np.all(u > 0, axis=-1)
    _, (vu, vq, vs) = decomp(s, u, q, np.broadcast_to(k3[..., None], s.shape))
    rng = np.random.default_rng(0)
    n = len(vs)
    su, sq, sc = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        su.append(float(np.mean(vu[i] / vs[i])))
        sq.append(float(np.mean(vq[i] / vs[i])))
        sc.append(float(np.mean(2 * (vs[i] - vu[i] - vq[i]) / vs[i])))
    q5 = lambda a: [round(float(np.percentile(a, 5)), 3), round(float(np.percentile(a, 95)), 3)]
    return {"n_cells": int(n), "n_boot": n_boot,
            "share_usage_ci90": q5(su), "share_q_ci90": q5(sq),
            "share_cov_ci90": q5(sc)}


def part_corr(s, u, q, doms):
    k3 = np.all(u > 0, axis=-1)
    out = {"s": {}, "u": {}, "q": {}}
    for name, arr in (("s", s), ("u", u), ("q", q)):
        for a in range(len(doms)):
            for b in range(a + 1, len(doms)):
                x = arr[..., a][k3]
                y = arr[..., b][k3]
                out[name][f"{doms[a]}-{doms[b]}"] = float(np.corrcoef(x, y)[0, 1])
    return out


def main():
    s, u, doms = load()
    q = np.where(u > 0, s / np.maximum(u, 1e-30), np.nan)
    out = {"stats": part_stats(s, u, doms),
           "subsets": part_subsets(s, u, q, doms),
           "bootstrap": part_bootstrap(s, u, q, doms),
           "cross_domain_corr": part_corr(s, u, q, doms)}
    p = os.path.join(T.RES_DIR, "bias_check.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[bias] saved {p}")


if __name__ == "__main__":
    main()
