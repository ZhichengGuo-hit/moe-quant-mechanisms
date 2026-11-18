"""E8: 成组 ΔL 金标准 —— 验证 s / usage 排序在组级扰动下是否有效。

E2 的单专家 ΔL 只有 1/256 专家、1/8 路由权重参与, 与评测噪声同量级 (Spearman 0.258)。
这里把"同一层 top-8 敏感专家"一起降为 INT2, 把信号放大 ~8 倍, 再测三域 NLL 变化。

组: 每域在层 [0,8,16,24,32,39] 上按该域 s 取 top-8 -> 6 个敏感组;
    另每域 2 个随机组 (该层 usage>0 专家中随机 8 个, seed 42); 共 24 组。
输出: results/group_delta.json
用法: python exp8_group_delta.py
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T

N_EVAL_SEQ = 8
K_GROUP = 8
LAYERS = (0, 8, 16, 24, 32, 39)
N_RANDOM = 2
SEED = 42


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
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def build_groups(s, u, doms):
    groups = []
    rng = np.random.default_rng(SEED)
    for di, dom in enumerate(doms):
        for layer in LAYERS:
            idx = np.argsort(-s[layer, :, di])[:K_GROUP]
            groups.append({"layer": int(layer), "experts": [int(i) for i in idx],
                           "tag": f"top-{dom}", "domain": dom})
        for layer in LAYERS[:N_RANDOM]:
            used = np.nonzero(u[layer, :, di] > 0)[0]
            if len(used) < K_GROUP:
                continue
            idx = rng.choice(used, size=K_GROUP, replace=False)
            groups.append({"layer": int(layer), "experts": [int(i) for i in idx],
                           "tag": f"rand-{dom}", "domain": dom})
    return groups


def main():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])          # [L,E,3]
    u = np.array(sm["usage"])
    doms = sm["domains"]
    groups = build_groups(s, u, doms)

    eng = T.build_engine()
    tok = eng["tok"]
    patcher = T.ExpertPatcher(eng["store"])
    wmap = T.load_weight_map()

    evalsets = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval")[:N_EVAL_SEQ]
        evalsets[dom] = [r["text"] for r in rows]

    base = {}
    for dom in T.DOMAINS:
        tot, n, per = T.eval_nll(eng["model"], tok, evalsets[dom],
                                 max_len=64, batch_size=8)
        base[dom] = per
        print(f"[base] {dom} nll={per:.4f}", flush=True)

    rows_out = []
    t_all = time.time()
    for gi, g in enumerate(groups):
        layer, experts = g["layer"], g["experts"]
        di = doms.index(g["domain"])
        sc_s = float(sum(s[layer, e, di] for e in experts))
        sc_u = float(sum(u[layer, e, di] for e in experts))
        t0 = time.time()
        try:
            for e in experts:
                w_gu, w_dn = T.read_expert_bf16(wmap, T.MOE_MODEL_PATH, layer, e)
                patcher.set_expert_from_bits(layer, e, 2, w_gu, w_dn)
            delta = {}
            for dom in T.DOMAINS:
                tot, n, per = T.eval_nll(eng["model"], tok, evalsets[dom],
                                         max_len=64, batch_size=8)
                delta[dom] = per - base[dom]
        finally:
            for e in experts:
                patcher.restore_expert(layer, e)
        res = {"layer": layer, "experts": experts, "tag": g["tag"],
               "domain": g["domain"], "score_s": sc_s, "score_usage": sc_u,
               "delta_nll": delta}
        rows_out.append(res)
        print(f"[{gi+1}/{len(groups)}] {g['tag']} L{layer} "
              f"dNLL=" + " ".join(f"{d}:{delta[d]:+.4f}" for d in T.DOMAINS) +
              f" ({time.time()-t0:.0f}s)", flush=True)

    # ---- 分析: 预测分数 vs 实际 ΔNLL ----
    analysis = {"per_domain": {}, "pooled": {}, "amplification": {}}
    for dom in T.DOMAINS:
        di = doms.index(dom)
        pred_s, pred_u, obs = [], [], []
        for r in rows_out:
            layer = r["layer"]
            m = sum(s[layer, e, di] for e in r["experts"])
            n = sum(u[layer, e, di] for e in r["experts"])
            pred_s.append(m)
            pred_u.append(n)
            obs.append(r["delta_nll"][dom])
        analysis["per_domain"][dom] = {
            "spearman_s": spearman(pred_s, obs),
            "spearman_usage": spearman(pred_u, obs),
            "mean_abs_delta": float(np.mean(np.abs(obs))),
            "max_abs_delta": float(np.max(np.abs(obs))),
        }

    # pooled: 每 (组, 域) 一个点
    ps, pu, po = [], [], []
    for dom in T.DOMAINS:
        di = doms.index(dom)
        for r in rows_out:
            layer = r["layer"]
            ps.append(sum(s[layer, e, di] for e in r["experts"]))
            pu.append(sum(u[layer, e, di] for e in r["experts"]))
            po.append(r["delta_nll"][dom])
    analysis["pooled"] = {"spearman_s": spearman(ps, po),
                          "spearman_usage": spearman(pu, po),
                          "n": len(po)}

    sens = [r for r in rows_out if r["tag"].startswith("top-")]
    rand = [r for r in rows_out if r["tag"].startswith("rand-")]
    for name, grp in (("sensitive", sens), ("random", rand)):
        per = {}
        for dom in T.DOMAINS:
            per[dom] = float(np.mean([abs(r["delta_nll"][dom]) for r in grp]))
        analysis["amplification"][name] = {"n": len(grp), "mean_abs_delta": per}
    if rand:
        ratio = (np.mean([abs(r["delta_nll"][r["domain"]]) for r in sens]) /
                 max(np.mean([abs(r["delta_nll"][r["domain"]]) for r in rand]), 1e-12))
        analysis["amplification"]["ratio_sensitive_over_random"] = float(ratio)

    out = {"n_eval_seq": N_EVAL_SEQ, "k_group": K_GROUP, "layers": list(LAYERS),
           "base_nll": base, "groups": rows_out, "analysis": analysis,
           "elapsed_s": round(time.time() - t_all, 1)}
    p = os.path.join(T.RES_DIR, "group_delta.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[group_delta] saved {p} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
