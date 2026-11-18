"""实验5: s 分解 (s ≈ usage × 内在量化误差) 与配额选择的归因。

阶段 A (无需模型):
  q(e,l,D) = s/u (被路由 token 上专家的平均相对量化误差), 并做跨域方差分解
  log s = log u + log q  ->  Var(log s) = Var(log u) + Var(log q) + 2Cov
  以及"分量置换"版敏感度 (用域平均替换一个分量) 的 Jaccard 归因

阶段 B (需模型):
  只用 usage / 只用 q 作为分数时, 任务条件配额能回收多少域内损伤
  (对照已有的 tc_math / tc_code = 完整 s)

用法: python exp5_decomp.py --part analysis   (秒级)
      python exp5_decomp.py --part alloc      (约 10 分钟)
输出: results/decomp.json
"""
import os
import sys
import json
import time
import argparse
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T

FRAC = 0.2
P_HIGH = 0.25
N_EVAL_SEQ = 8
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")


def top_sets(score, frac=FRAC):
    L, E = score.shape
    k = max(1, int(E * frac))
    out = []
    for l in range(L):
        idx = np.argsort(-score[l])[:k]
        out.append(set(int(i) for i in idx))
    return out


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def rand_pairwise(scores, doms):
    sets = {d: top_sets(scores[d]) for d in doms}
    res = {}
    for a, b in itertools.combinations(doms, 2):
        res[f"{a}-{b}"] = float(np.mean([jaccard(sets[a][l], sets[b][l])
                                         for l in range(len(sets[a]))]))
    return res


def load():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])
    u = np.array(sm["usage"])
    doms = sm["domains"]
    q = np.where(u > 0, s / np.maximum(u, 1e-30), np.nan)
    return s, u, q, doms


def part_analysis():
    s, u, q, doms = load()
    ok = np.all(u > 0, axis=-1)          # 三域都使用过 -> 可做分量分解
    ls, lu, lq = (np.log(np.where(x > 0, x, np.nan)) for x in (s, u, q))

    def v(x):
        return np.nanvar(np.where(ok[..., None], x, np.nan), axis=-1)

    def cov(a, b):
        A = np.where(ok[..., None], a, np.nan)
        B = np.where(ok[..., None], b, np.nan)
        return np.nanmean((A - np.nanmean(A, -1, keepdims=True)) *
                          (B - np.nanmean(B, -1, keepdims=True)), axis=-1)

    var_s = v(ls)[ok]
    var_u = v(lu)[ok]
    var_q = v(lq)[ok]
    c = cov(lu, lq)[ok]
    n = len(var_s)
    mean_var_s = float(np.nanmean(var_s))
    share_u = float(np.nanmean(var_u) / mean_var_s)
    share_q = float(np.nanmean(var_q) / mean_var_s)
    share_cov = float(np.nanmean(2 * c) / mean_var_s)
    frac_u_dom = float(np.nanmean(var_u > var_q))

    qbar = np.nanmean(q, axis=-1, keepdims=True)   # 域无关的内在误差
    ubar = np.nanmean(u, axis=-1, keepdims=True)
    qbar = np.where(np.isnan(qbar), 0.0, qbar)
    ubar = np.where(np.isnan(ubar), 0.0, ubar)

    variants = {
        "s_full": {d: np.where(u[..., i] > 0, s[..., i], 0.0)
                   for i, d in enumerate(doms)},
        "usage_only": {d: np.where(u[..., i] > 0, u[..., i], 0.0)
                       for i, d in enumerate(doms)},
        "q_only": {d: np.where(u[..., i] > 0, q[..., i], 0.0)
                   for i, d in enumerate(doms)},
        "s_from_usage": {d: np.where(u[..., i] > 0, u[..., i] * qbar[..., 0], 0.0)
                         for i, d in enumerate(doms)},
        "s_from_q": {d: np.where(u[..., i] > 0, ubar[..., 0] * q[..., i], 0.0)
                     for i, d in enumerate(doms)},
    }
    out = {
        "n_cells_all_domains": int(ok.sum()),
        "signal": "s = usage * q, q = mean relative INT4-vs-BF16 expert output error",
        "variance_decomp": {
            "mean_var_log_s": mean_var_s,
            "share_usage": share_u,
            "share_intrinsic_q": share_q,
            "share_cov": share_cov,
            "frac_cells_usage_dominates": frac_u_dom,
            "n_cells": n,
        },
        "jaccard_by_variant": {k: rand_pairwise(v, doms) for k, v in variants.items()},
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    p = os.path.join(T.RES_DIR, "decomp.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[decomp] saved {p}")
    return out


def select_per_layer(score, frac):
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        idx = np.argsort(-score[l])[:k]
        high[l, idx] = True
    return high


def part_alloc():
    s, u, q, doms = load()
    with open(os.path.join(T.RES_DIR, "allocation.json"), encoding="utf-8") as f:
        al = json.load(f)
    ref = {c["name"]: c["nll"] for c in al["configs"]}

    eng = T.build_engine()
    tok = eng["tok"]
    store = eng["store"]
    patcher = T.ExpertPatcher(store)
    store3 = T.STORE_INT3
    from moe_lib import Int4Store
    s3 = Int4Store(store3, mmap_mode="r")
    v3 = {l: s3.layer_views(l) for l in range(s3.num_layers)}

    evalsets = {}
    for dom in ("math", "code"):
        rows = T.load_domain(dom, "eval")[:N_EVAL_SEQ]
        evalsets[dom] = [r["text"] for r in rows]

    def run(name, score):
        t0 = time.time()
        high = select_per_layer(score, P_HIGH)
        for l in range(store.num_layers):
            for e in np.nonzero(~high[l])[0]:
                patcher.set_expert(l, int(e), {p: v3[l][p][int(e)] for p in LAYER_PARTS})
        nll = {}
        for dom in ("math", "code"):
            tot, n, per = T.eval_nll(eng["model"], tok, evalsets[dom],
                                     max_len=64, batch_size=8)
            nll[dom] = per
        for l in range(store.num_layers):
            for e in np.nonzero(~high[l])[0]:
                patcher.restore_expert(l, int(e))
        print(f"[{name}] nll=" + " ".join(f"{d}:{nll[d]:.4f}" for d in nll) +
              f" ({time.time()-t0:.0f}s)", flush=True)
        return {"name": name, "nll": nll}

    configs = []
    for i, dom in enumerate(doms):
        if dom == "general":
            continue          # general 对量化不敏感, 不做分配实验
        configs.append(run(f"{dom}_usage_only",
                           np.where(u[..., i] > 0, u[..., i], 0.0)))
        configs.append(run(f"{dom}_q_only",
                           np.where(u[..., i] > 0, q[..., i], 0.0)))

    glob = ref["global"]
    out = {"p_high": P_HIGH, "n_eval_seq": N_EVAL_SEQ, "ref_nll": ref,
           "configs": configs}
    for c in configs:
        c["gain_vs_global"] = {d: float(glob[d] - c["nll"][d]) for d in c["nll"]}
        print(f"  {c['name']} gain_vs_global=" +
              " ".join(f"{d}:{c['gain_vs_global'][d]:+.4f}" for d in c["nll"]), flush=True)

    p = os.path.join(T.RES_DIR, "decomp.json")
    prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    prev.update(out)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=1)
    print(f"[decomp] updated {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["analysis", "alloc"], required=True)
    a = ap.parse_args()
    part_analysis() if a.part == "analysis" else part_alloc()
