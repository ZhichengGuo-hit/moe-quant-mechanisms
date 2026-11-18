"""E10: 扩评测集复测 (64 条/域) —— 阶段 0 结论是否在更小噪声下保持。

配置: base_int4 / global / tc_math / tc_code / usage_math / usage_code
评测: data/{domain}_eval64.jsonl (64 条, 64 token)
统计: 每配置每域 NLL + 逐序列 NLL; 奇偶两半 NLL 差 (噪声标尺);
      以 global 为基准的域内增益 1000 次序列级 bootstrap 95% CI。
输出: results/large_eval.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T
from moe_lib import Int4Store

P_HIGH = 0.25
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")
N_BOOT = 1000
SEED = 7


def select_per_layer(score, frac):
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        high[l, np.argsort(-score[l])[:k]] = True
    return high


def agg(per_seq, idx=None):
    if idx is None:
        idx = range(len(per_seq))
    s = sum(per_seq[i][0] for i in idx)
    n = sum(per_seq[i][1] for i in idx)
    return s / max(n, 1)


def main():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])
    u = np.array(sm["usage"])
    doms = sm["domains"]
    di = {d: i for i, d in enumerate(doms)}

    eng = T.build_engine()
    tok = eng["tok"]
    patcher = T.ExpertPatcher(eng["store"])
    store3 = Int4Store(T.STORE_INT3, mmap_mode="r")
    v3 = {l: store3.layer_views(l) for l in range(store3.num_layers)}
    n_layers, n_experts = store3.num_layers, store3.num_experts

    evalsets = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval64")
        evalsets[dom] = [r["text"] for r in rows]
        print(f"[data] {dom}: {len(evalsets[dom])} seq", flush=True)

    configs = [
        ("base_int4", None),
        ("global", s.sum(-1)),
        ("tc_math", s[:, :, di["math"]]),
        ("tc_code", s[:, :, di["code"]]),
        ("usage_math", u[:, :, di["math"]]),
        ("usage_code", u[:, :, di["code"]]),
    ]

    per_seq = {}
    nll = {}
    t_all = time.time()
    for name, score in configs:
        t0 = time.time()
        high = select_per_layer(score, P_HIGH) if score is not None else None
        if high is not None:
            for l in range(n_layers):
                for e in np.nonzero(~high[l])[0]:
                    patcher.set_expert(l, int(e),
                                       {p: v3[l][p][int(e)] for p in LAYER_PARTS})
        per_seq[name] = {}
        nll[name] = {}
        for dom in T.DOMAINS:
            tot, n, per, ps = T.eval_nll(eng["model"], tok, evalsets[dom],
                                         max_len=64, batch_size=8,
                                         return_per_seq=True)
            per_seq[name][dom] = ps
            nll[name][dom] = per
        if high is not None:
            for l in range(n_layers):
                for e in np.nonzero(~high[l])[0]:
                    patcher.restore_expert(l, int(e))
        print(f"[{name}] nll=" + " ".join(f"{d}:{nll[name][d]:.4f}" for d in T.DOMAINS) +
              f" ({time.time()-t0:.0f}s)", flush=True)

    # ---- 噪声标尺: 奇偶两半 ----
    split = {}
    for name in nll:
        for dom in T.DOMAINS:
            ps = per_seq[name][dom]
            ev = list(range(0, len(ps), 2))
            od = list(range(1, len(ps), 2))
            split[f"{name}/{dom}"] = abs(agg(ps, ev) - agg(ps, od))

    # ---- 增益 + bootstrap CI (相对 global, 按域) ----
    rng = np.random.default_rng(SEED)
    ni = [len(per_seq["global"][d]) for d in T.DOMAINS]
    gains, ci = {}, {}
    for name in nll:
        gains[name] = {d: nll["global"][d] - nll[name][d] for d in T.DOMAINS}
        ci[name] = {}
        for dom in T.DOMAINS:
            g_ps, c_ps = per_seq["global"][dom], per_seq[name][dom]
            n = len(g_ps)
            vals = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                vals[b] = agg(g_ps, idx) - agg(c_ps, idx)
            ci[name][dom] = [float(np.percentile(vals, 2.5)),
                             float(np.percentile(vals, 97.5))]

    out = {"n_eval_seq": {d: len(evalsets[d]) for d in T.DOMAINS},
           "configs": [c[0] for c in configs],
           "nll": nll, "gain_vs_global": gains, "ci95": ci,
           "split_half_abs_delta": split,
           "n_boot": N_BOOT,
           "per_seq": {k: {d: per_seq[k][d] for d in T.DOMAINS} for k in per_seq},
           "elapsed_s": round(time.time() - t_all, 1)}
    p = os.path.join(T.RES_DIR, "large_eval.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[large_eval] saved {p} ({time.time()-t_all:.0f}s)", flush=True)
    for name in ("tc_math", "tc_code", "usage_math", "usage_code"):
        print(f"[gain] {name}: " + " ".join(
            f"{d}:{gains[name][d]:+.4f} ci{ci[name][d]}" for d in T.DOMAINS), flush=True)


if __name__ == "__main__":
    main()
