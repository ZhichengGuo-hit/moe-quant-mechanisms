"""C3: eval128 主表复测 + general 按任务类型拆分。

配置 (与 E10 一致): base_int4 / global / tc_math / tc_code / usage_math / usage_code
评测: data/{math,code,general}_eval128.jsonl (128 条/域, 64 token)
统计: 每配置每域 NLL + 1000 次序列级 bootstrap 95% CI (相对 global);
       general 按 subtype 分组 (generate / explain / extract / other) 的子集增益。
输出: results/c3_eval128.json
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
SEED = 61
MIN_SUBTYPE = 8


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
    s = np.array(sm["s"]); u = np.array(sm["usage"])
    doms, di = sm["domains"], {d: i for i, d in enumerate(sm["domains"])}

    eng = T.build_engine()
    tok = eng["tok"]
    patcher = T.ExpertPatcher(eng["store"])
    store3 = Int4Store(T.STORE_INT3, mmap_mode="r")
    v3 = {l: store3.layer_views(l) for l in range(store3.num_layers)}
    n_layers, n_experts = store3.num_layers, store3.num_experts

    evalsets, subtypes = {}, {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval128")
        evalsets[dom] = [r["text"] for r in rows]
        if dom == "general":
            subtypes = [r.get("subtype", "other") for r in rows]
        print(f"[data] {dom}: {len(evalsets[dom])} seq", flush=True)

    configs = [("base_int4", None), ("global", s.sum(-1)),
               ("tc_math", s[:, :, di["math"]]), ("tc_code", s[:, :, di["code"]]),
               ("usage_math", u[:, :, di["math"]]), ("usage_code", u[:, :, di["code"]])]

    per_seq, nll = {}, {}
    t_all = time.time()
    for name, score in configs:
        t0 = time.time()
        high = select_per_layer(score, P_HIGH) if score is not None else None
        if high is not None:
            for l in range(n_layers):
                for e in np.nonzero(~high[l])[0]:
                    patcher.set_expert(l, int(e),
                                       {p: v3[l][p][int(e)] for p in LAYER_PARTS})
        per_seq[name], nll[name] = {}, {}
        for dom in T.DOMAINS:
            tot, n, per, ps = T.eval_nll(eng["model"], tok, evalsets[dom],
                                         max_len=64, batch_size=8, return_per_seq=True)
            per_seq[name][dom] = ps; nll[name][dom] = per
        if high is not None:
            for l in range(n_layers):
                for e in np.nonzero(~high[l])[0]:
                    patcher.restore_expert(l, int(e))
        print(f"[{name}] " + " ".join(f"{d}:{nll[name][d]:.4f}" for d in T.DOMAINS) +
              f" ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    gains, ci, split = {}, {}, {}
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
        for dom in T.DOMAINS:
            ps = per_seq[name][dom]
            ev = list(range(0, len(ps), 2)); od = list(range(1, len(ps), 2))
            split[f"{name}/{dom}"] = abs(agg(ps, ev) - agg(ps, od))

    # general 子集
    sub = {"counts": {k: int(sum(1 for x in subtypes if x == k))
                      for k in sorted(set(subtypes))}, "by_config": {}}
    gen = list(range(len(subtypes)))
    for st, cnt in sub["counts"].items():
        idx = [i for i in gen if subtypes[i] == st]
        if cnt < MIN_SUBTYPE:
            continue
        sub["by_config"].setdefault(st, {})
        for name in nll:
            g_ps = [per_seq["global"]["general"][i] for i in idx]
            c_ps = [per_seq[name]["general"][i] for i in idx]
            gain = agg(g_ps) - agg(c_ps)
            vals = np.empty(N_BOOT)
            for b in range(N_BOOT):
                ii = rng.integers(0, cnt, size=cnt)
                vals[b] = agg(g_ps, ii) - agg(c_ps, ii)
            sub["by_config"][st][name] = {
                "gain": float(gain),
                "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                "nll_global": agg(g_ps), "nll_cfg": agg(c_ps)}

    out = {"n_eval_seq": {d: len(evalsets[d]) for d in T.DOMAINS},
           "configs": [c[0] for c in configs], "nll": nll,
           "gain_vs_global": gains, "ci95": ci, "split_half_abs_delta": split,
           "n_boot": N_BOOT, "general_subtype": sub,
           "per_seq": per_seq, "elapsed_s": round(time.time() - t_all, 1)}
    p = os.path.join(T.RES_DIR, "c3_eval128.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[c3] saved {p} ({time.time()-t_all:.0f}s)", flush=True)
    for name in ("base_int4", "tc_math", "tc_code", "usage_math", "usage_code"):
        print(f"[gain] {name}: " + " ".join(
            f"{d}:{gains[name][d]:+.4f} ci{[round(x,4) for x in ci[name][d]]}"
            for d in T.DOMAINS), flush=True)


if __name__ == "__main__":
    main()
