"""E13: 混合配额 eval64 复测 —— 判定"单一共享混合补丁"能否成为论文级结论。

沿用 E9 的 8 个配置, 但评测口径升级到 64 条/域 + 1000 次序列级 bootstrap 95% CI。
配置 (P_HIGH=0.25, INT3 底座 + 每层 64 个 INT4):
  base_int4        全 INT4 (无损上界)
  global           三域 s 求和
  realloc_mathcode 每层 math 32 + code 32, general 0
  mixed_l25/l50/l75 score = λ·s_math + (1−λ)·s_code
  task_math/task_code 单域 25%
输出: results/mixed_eval64.json
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
SEED = 11


def select_per_layer(score, frac):
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        high[l, np.argsort(-score[l])[:k]] = True
    return high


def select_realloc(s_math, s_code, k_each=32):
    L, E = s_math.shape
    total = 2 * k_each
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        im = np.argsort(-s_math[l])[:k_each]
        ic = np.argsort(-s_code[l])[:k_each]
        high[l, np.union1d(im, ic)] = True
        if high[l].sum() < total:
            rest = np.nonzero(~high[l])[0]
            key = np.maximum(s_math[l, rest], s_code[l, rest])
            fill = rest[np.argsort(-key)[:total - high[l].sum()]]
            high[l, fill] = True
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
        ("realloc_mathcode",
         select_realloc(s[:, :, di["math"]], s[:, :, di["code"]])),
        ("mixed_l25", 0.25 * s[:, :, di["math"]] + 0.75 * s[:, :, di["code"]]),
        ("mixed_l50", 0.50 * s[:, :, di["math"]] + 0.50 * s[:, :, di["code"]]),
        ("mixed_l75", 0.75 * s[:, :, di["math"]] + 0.25 * s[:, :, di["code"]]),
        ("task_math", s[:, :, di["math"]]),
        ("task_code", s[:, :, di["code"]]),
    ]

    per_seq, nll = {}, {}
    t_all = time.time()
    for name, spec in configs:
        t0 = time.time()
        if spec is None:
            high = None
        elif spec.dtype == bool:
            high = spec
        else:
            high = select_per_layer(spec, P_HIGH)
        patched = 0
        if high is not None:
            for l in range(n_layers):
                for e in np.nonzero(~high[l])[0]:
                    patcher.set_expert(l, int(e),
                                       {p: v3[l][p][int(e)] for p in LAYER_PARTS})
                    patched += 1
        per_seq[name], nll[name] = {}, {}
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
        print(f"[{name}] patched={patched} nll=" +
              " ".join(f"{d}:{nll[name][d]:.4f}" for d in T.DOMAINS) +
              f" ({time.time()-t0:.0f}s)", flush=True)

    # ---- 噪声标尺: 奇偶两半 ----
    split = {}
    for name in nll:
        for dom in T.DOMAINS:
            ps = per_seq[name][dom]
            ev = list(range(0, len(ps), 2))
            od = list(range(1, len(ps), 2))
            split[f"{name}/{dom}"] = abs(agg(ps, ev) - agg(ps, od))

    # ---- 相对 global 的增益 + bootstrap CI ----
    rng = np.random.default_rng(SEED)
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

    # ---- 判定分析: 域内增益 / 域外代价 (点估计 + paired CI) ----
    ana = {"vs_global": {}, "mixed_vs_task": {}}
    for name in nll:
        if name == "global":
            continue
        g = gains[name]
        if name in ("task_math", "task_code"):
            tgt = name.split("_")[1]
            in_gain = g[tgt]
            out_cost = float(np.mean([-g[d] for d in T.DOMAINS if d != tgt]))
        elif name in ("realloc_mathcode", "mixed_l25", "mixed_l50", "mixed_l75"):
            in_gain = float(np.mean([g["math"], g["code"]]))
            out_cost = float(np.mean([-g[d] for d in T.DOMAINS
                                      if d not in ("math", "code")]))
        else:
            continue
        ana["vs_global"][name] = {
            "in_domain_gain": in_gain, "out_domain_cost": out_cost,
            "net": in_gain - out_cost,
            "ci": {d: ci[name][d] for d in T.DOMAINS}}

    # 混合/重分配 与 双任务 oracle (task_math+task_code 平均) 的差距
    oracle_in = 0.5 * (nll["task_math"]["math"] + nll["task_code"]["code"])
    oracle_out = 0.5 * (nll["task_math"]["code"] + nll["task_code"]["math"])
    for name in ("realloc_mathcode", "mixed_l25", "mixed_l50", "mixed_l75"):
        in_d = 0.5 * (nll[name]["math"] + nll[name]["code"])
        out_d = 0.5 * (nll[name]["code"] + nll[name]["math"])
        ana["mixed_vs_task"][name] = {
            "in_domain_nll": in_d, "oracle_two_task_nll": oracle_in,
            "in_domain_shortfall": in_d - oracle_in,
            "out_domain_nll": out_d,
            "out_domain_penalty": out_d - oracle_out}

    out = {"p_high": P_HIGH,
           "n_eval_seq": {d: len(evalsets[d]) for d in T.DOMAINS},
           "configs": [c[0] for c in configs],
           "nll": nll, "gain_vs_global": gains, "ci95": ci,
           "analysis": ana, "split_half_abs_delta": split,
           "n_boot": N_BOOT, "per_seq": per_seq,
           "elapsed_s": round(time.time() - t_all, 1)}

    p = os.path.join(T.RES_DIR, "mixed_eval64.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[mixed_eval64] saved {p} ({time.time()-t_all:.0f}s)", flush=True)
    print(json.dumps(ana, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
