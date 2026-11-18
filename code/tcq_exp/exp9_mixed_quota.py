"""E9: 非等比例配额与混合校准 (H4 延伸)。

低精度 INT3 底座, 每层保留 64 个专家为 INT4 (25%, 总 2560), 与 E3 等位宽。
配置:
  base_int4        全 INT4 (无损上界)
  global           三域 s 求和
  realloc_mathcode 每层 math 32 + code 32, general 0 (名额从 general 转给两敏感域)
  mixed_l25/l50/l75 score = λ·s_math + (1−λ)·s_code
  task_math/task_code 单域 25% (应与 E3 的 tc_math/tc_code 在噪声内一致)
输出: results/mixed_quota.json
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
N_EVAL_SEQ = 8
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")


def select_per_layer(score, frac):
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        idx = np.argsort(-score[l])[:k]
        high[l, idx] = True
    return high


def select_realloc(s_math, s_code, k_each=32):
    """每层 math top-k_each ∪ code top-k_each, 不足 2*k_each 时用两域最大分数补齐"""
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


def main():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])            # [L,E,3]
    doms = sm["domains"]
    di = {d: i for i, d in enumerate(doms)}

    eng = T.build_engine()
    tok = eng["tok"]
    patcher = T.ExpertPatcher(eng["store"])
    store3 = Int4Store(T.STORE_INT3, mmap_mode="r")
    v3 = {l: store3.layer_views(l) for l in range(store3.num_layers)}

    evalsets = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval")[:N_EVAL_SEQ]
        evalsets[dom] = [r["text"] for r in rows]

    n_layers, n_experts = store3.num_layers, store3.num_experts

    def run(name, spec):
        """spec: None=全INT4; bool[L,E]=高精度掩码; float[L,E]=按分数取 top-P_HIGH"""
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
        nll = {}
        for dom in T.DOMAINS:
            tot, n, per = T.eval_nll(eng["model"], tok, evalsets[dom],
                                     max_len=64, batch_size=8)
            nll[dom] = per
        if high is not None:
            for l in range(n_layers):
                for e in np.nonzero(~high[l])[0]:
                    patcher.restore_expert(l, int(e))
        print(f"[{name}] patched={patched} nll=" +
              " ".join(f"{d}:{nll[d]:.4f}" for d in T.DOMAINS) +
              f" ({time.time()-t0:.0f}s)", flush=True)
        return {"name": name, "nll": nll, "patched": patched,
                "avg_bits": 3.0 + P_HIGH * 1.0}

    results = []
    results.append(run("base_int4", None))
    results.append(run("global", s.sum(-1)))
    results.append(run("realloc_mathcode",
                       select_realloc(s[:, :, di["math"]], s[:, :, di["code"]])))
    for lam in (0.25, 0.5, 0.75):
        score = lam * s[:, :, di["math"]] + (1 - lam) * s[:, :, di["code"]]
        results.append(run(f"mixed_l{int(lam*100)}", score))
    for dom in ("math", "code"):
        results.append(run(f"task_{dom}", s[:, :, di[dom]]))

    glob = next(c for c in results if c["name"] == "global")
    ana = {"gain_vs_global": {}, "net_task_weighted": {}}
    for c in results:
        g = {d: glob["nll"][d] - c["nll"][d] for d in T.DOMAINS}
        ana["gain_vs_global"][c["name"]] = g
        others = [d for d in T.DOMAINS if d not in ("math", "code")]
        if c["name"] in ("task_math", "task_code"):
            tgt = c["name"].split("_")[1]
            in_gain = g[tgt]
            out_cost = float(np.mean([-g[d] for d in T.DOMAINS if d != tgt]))
        elif c["name"] in ("realloc_mathcode", "mixed_l25", "mixed_l50", "mixed_l75"):
            in_gain = float(np.mean([g["math"], g["code"]]))
            out_cost = float(np.mean([-g[d] for d in others]))
        else:
            continue
        ana["net_task_weighted"][c["name"]] = {
            "in_domain_gain": in_gain, "out_domain_cost": out_cost,
            "net": in_gain - out_cost}

    out = {"p_high": P_HIGH, "n_eval_seq": N_EVAL_SEQ, "domains": list(T.DOMAINS),
           "configs": results, "analysis": ana}
    p = os.path.join(T.RES_DIR, "mixed_quota.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[mixed_quota] saved {p}", flush=True)
    print(json.dumps(ana["net_task_weighted"], ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
