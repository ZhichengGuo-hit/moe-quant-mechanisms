"""实验3: 等平均位宽下的任务条件化精度配额 vs 全局配额 (H4 核心验证)。
低精度层: INT3 (低位宽底座); 高精度层: INT4; 每层保留 p_high 比例专家为INT4。
配置: base_int4 / global(三域平均敏感度) / tc_math / tc_code / tc_general / random
指标: 三域 eval NLL。
输出: results/allocation.json
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
    """score [L,E] -> 每层选中 frac 比例(高精度INT4)"""
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        idx = np.argsort(-score[l])[:k]
        high[l, idx] = True
    return high


def main():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])            # [L,E,3]
    u = np.array(sm["usage"])
    doms = sm["domains"]

    eng = T.build_engine()
    tok = eng["tok"]
    store = eng["store"]
    patcher = T.ExpertPatcher(store)

    # 低精度底座: INT3 store
    store3 = Int4Store(T.STORE_INT3, mmap_mode="r")
    v3 = {l: store3.layer_views(l) for l in range(store3.num_layers)}

    # 分数: 量化误差影响度 = sum_t w*rel_err (s), 高使用高误差的专家优先保精度
    sc_global = s.sum(-1)
    sc_dom = {doms[i]: s[:, :, i] for i in range(len(doms))}
    rng = np.random.default_rng(7)
    sc_rand = rng.random(s.shape[:2])

    evesets = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval")[:N_EVAL_SEQ]
        evesets[dom] = [r["text"] for r in rows]

    n_layers, n_experts = store.num_layers, store.num_experts

    def run_config(name, score):
        t0 = time.time()
        high = select_per_layer(score, P_HIGH) if score is not None else None
        patched = 0
        if high is not None:
            for l in range(n_layers):
                for e in range(n_experts):
                    if high[l, e]:
                        continue
                    patcher.set_expert(l, e, {p: v3[l][p][e] for p in LAYER_PARTS})
                    patched += 1
        nll = {}
        for dom in T.DOMAINS:
            tot, n, per = T.eval_nll(eng["model"], tok, evesets[dom],
                                     max_len=64, batch_size=8)
            nll[dom] = per
        if high is not None:
            for l in range(n_layers):
                for e in range(n_experts):
                    if not high[l, e]:
                        patcher.restore_expert(l, e)
        print(f"[{name}] patched={patched} nll=" +
              " ".join(f"{d}:{nll[d]:.4f}" for d in T.DOMAINS) +
              f" ({time.time()-t0:.0f}s)", flush=True)
        return {"name": name, "nll": nll, "patched": patched,
                "avg_bits": 3.0 + P_HIGH * 1.0}

    results = []
    results.append(run_config("base_int4", None))
    results.append(run_config("global", sc_global))
    for dom in T.DOMAINS:
        results.append(run_config(f"tc_{dom}", sc_dom[dom]))
    results.append(run_config("random", sc_rand))

    out = {"p_high": P_HIGH, "n_eval_seq": N_EVAL_SEQ,
           "configs": results, "domains": list(T.DOMAINS)}
    p = os.path.join(T.RES_DIR, "allocation.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[alloc] saved {p}", flush=True)


if __name__ == "__main__":
    main()
