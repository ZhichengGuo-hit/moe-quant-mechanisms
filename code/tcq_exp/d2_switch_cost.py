"""D2(改): 零写入换装与合批代价 —— 取代 LoRA 路线的工程实验。

背景(D1 结论): 补丁增量 W_int4−W_int3 是白噪声型(rank-16 只恢复 5% 能量),
LoRA/低秩补丁路线不成立。但 A 线的运行时其实同时驻留两套容器:
  - packed3 密排 INT3 底座 (11.72 GiB, 全部 10240 专家)
  - INT4 nibble store      (15.47 GiB, 全部 10240 专家)
切换域 = 重写每层 patch_map(int32[E]) —— 纯元数据, **零权重写入**。
本实验量化:
  1) 换装延迟 (40 层 map 更新, µs)
  2) 分组批 vs 交错批的吞吐差 (同样 token 数, 前者只切 1 次, 后者切 15 次)
  3) 混批精度代价 (同一 batch 里 4 math + 4 code, 分别用 math/code 补丁时的分域 NLL)
输出: results/d2_switch_cost.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch

import tcq_lib as T
import int3_ext
from exp15_packed_runtime import Packed3Store, LayerMixedExt, select_per_layer

P_HIGH = 0.25
MAX_LEN = 64


def main():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])
    doms, di = sm["domains"], {d: i for i, d in enumerate(sm["domains"])}

    ext = int3_ext.load_ext(threads=16)
    eng = T.build_engine()
    model, tok = eng["model"], eng["tok"]
    n_layers = len(model.model.language_model.layers)
    E = eng["store"].num_experts
    p3s = Packed3Store()
    store4 = eng["store"]
    p4s = {l: store4.layer_views(l) for l in range(n_layers)}

    masks = {"math": select_per_layer(s[:, :, di["math"]], P_HIGH),
             "code": select_per_layer(s[:, :, di["code"]], P_HIGH)}

    def install(mask, layer_slice=None):
        ls = range(n_layers) if layer_slice is None else layer_slice
        for l in ls:
            pm = np.full(E, -1, dtype=np.int32)
            if mask is not None:
                pm[mask[l]] = np.nonzero(mask[l])[0]
            model.model.language_model.layers[l].mlp.direct_ext = LayerMixedExt(
                ext, p3s.layer_views(l), p4s[l], torch.from_numpy(pm))

    # ---- 1) 换装延迟 ----
    sw = {"math_to_code": [], "code_to_math": [], "to_base": []}
    for _ in range(30):
        t0 = time.perf_counter(); install(masks["math"]); t1 = time.perf_counter()
        install(masks["code"]); t2 = time.perf_counter()
        install(None); t3 = time.perf_counter()
        sw["math_to_code"].append((t1 - t0) * 1e6)
        sw["code_to_math"].append((t2 - t1) * 1e6)
        sw["to_base"].append((t3 - t2) * 1e6)
    switch_us = {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
                 for k, v in sw.items()}

    # ---- 数据 ----
    data = {}
    for d in T.DOMAINS:
        data[d] = [r["text"] for r in T.load_domain(d, "eval64")]

    def eval_dom(dom, n=None, bs=8):
        t0 = time.time()
        tot, nt, per = T.eval_nll(model, tok, data[dom][:n] if n else data[dom],
                                  max_len=MAX_LEN, batch_size=bs)
        return per, nt, time.time() - t0

    # ---- 2) 分组批 vs 交错批 ----
    n_each = 32
    t0 = time.time()
    install(masks["math"]); _, n_m, _ = eval_dom("math", n_each)
    install(masks["code"]); _, n_c, _ = eval_dom("code", n_each)
    grouped_s = time.time() - t0
    t0 = time.time()
    for _ in range(n_each // 8):  # 重复同一块以保持 token 量与分组批对齐, 只测换装开销
        install(masks["math"])
        eval_dom("math", 8)
        install(masks["code"])
        eval_dom("code", 8)
    interleaved_s = time.time() - t0

    thru = {"grouped_s": grouped_s, "interleaved_s": interleaved_s,
            "tokens": n_m + n_c,
            "grouped_tok_s": (n_m + n_c) / grouped_s,
            "interleaved_tok_s": (n_m + n_c) / interleaved_s,
            "overhead_ratio": interleaved_s / grouped_s - 1.0,
            "n_switches_each": 2 * (n_each // 8)}

    # ---- 3) 混批精度代价 ----
    def eval_pairs(dom_list, bs_pairs_per_dom=4):
        texts, tags = [], []
        for d in dom_list:
            texts += data[d][:bs_pairs_per_dom]
            tags += [d] * bs_pairs_per_dom
        tot, nt, per = T.eval_nll(model, tok, texts, max_len=MAX_LEN,
                                  batch_size=len(texts))
        return per, nt

    mixed = {}
    for patch in ("math", "code", "base3"):
        install(None if patch == "base3" else masks[patch])
        per, nt = eval_pairs(["math", "code"])
        mixed[patch] = {"nll_mixed_batch": per, "tokens": nt}
    for patch in ("math", "code"):
        install(masks[patch])
        per_m, nt_m, _ = eval_dom("math", 4, bs=4)
        per_c, nt_c, _ = eval_dom("code", 4, bs=4)
        mixed[f"{patch}_single_math"] = {"nll": per_m, "tokens": nt_m}
        mixed[f"{patch}_single_code"] = {"nll": per_c, "tokens": nt_c}

    # ---- 4) 单域对照 NLL (eval64, 用于回收率口径) ----
    nll = {}
    for name, mask in (("base3", None), ("task_math", masks["math"]),
                       ("task_code", masks["code"])):
        install(mask)
        row = {}
        for d in T.DOMAINS:
            per, _, _ = eval_dom(d, 16)
            row[d] = per
        nll[name] = row
        print(f"[{name}] " + " ".join(f"{d}:{row[d]:.4f}" for d in T.DOMAINS), flush=True)

    out = {"switch_us": switch_us, "throughput": thru, "mixed_batch": mixed,
           "nll_eval16": nll, "max_len": MAX_LEN, "p_high": P_HIGH,
           "note": ("切换只更新 40 层的 patch_map(int32[E]), 不写任何权重; "
                    "两套容器 (packed3 11.72GiB + int4 15.47GiB) 同时常驻")}
    p = os.path.join(T.RES_DIR, "d2_switch_cost.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[d2] switch_us {json.dumps(switch_us)}", flush=True)
    print(f"[d2] throughput {json.dumps(thru)}", flush=True)
    print(f"[d2] mixed {json.dumps(mixed)}", flush=True)
    print(f"[d2] saved {p}", flush=True)


if __name__ == "__main__":
    main()
