"""E18: Pareto 扫描 —— 2bit 底座 + INT4 补丁行的体积–NLL 前沿。

问题:
  1) 11.8 GiB 工作点是否唯一（同体积还有没有更优配置）;
  2) 配额降到 10% 能否再砍体积而质量不掉队.

做法: 复用 A1 的运行时容器 (stores/packed2, int2_ext 混合内核, 2bit 密排底座 +
INT4 整行补丁), 只改每层高精度专家数 k = frac·256, 扫 frac ∈ {0,5,10,15,25}%,
选择器 ∈ {global, task_math, task_code}; eval64 (三域 × 64 序列)。
体积按两种口径同时记录:
  keep-both = 底座全量 + 补丁整行另存 (与运行时双容器一致);
  replace   = 底座中补丁专家的 2bit 行被丢弃 (一次性重打包可达).

    python exp18_pareto.py --threads 8 --cpus 16-31
输出: results/pareto.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch

import bitpack
import tcq_lib as T
import int2_ext
from a1_two_bit_runtime import Packed2Store, LayerMixed2Ext, select_per_layer

SEED = 91
FRACS = (0.0, 0.05, 0.10, 0.15, 0.25)
SELECTORS = ("global", "task_math", "task_code")
# 必须用 4 级 (sym4) 容器: 内核按 (q-1.5)*s 解码; stores/packed2 是 3 级码,
# 解码偏置 0.5*s, 逐层放大到 inf -> 全 NaN (见 results/_diag_store.log)
PACKED2_DIR = os.path.join(HERE, "stores", "packed2_sym4")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpus", default=None)
    ap.add_argument("--slots", type=int, default=512)
    a = ap.parse_args(argv)
    if a.cpus:
        from pin import pin_self
        print(f"[pin] {pin_self(cpus=a.cpus)}", flush=True)

    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])
    doms, di = sm["domains"], {d: i for i, d in enumerate(sm["domains"])}

    ext = int2_ext.load_ext(threads=a.threads)
    print("[ext] int2_ext ready", flush=True)
    eng = T.build_engine(slots=a.slots, cpu_threads=a.threads)
    tok, model = eng["tok"], eng["model"]
    n_layers = len(model.model.language_model.layers)
    E = eng["store"].num_experts

    p2s = Packed2Store(PACKED2_DIR)
    p4s = {l: eng["store"].layer_views(l) for l in range(n_layers)}

    def install(mask):
        for l in range(n_layers):
            pm = np.full(E, -1, dtype=np.int32)
            if mask is not None:
                pm[mask[l]] = np.nonzero(mask[l])[0]
            model.model.language_model.layers[l].mlp.direct_ext = LayerMixed2Ext(
                ext, p2s.layer_views(l), p4s[l], torch.from_numpy(pm))

    evalsets = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval64")
        evalsets[dom] = [r["text"] for r in rows]
        print(f"[data] {dom}: {len(evalsets[dom])} seq", flush=True)

    # ---- 体积口径 ----
    N_W = bitpack.N_LAYERS * bitpack.N_EXPERTS * 3 * bitpack.HIDDEN * bitpack.MOE_INTER
    base_gib = p2s.meta["total_bytes"] / 2 ** 30
    # row4_gib: 全部专家按 4bit 存储的体积; frac 配额对应其 frac 比例
    row4_gib = bitpack.N_LAYERS * bitpack.N_EXPERTS * 3 * bitpack.HIDDEN * bitpack.MOE_INTER * 4 / 8 / 2 ** 30
    row3_gib = row4_gib * 0.75
    row2_gib = row4_gib * 0.5
    INT4_GIB = 15.58  # 真实 nibble 容器口径 (见 §6.3)

    def volume(frac):
        keep = base_gib + frac * row4_gib
        repl = base_gib - frac * row2_gib + frac * row4_gib
        return {"keep_both": round(keep, 2), "replace": round(repl, 2)}

    results = []
    t_all = time.time()
    for frac in FRACS:
        for sel in SELECTORS:
            score = s.sum(-1) if sel == "global" else s[:, :, di[sel.split("_")[1]]]
            mask = None if frac == 0.0 else select_per_layer(score, frac)
            install(mask)
            t1 = time.time()
            nll, per_seq = {}, {}
            for dom in T.DOMAINS:
                tot, n, per, ps = T.eval_nll(model, tok, evalsets[dom], max_len=64,
                                             batch_size=8, return_per_seq=True)
                nll[dom] = per
                per_seq[dom] = ps
            name = f"base2_q{int(frac*100)}_{sel}" if frac else "base2"
            rec = {"name": name, "base_bits": 2, "frac": frac, "selector": sel,
                   "nll": nll, "volume": volume(frac),
                   "eval_s": round(time.time() - t1, 1)}
            results.append(rec)
            print(f"[{name}] " + " ".join(f"{d}:{nll[d]:.4f}" for d in T.DOMAINS) +
                  f" vol_keep={rec['volume']['keep_both']} GiB ({rec['eval_s']}s)", flush=True)
            if not all(np.isfinite(nll[d]) for d in T.DOMAINS):
                print("[abort] NLL 非有限, 停止扫描 (容器/内核不匹配)", flush=True)
                sys.exit(2)
            if frac == 0.0:
                break  # frac=0 与选择器无关, 只跑一次

    # ---- 既有同口径参考点 (base3 / INT4), 用于完整前沿 ----
    ref = []
    try:
        with open(os.path.join(T.RES_DIR, "packed_runtime.json"), encoding="utf-8") as f:
            pr = json.load(f)
        for name in ("base3", "global", "task_math", "task_code"):
            if name in pr["nll"]:
                ref.append({"name": f"base3_{name}" if name != "base3" else "base3",
                            "base_bits": 3, "frac": 0.25 if name != "base3" else 0.0,
                            "nll": pr["nll"][name],
                            "volume": {"keep_both": round(11.72 + (0.25 * row4_gib if name != "base3" else 0.0), 2),
                                       "replace": round(11.72 + (0.25 * (row4_gib - row3_gib) if name != "base3" else 0.0), 2)}})
    except Exception as e:  # noqa: BLE001
        print(f"[ref] packed_runtime 读取失败 {e}", flush=True)
    try:
        with open(os.path.join(T.RES_DIR, "large_eval.json"), encoding="utf-8") as f:
            le = json.load(f)
        if "base_int4" in le["nll"]:
            ref.append({"name": "int4", "base_bits": 4, "frac": 0.0,
                        "nll": le["nll"]["base_int4"],
                        "volume": {"keep_both": INT4_GIB, "replace": INT4_GIB}})
    except Exception as e:  # noqa: BLE001
        print(f"[ref] large_eval 读取失败 {e}", flush=True)

    allp = results + ref
    # 前沿: 按 keep_both 体积升序, 保留 NLL(三域均值) 递减的点
    front = []
    best = float("inf")
    for r in sorted(allp, key=lambda x: x["volume"]["keep_both"]):
        m = float(np.mean([r["nll"][d] for d in T.DOMAINS]))
        r["nll_mean"] = m
        if m < best:
            best = m
            front.append(r["name"])

    out = {"n_eval_seq": {d: len(evalsets[d]) for d in T.DOMAINS},
           "fracs": list(FRACS), "selectors": list(SELECTORS),
           "store": os.path.basename(PACKED2_DIR),
           "configs": allp, "frontier": front,
           "volume_basis": {"base2_gib": round(base_gib, 2),
                            "row4_gib_per_all": round(row4_gib, 2),
                            "int4_gib": INT4_GIB},
           "elapsed_s": round(time.time() - t_all, 1)}
    p = os.path.join(T.RES_DIR, "pareto.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[pareto] saved {p} frontier={front}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
