"""D6: 共 scale 嵌套量化 (nested / shared-scale) —— 让补丁只花 1 bit。

方案: 底座与补丁共享同一个 4bit 网格 (步长 s4 = amax/7)。
  底座 3bit 码 = u >> 1 (u 为 INT4 nibble 的无符号码 0..15)  -> 偶数格点, 实际步长 2s4 = 0.286·amax
  补丁 1bit   = u & 1  (奇偶偏移)
  合成       = 2·(u>>1) + (u&1) = u, 与 INT4 码逐位相同 -> 补丁域内与全量 INT4 数值完全一致
与"独立 3bit"(maxq=3, 步长 0.333·amax)相比, 底座格点更细 14.3%, 但顶格被截断 (最大码 6s4 = 0.857·amax)。
体积: 底座容器 12.58 GB (12.58 GiB, 3.125bit) + 补丁 1bit×2560 专家 = 0.938 GiB/域。

本实验测:
  1) 底座权重误差: 嵌套 vs 独立 3bit vs INT4 vs "最近偶格点"(质量上限参考)
  2) 构建嵌套底座真实容器 stores/packed3n 并跑 eval64: base3n / base3n+task_math / base3n+task_code
  3) 补丁 1bit 编码的可实现性 (奇偶位逐位相等) 与体积口径
输出: results/d6_nested_scale.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
import safetensors

import bitpack
import tcq_lib as T
import int3_ext
from moe_lib import Int4Store
from exp15_packed_runtime import Packed3Store, LayerMixedExt, select_per_layer

P_HIGH = 0.25
N_LAYERS = bitpack.N_LAYERS
N_EXPERTS = bitpack.N_EXPERTS
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")
PACKED3N_DIR = os.path.join(HERE, "stores", "packed3n")
N_BOOT = 1000
SEED = 91


def nibbles(p):
    """uint8 [rows, cols/2] -> 无符号 4bit 码 [rows, cols] (低半字节在前)"""
    lo = p & 15
    hi = (p >> 4) & 15
    return np.stack([lo, hi], axis=-1).reshape(p.shape[0], p.shape[1] * 2)


def build_nested(store4):
    """从 INT4 nibble store 生成嵌套 3bit 容器 (码 = u>>1, scale = 2·s4)"""
    meta, part_off = bitpack.layout(3)
    os.makedirs(PACKED3N_DIR, exist_ok=True)
    path = os.path.join(PACKED3N_DIR, "experts.bin")
    meta_path = os.path.join(PACKED3N_DIR, "meta.json")
    if os.path.exists(meta_path) and os.path.getsize(path) == meta["total_bytes"]:
        print(f"[build] reuse {path}", flush=True)
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f), 0.0
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(meta["total_bytes"],))
    t0 = time.time()
    for l in range(N_LAYERS):
        v4 = store4.layer_views(l)
        for part in ("gu_q", "dn_q"):
            sp = part[:-1] + "s"
            shape = meta["parts"][part]
            sshape = meta["parts"][sp]
            base_q = part_off[part]
            base_s = part_off[sp]
            q_stride = int(np.prod(shape[1:]))
            s_stride = int(np.prod(sshape[1:]))
            q4 = v4[part].numpy()
            s4 = v4[sp].numpy()
            out_q = np.zeros((N_EXPERTS, q_stride), dtype=np.uint8)
            for e in range(N_EXPERTS):
                u = nibbles(q4[e])
                out_q[e] = bitpack.pack3(u >> 1).reshape(-1)
            out_s = (s4.astype(np.float32) * 2.0).astype(np.float16).reshape(N_EXPERTS, s_stride)
            mm[l * meta["layer_stride"] + base_q:
               l * meta["layer_stride"] + base_q + q_stride * N_EXPERTS] = out_q.reshape(-1)
            mm[l * meta["layer_stride"] + base_s:
               l * meta["layer_stride"] + base_s + s_stride * 2 * N_EXPERTS] = \
                out_s.view(np.uint8).reshape(-1)
        if l % 5 == 0 or l == N_LAYERS - 1:
            print(f"[build] layer {l} {time.time()-t0:.0f}s", flush=True)
    mm.flush()
    del mm
    with open(os.path.join(PACKED3N_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta, time.time() - t0


def weight_error(store4, p3, p3n, n_experts=40, seed=5):
    """抽样专家: 嵌套底座 / 独立底座 / INT4 / 最近偶格点 的相对误差"""
    rng = np.random.default_rng(seed)
    wmap = T.load_weight_map()
    shard = {}

    def get(key):
        sh = wmap[key]
        if sh not in shard:
            shard[sh] = safetensors.safe_open(os.path.join(T.MOE_MODEL_PATH, sh), framework="pt")
        return shard[sh].get_tensor(key)

    rows = []
    t0 = time.time()
    v4c, v3c, v3nc = {}, {}, {}

    def views(cache, store, l):
        if l not in cache:
            cache[l] = store.layer_views(l)
        return cache[l]

    for _ in range(n_experts):
        l = int(rng.integers(0, N_LAYERS))
        e = int(rng.integers(0, N_EXPERTS))
        proj, N, sub = ("gu", bitpack.HIDDEN, "gate_up_proj") if _ % 2 == 0 \
            else ("dn", bitpack.MOE_INTER, "down_proj")
        wbf = get(f"model.language_model.layers.{l}.mlp.experts.{sub}")[e].float().numpy()
        v4 = views(v4c, store4, l)
        u = nibbles(v4[f"{proj}_q"][e].numpy())
        s4 = v4[f"{proj}_s"][e].float().numpy().astype(np.float32)
        a4 = (u.astype(np.float32) - 8.0)
        a4 = a4.reshape(a4.shape[0], -1, bitpack.GROUP) * s4[:, :, None]
        a4 = a4.reshape(wbf.shape)
        v3 = views(v3c, p3, l)
        q3i = bitpack.unpack3(v3[f"{proj}_q"][e].numpy())
        s3i = v3[f"{proj}_s"][e].float().numpy().astype(np.float32)
        a3i = (q3i.astype(np.float32) - 4.0).reshape(q3i.shape[0], -1, bitpack.GROUP) * s3i[:, :, None]
        a3i = a3i.reshape(wbf.shape)
        v3n = views(v3nc, p3n, l)
        q3n = bitpack.unpack3(v3n[f"{proj}_q"][e].numpy())
        s3n = v3n[f"{proj}_s"][e].float().numpy().astype(np.float32)
        a3n = (q3n.astype(np.float32) - 4.0).reshape(q3n.shape[0], -1, bitpack.GROUP) * s3n[:, :, None]
        a3n = a3n.reshape(wbf.shape)
        # 最近偶格点 (质量上限参考, 对应 2bit 偏移方案)
        t = a4 / np.maximum(2.0 * np.repeat(s4, bitpack.GROUP, axis=1), 1e-8)
        qn = np.clip(np.round(t), -4, 3)
        a3ne = (qn.reshape(qn.shape[0], -1, bitpack.GROUP) *
                (2.0 * s4)[:, :, None]).reshape(wbf.shape)
        ref = float(np.linalg.norm(wbf))
        rows.append({"l": l, "e": e, "proj": proj, "ref": ref,
                     "int4": float(np.linalg.norm(a4 - wbf)) / ref,
                     "int3_indep": float(np.linalg.norm(a3i - wbf)) / ref,
                     "int3_nested": float(np.linalg.norm(a3n - wbf)) / ref,
                     "int3_near_even": float(np.linalg.norm(a3ne - wbf)) / ref,
                     "top_clip_frac": float((np.abs(t) >= 3.5).mean())})
    out = {}
    for k in ("int4", "int3_indep", "int3_nested", "int3_near_even", "top_clip_frac"):
        vals = np.array([r[k] for r in rows])
        out[k] = {"mean": float(vals.mean()), "median": float(np.median(vals)),
                  "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90))}
    print(f"[werr] {json.dumps(out, ensure_ascii=False)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def parity_check(store4, n_experts=16, seed=3):
    """奇偶位补丁: 逐位校验 2·(u>>1)+(u&1) == u, 并给出 1bit 打包后的字节数"""
    rng = np.random.default_rng(seed)
    ok = True
    n_bytes = 0
    n_w = 0
    for _ in range(n_experts):
        l = int(rng.integers(0, N_LAYERS))
        e = int(rng.integers(0, N_EXPERTS))
        proj = "gu" if _ % 2 == 0 else "dn"
        v4 = store4.layer_views(l)
        u = nibbles(v4[f"{proj}_q"][e].numpy())
        b = (u & 1).astype(np.uint8)
        rec = (2 * (u >> 1) + b)
        ok &= bool((rec == u).all())
        n_bytes += int(np.packbits(b.reshape(-1)).nbytes)
        n_w += int(b.size)
    return {"exact_roundtrip": ok, "bits_per_weight": n_bytes * 8 / n_w,
            "patch_gib_per_domain": 2560 * 3 * bitpack.HIDDEN * bitpack.MOE_INTER * 1 / 8 / 2 ** 30,
            "n_sampled": n_experts}


def agg(per_seq, idx=None):
    if idx is None:
        idx = range(len(per_seq))
    return sum(per_seq[i][0] for i in idx) / max(sum(per_seq[i][1] for i in idx), 1)


def main():
    t_all = time.time()
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])
    doms, di = sm["domains"], {d: i for i, d in enumerate(sm["domains"])}

    ext = int3_ext.load_ext(threads=16)
    eng = T.build_engine()
    model, tok = eng["model"], eng["tok"]
    E = eng["store"].num_experts
    store4 = eng["store"]

    p3n_meta, build_s = build_nested(store4)
    p3n = Packed3Store(PACKED3N_DIR)
    p3i = Packed3Store()
    print(f"[build] done {build_s:.0f}s, {p3n_meta['total_bytes']/2**30:.2f} GiB", flush=True)

    werr = weight_error(store4, p3i, p3n)
    pchk = parity_check(store4)

    p4s = {l: store4.layer_views(l) for l in range(N_LAYERS)}

    def install(mask, store):
        for l in range(N_LAYERS):
            pm = np.full(E, -1, dtype=np.int32)
            if mask is not None:
                pm[mask[l]] = np.nonzero(mask[l])[0]
            model.model.language_model.layers[l].mlp.direct_ext = LayerMixedExt(
                ext, store.layer_views(l), p4s[l], torch.from_numpy(pm))

    evalsets = {d: [r["text"] for r in T.load_domain(d, "eval64")] for d in T.DOMAINS}
    masks = {"task_math": select_per_layer(s[:, :, di["math"]], P_HIGH),
             "task_code": select_per_layer(s[:, :, di["code"]], P_HIGH)}

    nll, per_seq, timing = {}, {}, {}
    for name, mask in (("base3n", None), ("nest_math", masks["task_math"]),
                       ("nest_code", masks["task_code"])):
        t1 = time.time()
        install(mask, p3n)
        per_seq[name], nll[name] = {}, {}
        ntok = 0
        for dom in T.DOMAINS:
            tot, n, per, ps = T.eval_nll(model, tok, evalsets[dom], max_len=64,
                                         batch_size=8, return_per_seq=True)
            per_seq[name][dom] = ps; nll[name][dom] = per; ntok += n
        dt = time.time() - t1
        timing[name] = {"eval_s": round(dt, 1), "tok_per_s": round(ntok / dt, 2)}
        print(f"[{name}] " + " ".join(f"{d}:{nll[name][d]:.4f}" for d in T.DOMAINS) +
              f" ({dt:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    ci = {}
    for name in ("nest_math", "nest_code"):
        ci[name] = {}
        for dom in T.DOMAINS:
            gp, cp = per_seq["base3n"][dom], per_seq[name][dom]
            vals = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, len(gp), len(gp))
                vals[b] = agg(gp, idx) - agg(cp, idx)
            ci[name][dom] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    n_w = N_LAYERS * N_EXPERTS * 3 * bitpack.HIDDEN * bitpack.MOE_INTER
    base_gib = p3n_meta["total_bytes"] / 2 ** 30
    patch_gib = 2560 * 3 * bitpack.HIDDEN * bitpack.MOE_INTER * 1 / 8 / 2 ** 30
    union_n = 5018
    union_gib = union_n * 3 * bitpack.HIDDEN * bitpack.MOE_INTER * 1 / 8 / 2 ** 30
    out = {"weight_error": werr, "parity": pchk,
           "nll": nll, "ci95_vs_base3n": ci, "n_boot": N_BOOT, "timing": timing,
           "container": {"base3n_gib": round(base_gib, 2),
                         "patch_1bit_gib_per_domain": round(patch_gib, 3),
                         "single_domain_gib": round(base_gib + patch_gib, 2),
                         "eff_bits_single": round((base_gib + patch_gib) * 8 * 2 ** 30 / n_w, 3),
                         "union3_gib": round(union_gib, 3),
                         "eff_bits_union3": round((base_gib + union_gib) * 8 * 2 ** 30 / n_w, 3),
                         "n_high": 2560, "n_union3": union_n},
           "reference_prev": {"base3_nll": {"math": 1.7389, "code": 1.0039, "general": 1.8026},
                              "task_math_nll": {"math": 1.6557, "code": 0.8444, "general": 1.8171},
                              "task_code_nll": {"math": 1.7040, "code": 0.7176, "general": 1.8174},
                              "base_int4_nll": {"math": 1.6346, "code": 0.6955, "general": 1.8104}},
           "elapsed_s": round(time.time() - t_all, 1)}
    p = os.path.join(T.RES_DIR, "d6_nested_scale.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[d6] saved {p} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
