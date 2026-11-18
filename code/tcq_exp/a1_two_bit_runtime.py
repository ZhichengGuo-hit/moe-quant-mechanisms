"""A1: 2bit 真实容器的运行时接入 + 内核吞吐测量。

1) 新内核 moe_ffn_mixed2_batch (2bit 密排底座 + INT4 补丁行), 直接读 stores/packed2
   (7.97 GiB) 容器, 不走 nibble 反量化;
2) 配置: base2 (全 2bit) / global / task_math / task_code (25% INT4 补丁);
   eval64 + 1000 次序列级 bootstrap; 与 E15 的 3bit 结果并列;
3) 吞吐: 记录每种配置的 eval 时间 (token/s) 与 int3/int2 内核单次调用延迟。
输出: results/a1_two_bit.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch

import bitpack
import tcq_lib as T
import int2_ext
from moe_lib import Int4Store

P_HIGH = 0.25
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")
N_BOOT = 1000
SEED = 83
PACKED2_DIR = os.path.join(HERE, "stores", "packed2")


def select_per_layer(score, frac):
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        high[l, np.argsort(-score[l])[:k]] = True
    return high


class Packed2Store:
    def __init__(self, path=PACKED2_DIR):
        meta, part_off = bitpack.layout(2)
        self.meta, self.part_off = meta, part_off
        self.stride = meta["layer_stride"]
        self.parts = meta["parts"]
        self._mm = np.memmap(os.path.join(path, "experts.bin"), dtype=np.uint8,
                             mode="r", shape=(meta["total_bytes"],))
        self._keep = {}

    def layer_views(self, layer):
        out = {}
        for p, shape in self.parts.items():
            off = layer * self.stride + self.part_off[p]
            n = int(np.prod(shape))
            nbytes = n * (2 if p.endswith("_s") else 1)
            arr = self._mm[off:off + nbytes].view(np.float16 if p.endswith("_s") else np.uint8)
            t = torch.from_numpy(arr.reshape(shape))
            self._keep[(layer, p)] = t
            out[p] = t
        return out


class LayerMixed2Ext:
    def __init__(self, ext, p2, p4, patch_map):
        self.ext, self.p2, self.p4, self.pmap = ext, p2, p4, patch_map

    def moe_ffn_int4_batch(self, gu_q, gu_s, dn_q, dn_s, payload, out=None):
        return self.ext.moe_ffn_mixed2_batch(
            self.p4["gu_q"], self.p4["gu_s"], self.p4["dn_q"], self.p4["dn_s"],
            self.p2["gu_q"], self.p2["gu_s"], self.p2["dn_q"], self.p2["dn_s"],
            self.pmap, payload, out)

    def moe_ffn_int4(self, gu_q, gu_s, dn_q, dn_s, payload, out=None):
        return self.moe_ffn_int4_batch(gu_q, gu_s, dn_q, dn_s,
                                       payload.unsqueeze(0), None).squeeze(0)


def deq_nibble(q, s, N, group=bitpack.GROUP):
    """INT4 nibble [rows, N/2] uint8 + fp16 scale -> float32 [rows, N]"""
    lo = q & 15
    hi = (q >> 4) & 15
    v = torch.stack([lo, hi], dim=-1).reshape(q.shape[0], N)
    v = v.to(torch.float32) - 8.0
    v = v.reshape(q.shape[0], N // group, group) * s.float().unsqueeze(-1)
    return v.reshape(q.shape[0], N)


def numeric_check(ext, p2v, p4v, K=8, n_patch=4, seed=7, sym4=False):
    """用 torch 反量化参考实现校验 int2/int4 混合内核 (单层, 8 个专家)"""
    torch.manual_seed(seed)
    H = p2v["gu_q"].shape[2] * 4
    I = p2v["dn_q"].shape[2] * 4
    O2 = p2v["gu_q"].shape[1]
    dq2 = (lambda q, s: bitpack.dequant2_group(q, s)) if sym4 else \
          (lambda q, s: bitpack.dequant_group(q, s, 2))

    def dq_base(e, proj):
        q = p2v[f"{proj}_q"][e].numpy()
        s = p2v[f"{proj}_s"][e].float()
        return dq2(torch.from_numpy(bitpack.unpack2(q)), s)

    def dq_patch(e, proj, N):
        return deq_nibble(p4v[f"{proj}_q"][e], p4v[f"{proj}_s"][e], N)

    ids = torch.arange(K)
    wts = torch.rand(K) * 0.3
    payload = torch.zeros(1, H + 2 * K, dtype=torch.float16)
    payload[0, :H] = torch.randn(H).half()
    payload[0, H:H + K] = ids.half()
    payload[0, H + K:] = wts.half()
    pm = torch.full((p2v["gu_q"].shape[0],), -1, dtype=torch.int32)
    pm[:n_patch] = torch.arange(n_patch, dtype=torch.int32)
    out = ext.moe_ffn_mixed2_batch(p4v["gu_q"], p4v["gu_s"], p4v["dn_q"], p4v["dn_s"],
                                   p2v["gu_q"], p2v["gu_s"], p2v["dn_q"], p2v["dn_s"],
                                   pm, payload, None)

    x = payload[0, :H].float()
    acc = torch.zeros(H)
    for k in range(K):
        e = int(ids[k])
        g = (dq_patch(e, "gu", H) if e < n_patch else dq_base(e, "gu")) @ x
        act = torch.nn.functional.silu(g[:I]) * g[I:O2]
        d = dq_patch(e, "dn", I) if e < n_patch else dq_base(e, "dn")
        acc += float(payload[0, H + K + k]) * (d @ act)
    ref = acc
    got = out[0, :H]
    scale = float(ref.abs().max()) + 1e-9
    return {"max_abs_diff": float((ref - got).abs().max()),
            "rel_max_abs_diff": float((ref - got).abs().max()) / scale,
            "cosine": float(torch.nn.functional.cosine_similarity(ref, got, dim=0)),
            "n_patch": n_patch, "K": K}


def agg(per_seq, idx=None):
    if idx is None:
        idx = range(len(per_seq))
    s = sum(per_seq[i][0] for i in idx)
    n = sum(per_seq[i][1] for i in idx)
    return s / max(n, 1)


def main():
    store_dir = os.environ.get("A1_STORE", PACKED2_DIR)
    out_name = os.environ.get("A1_OUT", "a1_two_bit.json")
    sym4 = os.environ.get("A1_SYM4", "0") == "1"
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])
    doms, di = sm["domains"], {d: i for i, d in enumerate(sm["domains"])}

    t0 = time.time()
    ext = int2_ext.load_ext(threads=16)
    print(f"[ext] int2_ext loaded {time.time()-t0:.0f}s", flush=True)
    eng = T.build_engine()
    tok, model = eng["tok"], eng["model"]
    n_layers = len(model.model.language_model.layers)
    E = eng["store"].num_experts

    p2s = Packed2Store(store_dir)
    store4 = eng["store"]
    p4s = {l: store4.layer_views(l) for l in range(n_layers)}

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

    configs = [("base2", None),
               ("global", select_per_layer(s.sum(-1), P_HIGH)),
               ("task_math", select_per_layer(s[:, :, di["math"]], P_HIGH)),
               ("task_code", select_per_layer(s[:, :, di["code"]], P_HIGH))]

    nll, per_seq, timing = {}, {}, {}
    t_all = time.time()
    for name, mask in configs:
        t1 = time.time()
        install(mask)
        per_seq[name], nll[name] = {}, {}
        ntok = 0
        for dom in T.DOMAINS:
            tot, n, per, ps = T.eval_nll(model, tok, evalsets[dom], max_len=64,
                                         batch_size=8, return_per_seq=True)
            per_seq[name][dom] = ps; nll[name][dom] = per; ntok += n
        dt = time.time() - t1
        timing[name] = {"eval_s": round(dt, 1), "tokens": int(ntok),
                        "tok_per_s": round(ntok / dt, 2)}
        print(f"[{name}] " + " ".join(f"{d}:{nll[name][d]:.4f}" for d in T.DOMAINS) +
              f" ({dt:.0f}s, {ntok/dt:.2f} tok/s)", flush=True)

    rng = np.random.default_rng(SEED)
    gains, ci = {}, {}
    for name in nll:
        gains[name] = {d: nll["global"][d] - nll[name][d] for d in T.DOMAINS}
        ci[name] = {}
        for dom in T.DOMAINS:
            gp, cp = per_seq["global"][dom], per_seq[name][dom]
            vals = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, len(gp), len(gp))
                vals[b] = agg(gp, idx) - agg(cp, idx)
            ci[name][dom] = [float(np.percentile(vals, 2.5)),
                             float(np.percentile(vals, 97.5))]

    # ---- 内核单次延迟 (单层, B=1 与 B=8) ----
    kern = {}
    l0 = 0
    p2v = p2s.layer_views(l0)
    p4v = p4s[l0]
    O2 = p2v["gu_q"].shape[1]
    H = p2v["gu_q"].shape[2] * 4
    K = 8
    for B in (1, 8):
        payload = torch.randn(B, H + 2 * K, dtype=torch.float16)
        payload[:, H:H + K] = torch.arange(K, dtype=torch.float16)
        payload[:, H + K:] = 0.125
        pm = torch.full((E,), -1, dtype=torch.int32)
        ext.moe_ffn_mixed2_batch(p4v["gu_q"], p4v["gu_s"], p4v["dn_q"], p4v["dn_s"],
                                 p2v["gu_q"], p2v["gu_s"], p2v["dn_q"], p2v["dn_s"],
                                 pm, payload, None)
        ts = []
        for _ in range(20):
            t1 = time.perf_counter()
            ext.moe_ffn_mixed2_batch(p4v["gu_q"], p4v["gu_s"], p4v["dn_q"], p4v["dn_s"],
                                     p2v["gu_q"], p2v["gu_s"], p2v["dn_q"], p2v["dn_s"],
                                     pm, payload, None)
            ts.append(time.perf_counter() - t1)
        kern[f"int2_B{B}"] = {"layer_ms": round(float(np.median(ts)) * 1e3, 3),
                              "est_40layers_ms": round(float(np.median(ts)) * 1e3 * 40, 1)}
    print(f"[kern] {kern}", flush=True)

    # ---- 容器口径: 2bit 底座 + 25% 专家 4bit 整行补丁 ----
    N_WEIGHTS = (bitpack.N_LAYERS * bitpack.N_EXPERTS * 3 *
                 bitpack.HIDDEN * bitpack.MOE_INTER)
    n_high = int(P_HIGH * bitpack.N_LAYERS * bitpack.N_EXPERTS)
    base_gib = p2s.meta["total_bytes"] / 2 ** 30
    # (a) 保守口径 (与运行时双容器一致): 2bit 底座全量保留 + 补丁专家 4bit 整行另存
    row4_gib = n_high * 3 * bitpack.HIDDEN * bitpack.MOE_INTER * 4 / 8 / 2 ** 30
    row2_gib = n_high * 3 * bitpack.HIDDEN * bitpack.MOE_INTER * 2 / 8 / 2 ** 30
    # (b) 最小口径: 补丁专家丢弃 2bit 行, 只留 4bit 整行
    eff_keep_both = base_gib + row4_gib
    eff_replace = base_gib - row2_gib + row4_gib
    eff_bits = eff_keep_both * 8 * 2 ** 30 / N_WEIGHTS
    eff_bits_replace = eff_replace * 8 * 2 ** 30 / N_WEIGHTS

    # ---- 内核数值校验 (torch 反量化参考) ----
    try:
        ncheck = numeric_check(ext, p2s.layer_views(0), p4s[0], sym4=sym4)
    except Exception as ex:  # 校验失败不影响主表, 但必须记录
        ncheck = {"error": repr(ex)}
    print(f"[numeric] {ncheck}", flush=True)

    out = {"n_eval_seq": {d: len(evalsets[d]) for d in T.DOMAINS},
           "configs": [c[0] for c in configs], "nll": nll,
           "store": os.path.basename(store_dir), "sym4": sym4,
           "gain_vs_global": gains, "ci95": ci, "n_boot": N_BOOT,
           "timing": timing, "kernel": kern,
           "numeric_check": ncheck,
           "container": {"bits": 2, "file_gib": round(p2s.meta["total_bytes"] / 2 ** 30, 2),
                         "n_weights": int(N_WEIGHTS), "n_high": n_high,
                         "patch_row4_gib": round(row4_gib, 3),
                         "patch_row2_replaced_gib": round(row2_gib, 3),
                         "keep_both_gib": round(eff_keep_both, 2),
                         "keep_both_bits": round(eff_bits, 3),
                         "replace_gib": round(eff_replace, 2),
                         "replace_bits": round(eff_bits_replace, 3)},
           "elapsed_s": round(time.time() - t_all, 1)}
    p = os.path.join(T.RES_DIR, out_name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[a1] saved {p} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
