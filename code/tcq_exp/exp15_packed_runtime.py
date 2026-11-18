"""E15: 3bit 真实容器运行时接入 —— H4 在真实 3.38bit 上跑通。

现状遗留: 之前的 INT3 底座用 nibble 容器 (4bit 槽, 15.47GiB), 只是"名义 3.25bit"。
本实验把运行时直接接到 E12 产出的密排 3bit 容器 (11.72GiB), 高精度补丁从 INT4
nibble store 按专家读取, 计算路径 = 新增的 int3_ext 混合内核 (3bit 反量化 GEMV +
补丁行 INT4 GEMV), 不再经过 nibble 反量化。

配置 (eval64, 64 条/域):
  base3       全 3bit (无补丁, 3.0bit)
  global      3bit 底座 + 每层 64 INT4 (3.38bit)
  task_math / task_code / mixed_l50
对照:
  base3_nibble  同样的 INT3 量化权重, 走原 nibble int4_ext 路径 (eval 16 条)
                —— 验证两条运行时数值等价
输出: results/packed_runtime.json
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
import int3_ext
from moe_lib import Int4Store

P_HIGH = 0.25
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")
N_BOOT = 1000
SEED = 13
PACKED3_DIR = os.path.join(HERE, "stores", "packed3")


def select_per_layer(score, frac):
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        high[l, np.argsort(-score[l])[:k]] = True
    return high


class Packed3Store:
    """密排 3bit 容器 -> 每层 part 的 torch 视图 (零拷贝 mmap)"""

    def __init__(self, path=PACKED3_DIR):
        meta, part_off = bitpack.layout(3)
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


class LayerMixedExt:
    """把 per-layer 的 3bit 底座 + INT4 补丁视图包装成 direct_ext 接口"""

    def __init__(self, ext, p3, p4, patch_map):
        self.ext = ext
        self.p3, self.p4, self.pmap = p3, p4, patch_map

    def moe_ffn_int4_batch(self, gu_q, gu_s, dn_q, dn_s, payload, out=None):
        return self.ext.moe_ffn_mixed_batch(
            self.p4["gu_q"], self.p4["gu_s"], self.p4["dn_q"], self.p4["dn_s"],
            self.p3["gu_q"], self.p3["gu_s"], self.p3["dn_q"], self.p3["dn_s"],
            self.pmap, payload, out)

    def moe_ffn_int4(self, gu_q, gu_s, dn_q, dn_s, payload, out=None):
        return self.moe_ffn_int4_batch(gu_q, gu_s, dn_q, dn_s,
                                       payload.unsqueeze(0),
                                       None).squeeze(0)


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
    doms, di = sm["domains"], {d: i for i, d in enumerate(sm["domains"])}
    with open(os.path.join(T.RES_DIR, "packing.json"), encoding="utf-8") as f:
        pack = json.load(f)

    ext = int3_ext.load_ext(threads=16)
    eng = T.build_engine()
    tok = eng["tok"]
    model = eng["model"]
    n_layers = len(model.model.language_model.layers)
    E = eng["store"].num_experts

    p3s = Packed3Store()
    store4 = eng["store"]
    p4s = {l: store4.layer_views(l) for l in range(n_layers)}

    import psutil
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss / 2 ** 30

    def install(mask):
        """mask: None -> 全 3bit; bool[L,E] -> 补丁"""
        for l in range(n_layers):
            pm = np.full(E, -1, dtype=np.int32)
            if mask is not None:
                pm[mask[l]] = np.nonzero(mask[l])[0]
            model.model.language_model.layers[l].mlp.direct_ext = LayerMixedExt(
                ext, p3s.layer_views(l), p4s[l], torch.from_numpy(pm))

    evalsets = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval64")
        evalsets[dom] = [r["text"] for r in rows]
        print(f"[data] {dom}: {len(evalsets[dom])} seq", flush=True)

    configs = [("base3", None),
               ("global", select_per_layer(s.sum(-1), P_HIGH)),
               ("task_math", select_per_layer(s[:, :, di["math"]], P_HIGH)),
               ("task_code", select_per_layer(s[:, :, di["code"]], P_HIGH)),
               ("mixed_l50", select_per_layer(
                   0.5 * s[:, :, di["math"]] + 0.5 * s[:, :, di["code"]], P_HIGH))]

    nll, per_seq, timing = {}, {}, {}
    t_all = time.time()
    for name, mask in configs:
        t0 = time.time()
        install(mask)
        per_seq[name], nll[name] = {}, {}
        ntok = 0
        for dom in T.DOMAINS:
            tot, n, per, ps = T.eval_nll(model, tok, evalsets[dom],
                                         max_len=64, batch_size=8, return_per_seq=True)
            per_seq[name][dom] = ps
            nll[name][dom] = per
            ntok += n
        timing[name] = {"elapsed_s": round(time.time() - t0, 1),
                        "tok_per_s": round(ntok / max(time.time() - t0, 1e-9), 1)}
        print(f"[{name}] nll=" + " ".join(f"{d}:{nll[name][d]:.4f}" for d in T.DOMAINS) +
              f" ({timing[name]['elapsed_s']}s, {timing[name]['tok_per_s']} tok/s)", flush=True)

    rss1 = proc.memory_info().rss / 2 ** 30

    # ---- 噪声与 bootstrap CI (相对 global) ----
    split = {f"{n}/{d}": abs(agg(per_seq[n][d], list(range(0, 64, 2)))
                             - agg(per_seq[n][d], list(range(1, 64, 2))))
             for n in nll for d in T.DOMAINS}
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

    # ---- 数值等价对照: packed3 runtime vs nibble int3 runtime (eval 16) ----
    patcher = T.ExpertPatcher(store4)
    store3 = Int4Store(T.STORE_INT3, mmap_mode="r")
    v3 = {l: store3.layer_views(l) for l in range(n_layers)}
    t0 = time.time()
    for l in range(n_layers):
        for e in range(E):
            patcher.set_expert(l, e, {p: v3[l][p][e] for p in LAYER_PARTS})
    eval16 = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval")
        texts = [r["text"] for r in rows]
        tot, n, per = T.eval_nll(model, tok, texts, max_len=64, batch_size=8)
        eval16[dom] = per
    for l in range(n_layers):
        for e in range(E):
            patcher.restore_expert(l, e)
    print(f"[nibble-int3 eval16] " + " ".join(f"{d}:{eval16[d]:.4f}" for d in T.DOMAINS) +
          f" ({time.time()-t0:.0f}s)", flush=True)

    install(None)
    packed16 = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval")
        texts = [r["text"] for r in rows]
        tot, n, per = T.eval_nll(model, tok, texts, max_len=64, batch_size=8)
        packed16[dom] = per
    print(f"[packed3 eval16] " + " ".join(f"{d}:{packed16[d]:.4f}" for d in T.DOMAINS), flush=True)
    equiv = {d: packed16[d] - eval16[d] for d in T.DOMAINS}

    # ---- 容器体积口径 ----
    base_gib = pack["sizes"]["packed3"]["gib"]
    patch_gib = pack["patch"]["patch_gib"]
    eff_bits = (base_gib + patch_gib) * 8 * 2 ** 30 / (bitpack.N_LAYERS *
                                                       bitpack.N_EXPERTS *
                                                       (2 * bitpack.MOE_INTER * bitpack.HIDDEN +
                                                        bitpack.HIDDEN * bitpack.MOE_INTER))

    out = {"n_eval_seq": {d: len(evalsets[d]) for d in T.DOMAINS},
           "configs": [c[0] for c in configs],
           "nll": nll, "per_seq": per_seq, "gain_vs_global": gains, "ci95": ci,
           "split_half_abs_delta": split, "n_boot": N_BOOT,
           "timing": timing, "rss_gib": {"before": round(rss0, 2), "after": round(rss1, 2)},
           "equiv_check": {"eval16_nibble_int3": eval16, "eval16_packed3": packed16,
                           "delta": equiv},
           "container": {"base3_gib": base_gib, "patch_int4_gib": patch_gib,
                         "effective_gib": round(base_gib + patch_gib, 2),
                         "effective_bits": round(eff_bits, 3),
                         "n_high": int(P_HIGH * bitpack.N_LAYERS * bitpack.N_EXPERTS)},
           "elapsed_s": round(time.time() - t_all, 1)}

    p = os.path.join(T.RES_DIR, "packed_runtime.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[packed_runtime] saved {p} ({time.time()-t_all:.0f}s)", flush=True)
    print(f"[equiv] " + " ".join(f"{d}:{equiv[d]:+.4f}" for d in T.DOMAINS), flush=True)
    print(f"[rss] before={rss0:.2f}GiB after={rss1:.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
