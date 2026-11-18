"""D1: 补丁残差谱测量 —— INT4 补丁能否被低秩 LoRA 近似。

对每个"补丁专家"(E3 的 25% 高精度专家, 按域敏感度 s(l,e,D) 选) 计算:
  d = W_int4 - W_int3           (补丁真正要写进去的增量)
  SVD(d) 的奇异值谱; rank-r 能量恢复 = sum_{i<=r} sigma_i^2 / sum_i sigma_i^2
  rank-r 截断误差 ||d - d_r||_F = sqrt(sum_{i>r} sigma_i^2)
参考量:
  e4 = ||W_int4 - W_bf16||_F    (INT4 自身的量化误差)
  e3 = ||W_int3 - W_bf16||_F
判据(用户口径): r<=16 恢复 >=80% 能量 -> LoRA 路线成立。
附加判据(工程上更关键): r<=16 的 LoRA 截断误差 << INT4 自身量化误差 -> 用 LoRA 近似补丁
不会引入额外损伤。

实现: 每层逐个专家反量化 (packed3 密排 + int4 nibble + bf16 safetensors), GPU 做 svdvals。
输出: results/d1_residual_spectrum.json, results/d1_spectrum.npz, results/d1_curve.png
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

P_HIGH = 0.25
RS = (4, 8, 16, 32, 64)
GROUP = 128
P3_DIR = os.path.join(HERE, "stores", "packed3")
SEED = 11


class Packed3Store:
    def __init__(self, path=P3_DIR):
        meta, part_off = bitpack.layout(3)
        self.meta, self.part_off = meta, part_off
        self.stride = meta["layer_stride"]
        self.parts = meta["parts"]
        self._mm = np.memmap(os.path.join(path, "experts.bin"), dtype=np.uint8,
                             mode="r", shape=(meta["total_bytes"],))
        self._keep = {}

    def part(self, layer, name):
        key = (layer, name)
        if key in self._keep:
            return self._keep[key]
        shape = self.parts[name]
        off = layer * self.stride + self.part_off[name]
        n = int(np.prod(shape))
        nbytes = n * (2 if name.endswith("_s") else 1)
        arr = self._mm[off:off + nbytes].view(np.float16 if name.endswith("_s") else np.uint8)
        t = torch.from_numpy(arr.reshape(shape))
        self._keep[key] = t
        return t


def deq3(p3, layer, e, part):
    q = bitpack.unpack3(p3.part(layer, part)[e].numpy())
    s = p3.part(layer, part[:-1] + "s")[e].float()
    return bitpack.dequant_group(torch.from_numpy(q), s, 3)


def deq4(view, e, part, N, group=GROUP):
    q = view[part][e]
    lo = q & 15
    hi = (q >> 4) & 15
    v = torch.stack([lo, hi], dim=-1).reshape(q.shape[0], N).float() - 8.0
    s = view[part[:-1] + "s"][e].float()
    return (v.reshape(q.shape[0], N // group, group) * s.unsqueeze(-1)).reshape(q.shape[0], N)


def select_per_layer(score, frac=P_HIGH):
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        high[l, np.argsort(-score[l])[:k]] = True
    return high


def energy_frac(sv, rs=RS):
    e = sv ** 2
    tot = e.sum() + 1e-30
    return [float(e[:r].sum() / tot) for r in rs]


def trunc_err(sv, rs=RS):
    e = sv ** 2
    return [float(np.sqrt(max(e[r:].sum(), 0.0))) for r in rs]


def main():
    t_all = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s_mat = np.array(sm["s"])
    doms = sm["domains"]

    eng = T.build_engine()
    n_layers = len(eng["model"].model.language_model.layers)
    E = eng["store"].num_experts
    q4_store = eng["store"]
    p3 = Packed3Store()
    wmap = T.load_weight_map()
    shard = {}

    def get(key):
        sh = wmap[key]
        if sh not in shard:
            shard[sh] = safetensors.safe_open(os.path.join(T.MOE_MODEL_PATH, sh), framework="pt")
        return shard[sh].get_tensor(key)

    masks = {d: select_per_layer(s_mat[:, :, i]) for i, d in enumerate(doms)}
    masks["global"] = select_per_layer(s_mat.sum(-1))
    union = np.zeros((n_layers, E), dtype=bool)
    for m in masks.values():
        union |= m
    print(f"[d1] patch experts per-domain={int(masks[doms[0]].sum())}, "
          f"union={int(union.sum())}, dev={dev}", flush=True)

    rec, ter, e3n, e4n, dnorm = {}, {}, {}, {}, {}
    spec_keep = {}
    for proj in ("gu", "dn"):
        for key in RS:
            rec.setdefault((proj, key), {})
            ter.setdefault((proj, key), {})
            e3n.setdefault((proj, key), {})
            e4n.setdefault((proj, key), {})
            dnorm.setdefault((proj, key), {})
            for mname in masks:
                rec[(proj, key)][mname] = []
                ter[(proj, key)][mname] = []
                e3n[(proj, key)][mname] = []
                e4n[(proj, key)][mname] = []
                dnorm[(proj, key)][mname] = []

    t_svd = 0.0
    for l in range(n_layers):
        t0 = time.time()
        gu_bf = get(f"model.language_model.layers.{l}.mlp.experts.gate_up_proj")
        dn_bf = get(f"model.language_model.layers.{l}.mlp.experts.down_proj")
        v4 = q4_store.layer_views(l)
        todo = [e for e in range(E) if union[l, e]]
        for e in todo:
            for proj, N in (("gu", bitpack.HIDDEN), ("dn", bitpack.MOE_INTER)):
                part = f"{proj}_q"
                w3 = deq3(p3, l, e, part).float()
                w4 = deq4(v4, e, part, N).float()
                wb = (gu_bf[e] if proj == "gu" else dn_bf[e]).float()
                d = (w4 - w3)
                ts = time.time()
                sv = torch.linalg.svdvals(d.to(dev)).cpu().numpy()
                t_svd += time.time() - ts
                rf = energy_frac(sv)
                te = trunc_err(sv)
                a3 = float((w3 - wb).norm())
                a4 = float((w4 - wb).norm())
                dn_ = float(d.norm())
                for mname, m in masks.items():
                    if not m[l, e]:
                        continue
                    for i, key in enumerate(RS):
                        rec[(proj, key)][mname].append(rf[i])
                        ter[(proj, key)][mname].append(te[i])
                        e3n[(proj, key)][mname].append(a3)
                        e4n[(proj, key)][mname].append(a4)
                        dnorm[(proj, key)][mname].append(dn_)
                if e < 8:
                    spec_keep[f"{proj}_l{l}_e{e}"] = sv[:128].astype(np.float32)
        if l % 5 == 0 or l == n_layers - 1:
            print(f"[d1] layer {l}: {len(todo)} experts, {time.time()-t0:.1f}s", flush=True)

    def agg(curvs, ref_curves=None):
        out = {}
        for proj in ("gu", "dn"):
            out[proj] = {}
            for key in RS:
                out[proj][str(key)] = {}
                for mname in masks:
                    v = curvs[(proj, key)][mname]
                    if not v:
                        continue
                    row = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                           "p10": float(np.percentile(v, 10)),
                           "p90": float(np.percentile(v, 90)), "n": len(v)}
                    if ref_curves is not None:
                        rv = ref_curves[(proj, key)][mname]
                        row["ratio_to_int4_err"] = float(np.median(
                            np.asarray(v) / np.maximum(np.asarray(rv), 1e-12)))
                    out[proj][str(key)][mname] = row
        return out

    out = {"n_patch_per_domain": {k: int(v.sum()) for k, v in masks.items()},
           "r": list(RS), "energy_recovery": agg(rec),
           "trunc_error": agg(ter, e4n),
           "int4_err_norm_median": {proj: {m: float(np.median(e4n[(proj, 4)][m]))
                                           for m in masks} for proj in ("gu", "dn")},
           "int3_err_norm_median": {proj: {m: float(np.median(e3n[(proj, 4)][m]))
                                           for m in masks} for proj in ("gu", "dn")},
           "delta_norm_median": {proj: {m: float(np.median(dnorm[(proj, 4)][m]))
                                        for m in masks} for proj in ("gu", "dn")}}

    # 判据
    guard = {}
    for mname in masks:
        for proj in ("gu", "dn"):
            for i, key in enumerate(RS):
                if key > 16:
                    continue
                med = out["energy_recovery"][proj][str(key)][mname]["median"]
                p10 = out["energy_recovery"][proj][str(key)][mname]["p10"]
                frac80 = float(np.mean(np.asarray(rec[(proj, key)][mname]) >= 0.8)) \
                    if rec[(proj, key)][mname] else 0.0
                guard.setdefault(mname, {}).setdefault(proj, {})[str(key)] = {
                    "median_recovery": med, "p10_recovery": p10,
                    "frac_experts_ge_80pct": frac80}
    out["verdict"] = guard
    out["r16_lora_ok"] = bool(all(guard[m][p][str(16)]["median_recovery"] >= 0.8
                                  for m in guard for p in guard[m]))

    np.savez_compressed(os.path.join(T.RES_DIR, "d1_spectrum.npz"), **spec_keep)
    p = os.path.join(T.RES_DIR, "d1_residual_spectrum.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, proj in zip(axs, ("gu", "dn")):
            for mname in masks:
                ys = [out["energy_recovery"][proj][str(r)][mname]["median"] for r in RS]
                ax.plot(RS, ys, marker="o", label=mname)
            ax.axhline(0.8, ls="--", c="gray", lw=1)
            ax.set_xscale("log", base=2); ax.set_xlabel("rank r")
            ax.set_ylabel("energy recovery (median)")
            ax.set_title(f"{proj}_proj"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(T.RES_DIR, "d1_curve.png"), dpi=130)
        print("[d1] saved results/d1_curve.png", flush=True)
    except Exception as ex:
        print(f"[d1] plot skipped: {ex!r}", flush=True)

    print(f"[d1] r16_lora_ok={out['r16_lora_ok']} svd={t_svd:.0f}s "
          f"total={time.time()-t_all:.0f}s", flush=True)
    for mname in masks:
        for proj in ("gu", "dn"):
            print(f"  [{mname}/{proj}] " + " ".join(
                f"r{r}:{out['energy_recovery'][proj][str(r)][mname]['median']:.2f}"
                for r in RS), flush=True)


if __name__ == "__main__":
    main()
