"""D1b: 激活加权残差谱 —— 权重空间 SVD 之后必须排除的"度量错误"。

D1 已证明 d = W_int4 - W_int3 在权重 Frobenius 度量下几乎全秩 (r=16 仅恢复 5%)。
但 LoRA 的目标不是重建 d, 而是在激活分布下重建 d 作用在 token 上的效果:
    目标 = min_{A,B} E_x || (d - B A) x ||^2
最优 rank-r 解是 d·Σ^{1/2} 的 top-r 奇异三元组 (Σ = 该域路由 token 的输入二阶矩)。
本实验用对角近似 Σ ≈ diag(rms_j^2) (AWQ 同款), 即对 d 按通道 rms 缩放后做 SVD,
得到"功能最优"的 rank-r 能量恢复率:
  gu: rms 来自该专家在本域被路由到的 token 输入
  dn: rms 来自同一批 token 的中间激活 silu(g)*u (bf16 权重算)
判据: r=16 的功能最优恢复 >=80% -> LoRA 路线仍然成立; 否则彻底否掉。
输出: results/d1b_weighted_spectrum.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
import safetensors

import bitpack
import tcq_lib as T
from d1_residual_spectrum import Packed3Store, deq3, deq4, select_per_layer

RS = (4, 8, 16, 32, 64)
P_HIGH = 0.25
N_SEQ = 24
MAX_LEN = 32


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
    q4 = eng["store"]
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

    caps = {}
    for dom in doms:
        texts = [r["text"] for r in T.load_domain(dom, "calib")[:N_SEQ]]
        t0 = time.time()
        caps[dom] = T.capture_layer_stats(eng, texts, max_len=MAX_LEN, batch_size=6)
        print(f"[capture] {dom} {len(texts)} seq {time.time()-t0:.0f}s", flush=True)

    rec, rec_flat = {}, {}
    for proj in ("gu", "dn"):
        for r in RS:
            rec.setdefault((proj, r), {})
            rec_flat.setdefault((proj, r), {})
            for m in masks:
                rec[(proj, r)][m] = []
                rec_flat[(proj, r)][m] = []

    for l in range(n_layers):
        t0 = time.time()
        gu_bf = get(f"model.language_model.layers.{l}.mlp.experts.gate_up_proj")
        dn_bf = get(f"model.language_model.layers.{l}.mlp.experts.down_proj")
        v4 = q4.layer_views(l)
        for dom in doms:
            x = caps[dom][l]["x"].float()
            topi = caps[dom][l]["topi"].long()
            topv = caps[dom][l]["topv"].float()
            for e in range(E):
                if not (masks[dom][l, e] or masks["global"][l, e]):
                    continue
                tok, slot = torch.nonzero(topi == e, as_tuple=True)
                if tok.numel() < 2:
                    continue
                xe = x[tok]
                w3g = deq3(p3, l, e, "gu_q").float()
                w4g = deq4(v4, e, "gu_q", bitpack.HIDDEN).float()
                dg = (w4g - w3g)
                psig = xe.pow(2).mean(0).sqrt()
                mg = (dg * psig.unsqueeze(0)).to(dev)
                svg = torch.linalg.svdvals(mg).cpu().numpy()
                sg = (svg ** 2).sum() + 1e-30
                fw_g = ((svg ** 2) / sg).cumsum()

                w3d = deq3(p3, l, e, "dn_q").float()
                w4d = deq4(v4, e, "dn_q", bitpack.MOE_INTER).float()
                dd = (w4d - w3d)
                gu = xe @ gu_bf[e].float().t()
                g_, u_ = gu.chunk(2, dim=-1)
                h = F.silu(g_) * u_
                psih = h.pow(2).mean(0).sqrt()
                md = (dd * psih.unsqueeze(0)).to(dev)
                svd_ = torch.linalg.svdvals(md).cpu().numpy()
                sd = (svd_ ** 2).sum() + 1e-30
                fw_d = ((svd_ ** 2) / sd).cumsum()

                # 未加权(权重 Frobenius)参照: 直接累计占比
                svg0 = np.sort(svg)[::-1]
                f0_g = (svg0 ** 2).cumsum() / (svg0 ** 2).sum()
                svd0 = np.sort(svd_)[::-1]
                f0_d = (svd0 ** 2).cumsum() / (svd0 ** 2).sum()

                for mname, m in masks.items():
                    if not m[l, e]:
                        continue
                    for r in RS:
                        # 累计能量曲线取第 r 个元素 (rank-r 恢复率), 不能把前 r 项累加
                        rec[("gu", r)][mname].append(float(
                            fw_g[r - 1] if r <= len(fw_g) else 1.0))
                        rec[("dn", r)][mname].append(float(
                            fw_d[r - 1] if r <= len(fw_d) else 1.0))
                        rec_flat[("gu", r)][mname].append(float(
                            f0_g[r - 1] if r <= len(f0_g) else 1.0))
                        rec_flat[("dn", r)][mname].append(float(
                            f0_d[r - 1] if r <= len(f0_d) else 1.0))
        if l % 5 == 0 or l == n_layers - 1:
            print(f"[d1b] layer {l} {time.time()-t0:.1f}s", flush=True)

    out = {"r": list(RS), "n_seq": N_SEQ, "max_len": MAX_LEN,
           "energy_recovery_activation_weighted": {}, "energy_recovery_frobenius": {}}
    for proj in ("gu", "dn"):
        for key in ("energy_recovery_activation_weighted", "energy_recovery_frobenius"):
            out[key][proj] = {}
        out["energy_recovery_activation_weighted"][proj] = {}
    for proj in ("gu", "dn"):
        for r in RS:
            for mname in masks:
                v = rec[(proj, r)][mname]
                vf = rec_flat[(proj, r)][mname]
                if not v:
                    continue
                out["energy_recovery_activation_weighted"][proj].setdefault(str(r), {})[mname] = {
                    "median": float(np.median(v)), "mean": float(np.mean(v)),
                    "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                    "frac_ge_80pct": float(np.mean(np.asarray(v) >= 0.8)), "n": len(v)}
                out["energy_recovery_frobenius"][proj].setdefault(str(r), {})[mname] = {
                    "median": float(np.median(vf)), "frac_ge_80pct": float(np.mean(np.asarray(vf) >= 0.8))}

    m16 = [out["energy_recovery_activation_weighted"][p]["16"][m]["median"]
           for p in ("gu", "dn") for m in masks
           if "16" in out["energy_recovery_activation_weighted"][p]]
    out["r16_functional_lora_ok"] = bool(all(x >= 0.8 for x in m16))
    out["r16_min_median"] = float(min(m16)) if m16 else 0.0
    out["r32_min_median"] = float(min(
        out["energy_recovery_activation_weighted"][p]["32"][m]["median"]
        for p in ("gu", "dn") for m in masks)) if m16 else 0.0

    p_out = os.path.join(T.RES_DIR, "d1b_weighted_spectrum.json")
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[d1b] r16_ok={out['r16_functional_lora_ok']} min_median={out['r16_min_median']:.3f} "
          f"r32_min={out['r32_min_median']:.3f} total={time.time()-t_all:.0f}s", flush=True)
    for mname in masks:
        print(f"  [{mname}/gu] " + " ".join(
            f"r{r}:{out['energy_recovery_activation_weighted']['gu'][str(r)][mname]['median']:.2f}"
            for r in RS), flush=True)
        print(f"  [{mname}/dn] " + " ".join(
            f"r{r}:{out['energy_recovery_activation_weighted']['dn'][str(r)][mname]['median']:.2f}"
            for r in RS), flush=True)


if __name__ == "__main__":
    main()
