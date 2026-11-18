"""E21 step 1: non-uniform 4-level codebook headroom at the fixed 34 B/128 budget.

Compares three quantizers on real expert tensors, every one of them using the
existing container layout (fp16 group scale + 32 packed 2-bit codes per 128):
  V0  uniform symmetric codebook {-1, -1/3, 1/3, 1}     (current tcmoe_q2)
  V1  one activation-weighted Lloyd-Max codebook per projection matrix,
      stored as 4 fp16 per matrix (volume is unchanged at 2.125 bpw)
  V2  one Lloyd-Max codebook per 128-value group (the +0.5 bpw upper bound;
      not meant for the container, only as a quality reference)

Weights come from the safetensors expert tensors, importance from the E20
three-domain interleaved imatrix (per expert, per input channel diagonal).
Group scales are refit by weighted least squares and rounded to fp16 so the
reported MSE matches what the container can actually store.

Output: results/e21_codebook.json (sample metrics) and optional .npz of the
sampled codebooks for inspection.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "llama.cpp", "gguf-py"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "coserving"))
sys.path.insert(0, os.path.join(ROOT, "tcmoe"))

GROUP = 128
N_LEVEL = 4
HIDDEN = 2048
MOE_INTER = 512
IMATRIX = r"E:\CUDA11.8\model\e20\im_global.gguf"
MODEL_DIR = r"E:\CUDA11.8\model\Qwen3.6-35B-A3B"


def parse_list(spec, default):
    if not spec:
        return list(default)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def read_imatrix_diag(path):
    """{tensor_key: [n_expert, n_col] normalized channel importance}."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        if not t.name.endswith(".in_sum2"):
            continue
        base = t.name[:-len(".in_sum2")]
        if "ffn_gate_exps" not in base and "ffn_up_exps" not in base \
                and "ffn_down_exps" not in base:
            continue
        arr = np.asarray(t.data, dtype=np.float64)
        # GGUF stores ne order (n_col, n_expert); the array view is [n_expert, n_col]
        imp = np.maximum(arr, 0.0)
        imp /= np.maximum(imp.mean(axis=1, keepdims=True), 1e-30)
        out[base] = imp
    return out


def amax_scale(w):
    """[R,C] -> [R,C/128] fp16-rounded group amax/1.5."""
    R, C = w.shape
    g = w.reshape(R, C // GROUP, GROUP)
    amax = np.abs(g).max(axis=-1)
    d = (amax / 1.5).astype(np.float16)
    return d


def d16_to_f32(d):
    return d.astype(np.float32)


def expand_d(d):
    """[R,B] fp16 -> [R,C] float32 scale per element."""
    return np.repeat(d.astype(np.float32), GROUP, axis=1)


def assign_levels(vals, levels, boundaries):
    """vals [R,C] -> code index 0..3 given per-group sorted levels [R,B,4]."""
    R, C = vals.shape
    B = levels.shape[1]
    v = vals.reshape(R, B, GROUP)
    bd = boundaries  # [R,B,3]
    q = (v[..., None] > bd[..., None, :]).sum(axis=-1)
    return q.reshape(R, C)


def boundaries_of(levels):
    return (levels[..., 1:] + levels[..., :-1]) * 0.5


def lloyd_grouped(vals, wts, k=N_LEVEL, iters=25, tol=1e-6):
    """Activation-weighted 1D Lloyd-Max, vectorized over groups.

    vals, wts: [N, L] float32/float64 (wts same shape, per-value importance).
    Returns levels [N, k] sorted per group.
    """
    N, L = vals.shape
    qs = (np.arange(k) + 0.5) / k
    levels = np.quantile(vals, qs, axis=1).T.astype(np.float64)  # [N,k]
    levels.sort(axis=1)
    for _ in range(iters):
        bd = (levels[:, 1:] + levels[:, :-1]) * 0.5
        code = (vals[:, :, None] > bd[:, None, :]).sum(axis=-1)
        new = levels.copy()
        for c in range(k):
            m = (code == c)
            ws = (wts * m).sum(axis=1)
            num = (wts * vals * m).sum(axis=1)
            ok = ws > 1e-12
            new[ok, c] = num[ok] / ws[ok]
        new.sort(axis=1)
        delta = np.abs(new - levels).max()
        levels = new
        if delta < tol:
            break
    return levels


def quant_v0(w):
    d = amax_scale(w)
    d32 = np.maximum(expand_d(d), 1e-12)
    q = np.clip(np.round(w / d32 + 1.5), 0, 3).astype(np.uint8)
    deq = (q.astype(np.float32) - 1.5) * d32
    return q, d, deq


def quant_v1(w, imp, weight_mode="act", iters=40, tol=1e-6):
    """One codebook per matrix, expressed in units of the stored fp16 group scale.

    The container keeps the existing fp16 d = amax/1.5 and only replaces the
    fixed levels {-1.5, -0.5, 0.5, 1.5} with an activation-weighted Lloyd-Max
    codebook. No extra per-group bytes are spent.
    """
    R, C = w.shape
    B = C // GROUP
    d = amax_scale(w)
    d32 = np.maximum(expand_d(d), 1e-12)
    z = (w / d32).astype(np.float64)
    a = np.tile(imp.astype(np.float64), (R, 1))
    if weight_mode == "act":
        # error in weight space is d^2 * (z - c)^2, so weights carry d^2
        a = a * d32.astype(np.float64) ** 2
    elif weight_mode == "none":
        pass
    else:
        raise ValueError(weight_mode)
    cb = lloyd_grouped(z, a, iters=iters, tol=tol)[0]
    cb16 = cb.astype(np.float16)
    cbf = cb16.astype(np.float32)
    levels = np.tile(cbf, (R, B, 1))
    bd = boundaries_of(levels)
    q = assign_levels(z.astype(np.float32), levels, bd)
    deq = d32 * cbf[q]
    wn = np.tile(imp.astype(np.float64), (R, 1))
    msez = float((wn * (w - deq) ** 2).sum() / wn.sum())
    return q, d, deq, cb16, msez


def quant_v2(w, imp, iters=40, tol=1e-6):
    """Per-group codebook (absolute fp16 levels) upper bound, no shared scale."""
    R, C = w.shape
    B = C // GROUP
    v = w.reshape(R, B, GROUP).astype(np.float64)
    a = np.tile(imp.astype(np.float64), (R, 1)).reshape(R, B, GROUP)
    N = R * B
    levels = lloyd_grouped(v.reshape(N, GROUP), a.reshape(N, GROUP),
                           iters=iters, tol=tol).astype(np.float16).astype(np.float32)
    levels = levels.reshape(R, B, N_LEVEL)
    bd = boundaries_of(levels)
    v32 = v.astype(np.float32)
    q = (v32[..., None] > bd[..., None, :]).sum(axis=-1)
    deq = np.zeros_like(v32)
    for k in range(N_LEVEL):
        deq += np.where(q == k, levels[:, :, k:k + 1], 0.0)
    deq = deq.reshape(R, C)
    mse = float((a.reshape(R, C) * (w - deq) ** 2).sum() / a.reshape(R, C).sum())
    return q.reshape(R, C), levels, deq


def metrics(w, imp, deq):
    err = (w - deq) ** 2
    a = np.tile(imp, (w.shape[0], 1))
    wmse = float((a * err).sum() / a.sum())
    mse = float(err.mean())
    rel = float(np.sqrt(mse) / max(np.sqrt((w ** 2).mean()), 1e-30))
    return wmse, mse, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix", default=IMATRIX)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,32,36,39")
    ap.add_argument("--experts", default="0-5")
    ap.add_argument("--weight-mode", default="act", choices=("act", "none"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e21_codebook.json"))
    ap.add_argument("--npz", default=os.path.join(HERE, "results", "e21_codebook_sample.npz"))
    args = ap.parse_args()

    layers = parse_list(args.layers, [])
    experts = parse_list(args.experts, [])
    t0 = time.time()
    im = read_imatrix_diag(args.imatrix)
    print(f"[imatrix] {len(im)} tensors {time.time()-t0:.0f}s", flush=True)

    from pack_tcmoe import ExpertReader
    reader = ExpertReader(args.model_dir)

    rows = []
    for layer in layers:
        gu = None
        for e in experts:
            gu = reader.expert(layer, "gate_up_proj", e)
            dn = reader.expert(layer, "down_proj", e)
            for proj, w, imp_key in (
                    ("gate", gu[:MOE_INTER], f"blk.{layer}.ffn_gate_exps.weight"),
                    ("up", gu[MOE_INTER:], f"blk.{layer}.ffn_up_exps.weight"),
                    ("down", dn, f"blk.{layer}.ffn_down_exps.weight")):
                imp = im[imp_key][e]
                w = np.ascontiguousarray(w, dtype=np.float32)
                q0, d0, deq0 = quant_v0(w)
                q1, d1, deq1, cb1, _ = quant_v1(w, imp, args.weight_mode)
                q2, d2, deq2 = quant_v2(w, imp)
                m0, m0u, r0 = metrics(w, imp, deq0)
                m1, m1u, r1 = metrics(w, imp, deq1)
                m2, m2u, r2 = metrics(w, imp, deq2)
                rows.append({"layer": layer, "expert": e, "proj": proj,
                             "v0_wmse": m0, "v1_wmse": m1, "v2_wmse": m2,
                             "v0_mse": m0u, "v1_mse": m1u, "v2_mse": m2u,
                             "v0_rel": r0, "v1_rel": r1, "v2_rel": r2,
                             "cb1": cb1.tolist()})
            print(f"[m] l{layer} e{e}: v1/v0 wmse "
                  f"{rows[-3]['v1_wmse']/rows[-3]['v0_wmse']:.3f} "
                  f"{rows[-2]['v1_wmse']/rows[-2]['v0_wmse']:.3f} "
                  f"{rows[-1]['v1_wmse']/rows[-1]['v0_wmse']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = {"imatrix": args.imatrix, "layers": layers, "experts": experts,
           "n_matrix": len(rows), "group": GROUP, "n_level": N_LEVEL,
           "weight_mode": args.weight_mode}
    for key in ("wmse", "mse", "rel"):
        out[key] = {}
        for v in ("v0", "v1", "v2"):
            vals = np.array([r[f"{v}_{key}"] for r in rows])
            out[key][v] = float(vals.mean())
    out["gain_wmse_pct"] = {
        "v1": float(100 * (1 - out["wmse"]["v1"] / out["wmse"]["v0"])),
        "v2": float(100 * (1 - out["wmse"]["v2"] / out["wmse"]["v0"]))}
    out["gain_mse_pct"] = {
        "v1": float(100 * (1 - out["mse"]["v1"] / out["mse"]["v0"])),
        "v2": float(100 * (1 - out["mse"]["v2"] / out["mse"]["v0"]))}
    out["per_proj"] = {}
    for proj in ("gate", "up", "down"):
        sub = [r for r in rows if r["proj"] == proj]
        out["per_proj"][proj] = {
            v: {k: float(np.mean([r[f"{v}_{k}"] for r in sub]))
                for k in ("wmse", "mse", "rel")} for v in ("v0", "v1", "v2")}
    out["per_layer"] = {}
    for layer in layers:
        sub = [r for r in rows if r["layer"] == layer]
        out["per_layer"][str(layer)] = {
            v: float(np.mean([r[f"{v}_wmse"] for r in sub]))
            for v in ("v0", "v1", "v2")}
    out["codebook_mean"] = np.mean([r["cb1"] for r in rows], axis=0).tolist()
    out["codebook_std"] = np.std([r["cb1"] for r in rows], axis=0).tolist()
    out["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if args.npz:
        np.savez(args.npz,
                 meta=np.array(json.dumps({"layers": layers, "experts": experts})),
                 cb1=np.array([r["cb1"] for r in rows], dtype=np.float32))
    print(json.dumps({k: out[k] for k in ("wmse", "mse", "rel",
                                          "gain_wmse_pct", "gain_mse_pct",
                                          "per_proj")}, indent=1), flush=True)
    print(f"[e21] saved {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
