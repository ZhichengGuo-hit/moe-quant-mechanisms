"""E21: eval64 NLL with the non-uniform 2-bit codebook (type-45 precursor).

Runs the same E17 configurations as the sym4 baseline (base2 / global /
task_code) but the 2-bit base is decoded through the per-tensor 4-entry
codebook instead of the fixed uniform levels. The kernel and the container
layout are unchanged apart from the extra cb argument.

Usage:
  python e21_nll.py --tag v1 --limit 4          # smoke
  python e21_nll.py --tag v1                    # full eval64, 3 configs
  python e21_nll.py --tag v0 --store packed2_sym4 --uniform
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
import a1_two_bit_runtime as A1
from moe_lib import Int4Store

P_HIGH = 0.25
N_BOOT = 1000
SEED = 83
UNIFORM = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)


class LayerMixed2ExtNL:
    def __init__(self, ext, p2, p4, patch_map, gu_cb, dn_cb):
        self.ext, self.p2, self.p4, self.pmap = ext, p2, p4, patch_map
        self.gu_cb, self.dn_cb = gu_cb, dn_cb

    def moe_ffn_int4_batch(self, gu_q, gu_s, dn_q, dn_s, payload, out=None):
        return self.ext.moe_ffn_mixed2_batch(
            self.p4["gu_q"], self.p4["gu_s"], self.p4["dn_q"], self.p4["dn_s"],
            self.p2["gu_q"], self.p2["gu_s"], self.p2["dn_q"], self.p2["dn_s"],
            self.pmap, payload, out, self.gu_cb, self.dn_cb)

    def moe_ffn_int4(self, gu_q, gu_s, dn_q, dn_s, payload, out=None):
        raise NotImplementedError("decode path is prefill-only for E21")


def agg(per_seq, idx=None):
    if idx is None:
        idx = range(len(per_seq))
    return sum(per_seq[i][0] for i in idx) / max(sum(per_seq[i][1] for i in idx), 1)


def numeric_check(ext, p2v, p4v, gu_cb, dn_cb, n_patch=4, K=8):
    p2q, p2s, d2q, d2s = p2v["gu_q"], p2v["gu_s"], p2v["dn_q"], p2v["dn_s"]
    H = p2q.shape[2] * 4
    O2 = p2q.shape[1]
    I = d2q.shape[2] * 4

    def deq(e, proj):
        q = (p2q, p2s) if proj == "gu" else (d2q, d2s)
        qq = torch.from_numpy(bitpack.unpack2(np.asarray(q[0][e]))).long()
        ss = q[1][e].float()
        rows = qq.shape[0]
        vals = torch.empty(qq.shape)
        if proj == "gu":
            cb8 = gu_cb.float()
            vals[:I] = cb8[0:4][qq[:I]]
            vals[I:] = cb8[4:8][qq[I:]]
        else:
            vals[:] = dn_cb.float()[qq]
        return vals * torch.repeat_interleave(ss, 128, dim=1)

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
                                   pm, payload, None, gu_cb, dn_cb)
    x = payload[0, :H].float()
    acc = torch.zeros(H)
    for k in range(K):
        e = int(ids[k])
        if e < n_patch:
            gw = A1.deq_nibble(p4v["gu_q"][e], p4v["gu_s"][e], H)
            dw = A1.deq_nibble(p4v["dn_q"][e], p4v["dn_s"][e], I)
        else:
            gw, dw = deq(e, "gu"), deq(e, "dn")
        g = gw @ x
        act = torch.nn.functional.silu(g[:I]) * g[I:O2]
        acc += float(payload[0, H + K + k]) * (dw @ act)
    got = out[0, :H]
    scale = float(acc.abs().max()) + 1e-9
    return {"max_abs_diff": float((acc - got).abs().max()),
            "rel_max_abs_diff": float((acc - got).abs().max()) / scale,
            "cosine": float(torch.nn.functional.cosine_similarity(acc, got, dim=0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="packed2_v1")
    ap.add_argument("--cb", default=os.path.join(HERE, "stores", "packed2_v1", "e21_cb.npz"))
    ap.add_argument("--uniform", action="store_true",
                    help="ignore the fitted table and use the V0 uniform levels")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", default="base2,global,task_code")
    args = ap.parse_args()

    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])
    di = {d: i for i, d in enumerate(sm["domains"])}

    t0 = time.time()
    ext = int2_ext.load_ext(threads=16)
    eng = T.build_engine()
    tok, model = eng["tok"], eng["model"]
    n_layers = len(model.model.language_model.layers)
    E = eng["store"].num_experts
    p2s = A1.Packed2Store(os.path.join(HERE, "stores", args.store))
    store4 = eng["store"]
    p4s = {l: store4.layer_views(l) for l in range(n_layers)}
    cb = np.load(args.cb)["cb"].astype(np.float32) if not args.uniform else None

    def cb_for(l):
        if cb is None:
            g = UNIFORM
            u = UNIFORM
            d = UNIFORM
        else:
            g, u, d = cb[l, 0], cb[l, 1], cb[l, 2]
        return (torch.from_numpy(np.concatenate([g, u]).astype(np.float32)),
                torch.from_numpy(d.astype(np.float32)))

    def install(mask):
        for l in range(n_layers):
            pm = np.full(E, -1, dtype=np.int32)
            if mask is not None:
                pm[mask[l]] = np.nonzero(mask[l])[0]
            gc, dc = cb_for(l)
            model.model.language_model.layers[l].mlp.direct_ext = LayerMixed2ExtNL(
                ext, p2s.layer_views(l), p4s[l], torch.from_numpy(pm), gc, dc)

    evalsets = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval64")
        if args.limit:
            rows = rows[:args.limit]
        evalsets[dom] = [r["text"] for r in rows]
    print(f"[data] " + " ".join(f"{d}:{len(evalsets[d])}" for d in T.DOMAINS), flush=True)

    all_cfg = {"base2": None,
               "global": A1.select_per_layer(s.sum(-1), P_HIGH),
               "task_math": A1.select_per_layer(s[:, :, di["math"]], P_HIGH),
               "task_code": A1.select_per_layer(s[:, :, di["code"]], P_HIGH)}
    configs = [(c, all_cfg[c]) for c in args.configs.split(",")]

    nll, per_seq, timing = {}, {}, {}
    for name, mask in configs:
        t1 = time.time()
        install(mask)
        per_seq[name], nll[name] = {}, {}
        ntok = 0
        for dom in T.DOMAINS:
            tot, n, per, ps = T.eval_nll(model, tok, evalsets[dom], max_len=64,
                                         batch_size=8, return_per_seq=True)
            per_seq[name][dom] = ps
            nll[name][dom] = per
            ntok += n
        dt = time.time() - t1
        timing[name] = {"eval_s": round(dt, 1), "tokens": int(ntok),
                        "tok_per_s": round(ntok / dt, 2)}
        print(f"[{name}] " + " ".join(f"{d}:{nll[name][d]:.4f}" for d in T.DOMAINS) +
              f" ({dt:.0f}s, {ntok/dt:.2f} tok/s)", flush=True)

    gains, ci = {}, {}
    base = "global" if "global" in nll else next(iter(nll))
    rng = np.random.default_rng(SEED)
    for name in nll:
        gains[name] = {d: nll[base][d] - nll[name][d] for d in T.DOMAINS}
        ci[name] = {}
        for dom in T.DOMAINS:
            gp, cp = per_seq[base][dom], per_seq[name][dom]
            vals = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, len(gp), len(gp))
                vals[b] = agg(gp, idx) - agg(cp, idx)
            ci[name][dom] = [float(np.percentile(vals, 2.5)),
                             float(np.percentile(vals, 97.5))]

    try:
        ncheck = numeric_check(ext, p2s.layer_views(0), p4s[0], *cb_for(0))
    except Exception as ex:
        ncheck = {"error": repr(ex)}
    print(f"[numeric] {ncheck}", flush=True)

    out = {"tag": args.tag, "store": args.store,
           "cb_mean": None if cb is None else cb.mean(axis=0).tolist(),
           "uniform": bool(args.uniform), "limit": args.limit,
           "n_eval_seq": {d: len(evalsets[d]) for d in T.DOMAINS},
           "configs": [c[0] for c in configs], "nll": nll,
           "mean_over_domains": {k: float(np.mean(list(v.values()))) for k, v in nll.items()},
           "gain_vs_global": gains, "ci95": ci, "n_boot": N_BOOT,
           "timing": timing, "numeric_check": ncheck,
           "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(T.RES_DIR, exist_ok=True)
    p = os.path.join(T.RES_DIR, f"e21_nll_{args.tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[e21] saved {p} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
