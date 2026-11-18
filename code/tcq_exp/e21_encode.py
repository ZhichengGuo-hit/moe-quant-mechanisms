"""E21: fit per-tensor non-uniform 4-level codebooks and re-encode the 2-bit base.

The container layout is untouched: one fp16 group scale d = amax/1.5 per 128
weights plus 32 bytes of 2-bit code indices. Only the meaning of the code index
changes, from the fixed levels {-1.5,-0.5,0.5,1.5}*d to cb[k]*d, where cb is a
4-entry table stored once per GGUF tensor (layer, projection).

Subcommands
  fit     fit cb[layer][gate|up|down] from activation-weighted expert samples
          and dump the per-expert channel importance used for encoding
  encode  rewrite only the q bytes of a store copy (d bytes stay untouched)
  verify  encode one layer with the uniform table and diff against the V0 store

Usage:
  python e21_encode.py fit
  python e21_encode.py encode --layers 0,8,20,32 --workers 8
  python e21_encode.py verify --layer 0
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tcmoe"))
sys.path.insert(0, os.path.join(ROOT, "llama.cpp", "gguf-py"))

import bitpack

MODEL_DIR = r"E:\CUDA11.8\model\Qwen3.6-35B-A3B"
IMATRIX = r"E:\CUDA11.8\model\e20\im_global.gguf"
SRC_STORE = os.path.join(HERE, "stores", "packed2_sym4")
DST_STORE = os.path.join(HERE, "stores", "packed2_v1")
CB_PATH = os.path.join(HERE, "stores", "packed2_v1", "e21_cb.npz")
PROJS = ("gate", "up", "down")
GROUP = 128


def load_imatrix(path=IMATRIX):
    """{tensor_key: [n_expert, n_col]} channel energy, normalized per expert."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        if not t.name.endswith(".in_sum2"):
            continue
        base = t.name[:-len(".in_sum2")]
        if not any(f"ffn_{p}_exps" in base for p in PROJS):
            continue
        arr = np.maximum(np.asarray(t.data, dtype=np.float64), 0.0)
        arr /= np.maximum(arr.mean(axis=1, keepdims=True), 1e-30)
        out[base] = arr.astype(np.float32)
    return out


def boundaries(cb):
    return (cb[1:] + cb[:-1]) * 0.5


def assign(z, cb):
    return (z[..., None] > boundaries(cb)).sum(axis=-1).astype(np.uint8)


def fit_cb(zs, ws, k=4, iters=40, tol=1e-6):
    """Weighted 1D Lloyd-Max on the pooled normalized values."""
    z = np.concatenate(zs)
    w = np.concatenate(ws)
    qs = (np.arange(k) + 0.5) / k
    lv = np.quantile(z, qs).astype(np.float64)
    lv.sort()
    for _ in range(iters):
        code = (z[:, None] > ((lv[1:] + lv[:-1]) * 0.5)).sum(axis=1)
        new = lv.copy()
        for c in range(k):
            m = code == c
            ww = w[m]
            if ww.sum() > 1e-12:
                new[c] = (ww * z[m]).sum() / ww.sum()
        new.sort()
        if np.abs(new - lv).max() < tol:
            lv = new
            break
        lv = new
    return lv


def w_of(gu_dn, proj):
    gu, dn = gu_dn
    return gu[:bitpack.MOE_INTER] if proj == "gate" else \
           gu[bitpack.MOE_INTER:] if proj == "up" else dn


def cmd_fit(args):
    from pack_tcmoe import ExpertReader
    t0 = time.time()
    im = load_imatrix(args.imatrix)
    rd = ExpertReader(args.model_dir)
    n_layer = bitpack.N_LAYERS
    cb = np.zeros((n_layer, len(PROJS), 4), dtype=np.float16)
    imp_dump = {}
    experts = [int(x) for x in args.fit_experts.split(",")]
    for l in range(n_layer):
        for pi, proj in enumerate(PROJS):
            key = f"blk.{l}.ffn_{proj}_exps.weight"
            imp_dump[f"{l}_{proj}"] = im[key]
            zs, ws = [], []
            for e in experts:
                gu = rd.expert(l, "gate_up_proj", e)
                dn = rd.expert(l, "down_proj", e)
                w = np.ascontiguousarray(w_of((gu, dn), proj), dtype=np.float32)
                # reuse the stored fp16 scale: the encoder can only change codes
                d = np.float16(np.abs(w.reshape(w.shape[0], -1, GROUP)).max(-1) / 1.5)
                d32 = np.maximum(np.repeat(d.astype(np.float32), GROUP, axis=1), 1e-12)
                z = (w / d32).astype(np.float64)
                a = np.tile(im[key][e].astype(np.float64), (w.shape[0], 1)) * d32 ** 2
                sel = np.arange(0, w.shape[0], args.row_stride)
                zs.append(z[sel].reshape(-1))
                ws.append(a[sel].reshape(-1))
            cb[l, pi] = fit_cb(zs, ws, iters=args.iters).astype(np.float16)
        print(f"[fit] l{l} " + " ".join(
            f"{p}={np.round(cb[l, i].astype(np.float32), 3).tolist()}"
            for i, p in enumerate(PROJS)) + f" ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.cb), exist_ok=True)
    np.savez(args.cb, cb=cb)
    np.savez(os.path.join(os.path.dirname(args.cb), "e21_imp.npz"), **imp_dump)
    print(f"[fit] saved {args.cb} and e21_imp.npz ({time.time()-t0:.0f}s)", flush=True)


def _encode_layer(job):
    layer, dst, src, cb_path, model_dir = job
    import bitpack as bp
    from pack_tcmoe import ExpertReader
    cb_all = np.load(cb_path)["cb"]
    meta, part_off = bp.layout(2)
    stride = meta["layer_stride"]
    reader = ExpertReader(model_dir)
    mm = np.memmap(dst, dtype=np.uint8, mode="r+", shape=(meta["total_bytes"],))
    n_exp = bp.N_EXPERTS
    gu_q_rows, gu_q_cols = 2 * bp.MOE_INTER, bp.HIDDEN // 4
    dn_q_rows, dn_q_cols = bp.HIDDEN, bp.MOE_INTER // 4
    gu_s_rows = 2 * bp.MOE_INTER
    dn_s_rows = bp.HIDDEN
    written = 0
    for e in range(n_exp):
        gu = np.ascontiguousarray(reader.expert(layer, "gate_up_proj", e), dtype=np.float32)
        dn = np.ascontiguousarray(reader.expert(layer, "down_proj", e), dtype=np.float32)
        gu_s_off = layer * stride + part_off["gu_s"] + e * gu_s_rows * (bp.HIDDEN // GROUP) * 2
        dn_s_off = layer * stride + part_off["dn_s"] + e * dn_s_rows * (bp.MOE_INTER // GROUP) * 2
        gu_d = np.frombuffer(mm[gu_s_off:gu_s_off + gu_s_rows * (bp.HIDDEN // GROUP) * 2],
                             dtype=np.float16).reshape(gu_s_rows, bp.HIDDEN // GROUP)
        dn_d = np.frombuffer(mm[dn_s_off:dn_s_off + dn_s_rows * (bp.MOE_INTER // GROUP) * 2],
                             dtype=np.float16).reshape(dn_s_rows, bp.MOE_INTER // GROUP)
        for proj, w, d in (("gate", gu[:bp.MOE_INTER], gu_d[:bp.MOE_INTER]),
                           ("up", gu[bp.MOE_INTER:], gu_d[bp.MOE_INTER:]),
                           ("down", dn, dn_d)):
            cbf = cb_all[layer, PROJS.index(proj)].astype(np.float32)
            d32 = np.maximum(np.repeat(d.astype(np.float32), GROUP, axis=1), 1e-12)
            q = assign(w / d32, cbf)
            packed = bp.pack2(q)
            if proj in ("gate", "up"):
                row0 = 0 if proj == "gate" else bp.MOE_INTER
                off = layer * stride + part_off["gu_q"] + e * gu_q_rows * gu_q_cols + row0 * gu_q_cols
                mm[off:off + q.shape[0] * gu_q_cols] = packed.reshape(-1)
                written += packed.size
            else:
                off = layer * stride + part_off["dn_q"] + e * dn_q_rows * dn_q_cols
                mm[off:off + q.shape[0] * dn_q_cols] = packed.reshape(-1)
                written += packed.size
    mm.flush()
    return layer, written


def cmd_encode(args):
    src, dst = args.src, args.dst
    dst_bin = os.path.join(dst, "experts.bin")
    if not os.path.exists(dst_bin):
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        shutil.copyfile(os.path.join(src, "experts.bin"), dst_bin)
        idx = os.path.join(src, "index.json")
        if os.path.exists(idx):
            shutil.copyfile(idx, os.path.join(dst, "index.json"))
        print(f"[encode] copied store in {time.time()-t0:.0f}s", flush=True)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(bitpack.N_LAYERS))
    jobs = [(l, dst_bin, src, os.path.join(dst, os.path.basename(args.cb)),
             args.model_dir) for l in layers]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for layer, n in ex.map(_encode_layer, jobs):
            print(f"[encode] l{layer} {n/2**20:.1f} MiB q ({time.time()-t0:.0f}s)", flush=True)
    print(f"[encode] done {len(layers)} layers ({time.time()-t0:.0f}s)", flush=True)


def cmd_verify(args):
    """Uniform cb must reproduce the V0 q bytes (bit-exact sanity check)."""
    from pack_tcmoe import ExpertReader
    meta, part_off = bitpack.layout(2)
    stride = meta["layer_stride"]
    src = np.memmap(os.path.join(args.src, "experts.bin"), dtype=np.uint8, mode="r",
                    shape=(meta["total_bytes"],))
    reader = ExpertReader(args.model_dir)
    cb_uniform = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    layer = args.layer
    n_diff = n_tot = 0
    for e in range(args.experts):
        gu = np.ascontiguousarray(reader.expert(layer, "gate_up_proj", e), dtype=np.float32)
        dn = np.ascontiguousarray(reader.expert(layer, "down_proj", e), dtype=np.float32)
        for proj, w in (("gate", gu[:bitpack.MOE_INTER]), ("up", gu[bitpack.MOE_INTER:]),
                        ("down", dn)):
            part = "gu" if proj != "down" else "dn"
            rows = 2 * bitpack.MOE_INTER if part == "gu" else bitpack.HIDDEN
            qcols = bitpack.HIDDEN // 4 if part == "gu" else bitpack.MOE_INTER // 4
            scols = bitpack.HIDDEN // GROUP if part == "gu" else bitpack.MOE_INTER // GROUP
            s_off = layer * stride + part_off[part + "_s"] + e * rows * scols * 2
            d = np.frombuffer(src[s_off:s_off + rows * scols * 2],
                              dtype=np.float16).reshape(rows, scols)
            row0 = bitpack.MOE_INTER if proj == "up" else 0
            d = d[row0:row0 + w.shape[0]]
            d32 = np.maximum(np.repeat(d.astype(np.float32), GROUP, axis=1), 1e-12)
            got = bitpack.pack2(assign(w / d32, cb_uniform))
            off = layer * stride + part_off[part + "_q"] + e * rows * qcols
            ref = np.frombuffer(src[off:off + rows * qcols], dtype=np.uint8).reshape(rows, qcols)
            ref = ref[row0:row0 + w.shape[0]]
            same = (got == ref)
            # all-zero groups carry d = 0, so their code index is arbitrary;
            # near-zero weights sit exactly on the V0 code boundary (round tie)
            live = np.repeat(d.astype(np.float32) > 0, GROUP, axis=1).reshape(w.shape[0], -1, 4).all(-1)
            live &= (np.abs(w / d32) > 1e-6).reshape(w.shape[0], -1, 4).all(-1)
            n_diff += int((~same & live).sum())
            n_tot += int(live.sum())
    print(f"[verify] l{layer} experts<{args.experts}: {n_diff}/{n_tot} bytes differ "
          f"({100*n_diff/max(n_tot,1):.4f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--imatrix", default=IMATRIX)
    f.add_argument("--model-dir", default=MODEL_DIR)
    f.add_argument("--fit-experts", default="0,85,170,255")
    f.add_argument("--row-stride", type=int, default=4)
    f.add_argument("--iters", type=int, default=40)
    f.add_argument("--cb", default=CB_PATH)
    f.set_defaults(func=cmd_fit)
    e = sub.add_parser("encode")
    e.add_argument("--src", default=SRC_STORE)
    e.add_argument("--dst", default=DST_STORE)
    e.add_argument("--cb", default=CB_PATH)
    e.add_argument("--imp", default=os.path.join(os.path.dirname(CB_PATH), "e21_imp.npz"))
    e.add_argument("--layers", default="")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--model-dir", default=MODEL_DIR)
    e.set_defaults(func=cmd_encode)
    v = sub.add_parser("verify")
    v.add_argument("--src", default=SRC_STORE)
    v.add_argument("--model-dir", default=MODEL_DIR)
    v.add_argument("--layer", type=int, default=0)
    v.add_argument("--experts", type=int, default=8)
    v.set_defaults(func=cmd_verify)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
