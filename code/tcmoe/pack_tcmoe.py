#!/usr/bin/env python3
"""Pack a tcmoe GGUF: 2-bit base + per-domain 4-bit expert patch tables.

Format: tcmoe/spec/TCMOE-GGUF-PATCH-SPEC.md

Examples:
    # patch-only test file (layers 0-1, 3 domains)
    python pack_tcmoe.py --out test.gguf --layers 0-1 --domains math,code

    # full single-file model: copy a Q4_K_M GGUF, replace expert tensors with tcmoe_q2
    python pack_tcmoe.py --out model-tcmoe.gguf --skeleton Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
        --base tcmoe_q2 --p2-dir ../tcq_exp/stores/packed2_sym4
"""

import argparse
import ctypes
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(ROOT / "coserving"))
sys.path.insert(0, str(ROOT / "tcq_exp"))

from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType, GGUFValueType  # noqa: E402

import torch  # noqa: E402
from moe_lib import quantize_int4  # noqa: E402

PROJECTIONS = ("gate", "up", "down")

GGML_TYPE_Q4_K = 12
DEFAULT_GGML_DLL = ROOT / "llama.cpp" / "build-tcmoe" / "bin" / "Release" / "ggml-base.dll"


def parse_layers(spec, n_layers):
    if not spec or spec == "all":
        return list(range(n_layers))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [l for l in out if 0 <= l < n_layers]


def domain_score(s, dom_names, dom):
    if dom == "global":
        return s.sum(-1)
    if dom not in dom_names:
        raise SystemExit(f"unknown domain '{dom}', available: {dom_names} + ['global']")
    return s[:, :, dom_names.index(dom)]


def select_per_layer(score, frac):
    """Top-k experts per layer, returned as sorted (strictly increasing) index arrays."""
    n_layers, n_expert = score.shape
    k = max(1, int(n_expert * frac))
    return [np.sort(np.argsort(-score[l])[:k]).astype(np.int32) for l in range(n_layers)]


def load_ggml_dll(path):
    """The reference quantizers live in the C library; use them instead of a numpy port."""
    lib = ctypes.CDLL(str(path))
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.POINTER(ctypes.c_float)]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    return lib


def to_q4_k_rows(mat, lib):
    """[rows, cols] float32 -> Q4_K container bytes [rows, cols//256*144]."""
    rows, cols = mat.shape
    assert cols % 256 == 0 and rows and cols
    src = np.ascontiguousarray(mat, dtype=np.float32)
    out = np.empty((rows, cols // 256 * 144), dtype=np.uint8)
    n = lib.ggml_quantize_chunk(GGML_TYPE_Q4_K, src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                out.ctypes.data_as(ctypes.c_void_p), 0, rows, cols, None)
    assert n == out.size, f"quantized {n} bytes, expected {out.size}"
    return out


def to_q4_0_rows(mat):
    """[rows, cols] float32 -> Q4_0 container bytes [rows, cols//32*18].

    Quantization is moe_lib.quantize_int4 (group 128); the group scale is copied
    into each 32-value block and nibbles are reordered to the Q4_0 layout.
    """
    rows, cols = mat.shape
    assert cols % 128 == 0 and rows and cols
    packed, scale = quantize_int4(torch.from_numpy(np.ascontiguousarray(mat, dtype=np.float32)))
    p = packed.numpy()
    s = scale.numpy()

    q = np.empty((rows, cols), dtype=np.uint8)
    q[:, 0::2] = p & 0x0F
    q[:, 1::2] = (p >> 4) & 0x0F

    blk = q.reshape(rows, cols // 32, 32)
    qs = blk[:, :, :16] | (blk[:, :, 16:] << 4)

    dt = np.dtype([("d", "<f2"), ("qs", "u1", 16)])
    out = np.empty((rows, cols // 32), dtype=dt)
    out["d"] = np.repeat(s.astype("<f2"), 4, axis=1)
    out["qs"] = qs
    return np.ascontiguousarray(out.view(np.uint8).reshape(rows, cols // 32 * 18))


def to_tcmoe_q2_rows(q_bytes, scale_fp16, rows, cols):
    """[rows, cols//4] 2-bit codes + [rows, cols//128] fp16 scales -> tcmoe_q2 blocks."""
    q = np.ascontiguousarray(q_bytes).reshape(rows, cols // 128, 32)
    dt = np.dtype([("d", "<f2"), ("qs", "u1", 32)])
    out = np.empty((rows, cols // 128), dtype=dt)
    out["d"] = np.asarray(scale_fp16, dtype=np.float16)
    out["qs"] = q
    return np.ascontiguousarray(out.view(np.uint8).reshape(rows, cols // 128 * 34))


class ExpertReader:
    """Random access to expert weights of a safetensors model, with a small shard cache."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        index = json.load(open(model_dir / "model.safetensors.index.json", encoding="utf-8"))
        self.model_dir = model_dir
        self.weight_map = index["weight_map"]
        self.cache = {}

    def _shard(self, name):
        from safetensors import safe_open
        shard = self.weight_map[name]
        if shard not in self.cache:
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[shard] = safe_open(str(self.model_dir / shard), framework="pt")
        return self.cache[shard]

    def expert(self, layer, proj, expert, prefix="model.language_model.layers"):
        name = f"{prefix}.{layer}.mlp.experts.{proj}"
        return self._shard(name).get_slice(name)[expert].float().numpy()


class Packed2Base:
    """2-bit symmetric base from tcq_exp/stores/packed2* (per-expert parts)."""

    def __init__(self, p2_dir, n_layer, n_expert, hidden, moe_inter):
        sys.path.insert(0, str(ROOT / "tcq_exp"))
        import bitpack
        meta, self.part_off = bitpack.layout(2)
        self.meta = meta
        self.parts = meta["parts"]
        self.E, self.H, self.I = n_expert, hidden, moe_inter
        assert self.parts["dn_q"][1] == hidden and self.parts["dn_q"][2] == moe_inter // 4
        self.mm = np.memmap(Path(p2_dir) / "experts.bin", dtype=np.uint8, mode="r",
                            shape=(meta["total_bytes"],))
        self.stride = meta["layer_stride"]

    def _part(self, layer, name, expert):
        shape = self.parts[name]
        n = int(np.prod(shape[1:]))
        width = 2 if name.endswith("_s") else 1
        base = layer * self.stride + self.part_off[name] + expert * n * width
        view = self.mm[base:base + n * width]
        return view.view(np.float16 if name.endswith("_s") else np.uint8).reshape(shape[1:])

    def expert_tensors(self, layer, expert):
        """Returns (gate_rows, up_rows, down_rows) as tcmoe_q2 byte arrays."""
        gu_q = self._part(layer, "gu_q", expert)
        gu_s = self._part(layer, "gu_s", expert)
        dn_q = self._part(layer, "dn_q", expert)
        dn_s = self._part(layer, "dn_s", expert)

        gate = to_tcmoe_q2_rows(gu_q[:self.I], gu_s[:self.I], self.I, self.H)
        up = to_tcmoe_q2_rows(gu_q[self.I:], gu_s[self.I:], self.I, self.H)
        down = to_tcmoe_q2_rows(dn_q, dn_s, self.H, self.I)
        return gate, up, down

    def expert_tensors_recode(self, layer, expert, reader, cb_projs):
        """Re-derive the 2-bit indices from f32 weights under a new codebook.

        The packed store only carries uniform-codebook indices; when the decoder
        switches to a non-uniform shared table the indices must be reassigned,
        otherwise index k no longer means the level it was quantized to. The
        group scales stay exactly as stored (d = amax/1.5, fp16).
        """
        gu_s = self._part(layer, "gu_s", expert)
        dn_s = self._part(layer, "dn_s", expert)
        import bitpack
        gu = np.ascontiguousarray(reader.expert(layer, "gate_up_proj", expert),
                                  dtype=np.float32)
        dn = np.ascontiguousarray(reader.expert(layer, "down_proj", expert),
                                  dtype=np.float32)
        out = []
        for proj, w, s in (("gate", gu[:self.I], gu_s[:self.I]),
                           ("up", gu[self.I:], gu_s[self.I:]),
                           ("down", dn, dn_s)):
            cb = np.asarray(cb_projs[PROJECTIONS.index(proj)], dtype=np.float32)
            d32 = np.maximum(np.repeat(s.astype(np.float32), 128, axis=1), 1e-12)
            q = ((w / d32)[..., None] > ((cb[1:] + cb[:-1]) * 0.5)).sum(axis=-1)
            rows, cols = w.shape
            packed = bitpack.pack2(q.astype(np.uint8))
            out.append(to_tcmoe_q2_rows(packed, s, rows, cols))
        return tuple(out)


def copy_metadata(w, reader, skip_prefix=("tcmoe.", "GGUF."), skip_keys=("general.architecture",)):
    """Copy the skeleton metadata. 'GGUF.*' are synthetic header fields of GGUFReader, not real keys."""
    for key, field in reader.fields.items():
        if key.startswith(skip_prefix) or key in skip_keys:
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            w.add_key_value(key, list(field.contents()), GGUFValueType.ARRAY, field.types[-1])
        else:
            w.add_key_value(key, field.contents(), vtype)


def copy_tensors(w, reader, skip_names):
    for t in reader.tensors:
        if t.name in skip_names:
            continue
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)


def main():
    ap = argparse.ArgumentParser(description="pack a tcmoe GGUF")
    ap.add_argument("--model-dir", default=str(ROOT.parent / "model" / "Qwen3.6-35B-A3B"))
    ap.add_argument("--sens", default=str(ROOT / "tcq_exp" / "results" / "sens_matrix.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="global,math,code")
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--sensor", default="usage_topk_40x256")
    ap.add_argument("--arch", default="qwen35moe")
    ap.add_argument("--skeleton", default=None, help="existing GGUF whose non-expert tensors are copied")
    ap.add_argument("--base", default="external", choices=("external", "tcmoe_q2", "tcmoe_q2nl"))
    ap.add_argument("--p2-dir", default=str(ROOT / "tcq_exp" / "stores" / "packed2_sym4"))
    ap.add_argument("--cb", default=None,
                    help="codebook npz of --base tcmoe_q2nl (default: <p2-dir>/e21_cb.npz)")
    ap.add_argument("--recode-from", default=None,
                    help="tcmoe_q2nl: safetensors model dir; reassign 2-bit indices from "
                         "f32 weights under the --cb table instead of reusing uniform codes")
    ap.add_argument("--patch-type", default="q4_k", choices=("q4_0", "q4_k"),
                    help="q4_k rows go through the native Q4_K kernels, q4_0 keeps the old container")
    ap.add_argument("--ggml-dll", default=str(DEFAULT_GGML_DLL), help="ggml library that provides the Q4_K quantizer")
    args = ap.parse_args()

    sm = json.load(open(args.sens, encoding="utf-8"))
    s = np.array(sm["s"], dtype=np.float64)
    dom_names = sm["domains"]
    n_layers, n_expert = s.shape[0], s.shape[1]
    layers = parse_layers(args.layers, n_layers)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    masks = {d: select_per_layer(domain_score(s, dom_names, d), args.frac) for d in domains}

    reader = ExpertReader(args.model_dir)
    rec_reader = ExpertReader(args.recode_from) if args.recode_from else None
    skeleton = GGUFReader(args.skeleton) if args.skeleton else None

    hidden, moe_inter = 2048, 512
    skip_names = set()
    if skeleton is not None:
        for t in skeleton.tensors:
            if "ffn_gate_exps" in t.name or "ffn_up_exps" in t.name or "ffn_down_exps" in t.name:
                skip_names.add(t.name)
        for t in skeleton.tensors:
            if t.name.endswith(".ffn_gate_exps.weight"):
                # shape is the ggml ne order: (n_embd, n_ff_exp, n_expert)
                hidden = int(t.shape[0])
                moe_inter = int(t.shape[1])

    base = None
    base_dtype = GGMLQuantizationType.TCMOE_Q2
    cb_all = None
    if args.recode_from and args.base != "tcmoe_q2nl":
        raise SystemExit("--recode-from is only meaningful with --base tcmoe_q2nl")
    if args.base in ("tcmoe_q2", "tcmoe_q2nl"):
        if args.skeleton is None:
            raise SystemExit(f"--base {args.base} needs --skeleton for the non-expert tensors")
        base = Packed2Base(args.p2_dir, n_layers, n_expert, hidden, moe_inter)
        if args.base == "tcmoe_q2nl":
            base_dtype = GGMLQuantizationType.TCMOE_Q2NL
            cb_path = Path(args.cb) if args.cb else Path(args.p2_dir) / "e21_cb.npz"
            cb_all = np.load(cb_path)["cb"].astype(np.float32)  # [layer, gate/up/down, 4]
            if cb_all.shape != (n_layers, 3, 4):
                raise SystemExit(f"{cb_path}: codebook shape {cb_all.shape}, expected {(n_layers, 3, 4)}")
            # the AVX2 kernel stores the four levels as int8 times one scale, so round the
            # table here and write the rounded values: dequantize and vec_dot then agree
            flat = cb_all.reshape(-1, 4)
            amax = np.abs(flat).max(axis=1)
            scale = np.where(amax > 0.0, amax / 127.0, 1.0)
            c8 = np.clip(np.round(flat / scale[:, None]), -127, 127)
            cb_all = (c8 * scale[:, None]).reshape(n_layers, 3, 4).astype(np.float32)

    arch = args.arch
    if skeleton is not None:
        arch = skeleton.get_field("general.architecture").contents()

    if args.patch_type == "q4_k":
        patch_dtype = GGMLQuantizationType.Q4_K
        ggml = load_ggml_dll(args.ggml_dll)

        def qp(m):
            return to_q4_k_rows(m, ggml)

        for cols in (hidden, moe_inter):
            if cols % 256 != 0:
                raise SystemExit(f"Q4_K patch rows need a width divisible by 256, got {cols}")
    else:
        patch_dtype = GGMLQuantizationType.Q4_0
        qp = to_q4_0_rows

    def quant_expert(gu, dn):
        return qp(gu[:moe_inter]), qp(gu[moe_inter:]), qp(dn)

    pool = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1))

    w = GGUFWriter(args.out, arch, use_temp_file=True)
    if skeleton is not None:
        copy_metadata(w, skeleton)
        copy_tensors(w, skeleton, skip_names)

    w.add_uint32("tcmoe.version", 1)
    w.add_bool("tcmoe.enabled", True)
    w.add_string("tcmoe.base_mode", args.base)
    if args.patch_type == "q4_k":
        w.add_string("tcmoe.patch_type", "q4_k_rows")
        w.add_uint32("tcmoe.group", 256)
    else:
        w.add_string("tcmoe.patch_type", "q4_0_rows")
        w.add_uint32("tcmoe.group", 128)
    w.add_float32("tcmoe.patch_frac", args.frac)
    w.add_string("tcmoe.sensor", args.sensor)
    w.add_string("tcmoe.default_domain", domains[0])
    w.add_array("tcmoe.domains", domains)
    w.add_uint32("tcmoe.n_layer", len(layers))
    w.add_uint32("tcmoe.n_expert", n_expert)
    if cb_all is not None:
        # layer-major, gate/up/down, four levels per tensor
        w.add_array("tcmoe.cb", cb_all.reshape(-1).tolist())

    for l in layers:
        # Quantize the union of the per-domain experts once: domains overlap heavily, and
        # reading gate_up_proj twice per expert (gate + up) doubles the safetensors traffic.
        # The C quantizer and the torch one both release the GIL, so a small pool keeps the
        # core count busy; the window bounds how many f32 experts are in flight.
        union = sorted({int(e) for dom in domains for e in masks[dom][l]})
        quant = {}
        workers = min(16, os.cpu_count() or 1)
        for start in range(0, len(union), 2*workers):
            futs = {}
            for e in union[start:start + 2*workers]:
                gu = reader.expert(l, "gate_up_proj", e)
                dn = reader.expert(l, "down_proj", e)
                futs[e] = pool.submit(quant_expert, gu, dn)
            for e, f in futs.items():
                quant[e] = f.result()

        for d, dom in enumerate(domains):
            idx = masks[dom][l]
            w.add_tensor(f"blk.{l}.tcmoe.idx.{d}", idx.astype(np.int32))

            gate = np.stack([quant[int(e)][0] for e in idx])
            up = np.stack([quant[int(e)][1] for e in idx])
            down = np.stack([quant[int(e)][2] for e in idx])

            w.add_tensor(f"blk.{l}.tcmoe.gate.{d}", gate, raw_dtype=patch_dtype)
            w.add_tensor(f"blk.{l}.tcmoe.up.{d}", up, raw_dtype=patch_dtype)
            w.add_tensor(f"blk.{l}.tcmoe.down.{d}", down, raw_dtype=patch_dtype)

        if base is not None:
            gate = np.empty((n_expert, moe_inter, hidden // 128 * 34), dtype=np.uint8)
            up = np.empty_like(gate)
            down = np.empty((n_expert, hidden, moe_inter // 128 * 34), dtype=np.uint8)
            for e in range(n_expert):
                if rec_reader is not None:
                    gate[e], up[e], down[e] = base.expert_tensors_recode(
                        l, e, rec_reader, cb_all[l])
                else:
                    gate[e], up[e], down[e] = base.expert_tensors(l, e)
            w.add_tensor(f"blk.{l}.ffn_gate_exps.weight", gate, raw_dtype=base_dtype)
            w.add_tensor(f"blk.{l}.ffn_up_exps.weight", up, raw_dtype=base_dtype)
            w.add_tensor(f"blk.{l}.ffn_down_exps.weight", down, raw_dtype=base_dtype)

        print(f"[pack] layer {l}: " + ", ".join(f"{d}={len(masks[d][l])}" for d in domains), flush=True)

    pool.shutdown()
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[pack] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
