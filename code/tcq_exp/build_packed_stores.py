"""离线构建真实 3bit / 2bit 专家库 (密排格式, 仅供体积与一致性校验; 引擎不读取)。

用法: python build_packed_stores.py --bits 3 [--force] [--out stores/packed3]

布局与 nibble 库同序: [layer][expert][gu_q | gu_s | dn_q | dn_s],
其中 q 按 bits 密排 (3bit: 8 值 3 字节; 2bit: 4 值 1 字节), scale fp16 不变。
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import safetensors

import bitpack
import tcq_lib as T

RESULTS = T.RES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, choices=[2, 3], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sym4", action="store_true",
                    help="bits=2 时改用 4 级对称量化 (码 k -> (k-1.5)s, s=amax/1.5), 内核需按 1.5 解码")
    ap.add_argument("--layers", type=int, default=None, help="仅构建前 N 层(调试)")
    a = ap.parse_args()

    out_dir = a.out or os.path.join(HERE, "stores", f"packed{a.bits}")
    os.makedirs(out_dir, exist_ok=True)
    meta, part_off = bitpack.layout(a.bits)
    if a.sym4:
        assert a.bits == 2, "--sym4 只对 2bit 有意义"
        meta["scheme"] = "sym4-int2-packed-group128-scale-fp16"
    if a.layers:
        meta["num_layers"] = a.layers
        meta["total_bytes"] = meta["layer_stride"] * a.layers
    idx_p = os.path.join(out_dir, "index.json")
    bin_p = os.path.join(out_dir, "experts.bin")

    if os.path.exists(idx_p) and os.path.exists(bin_p) and not a.force:
        old = json.load(open(idx_p, encoding="utf-8"))
        if (old.get("bits") == a.bits and old.get("total_bytes") == meta["total_bytes"]
                and old.get("scheme") == meta["scheme"]):
            print(f"[packed{a.bits}] exists, skip ({old['total_bytes']/2**30:.2f} GiB)")
            return

    wmap = T.load_weight_map()
    shards = {}
    t_all = time.time()

    def shard(key):
        sh = wmap[key]
        if sh not in shards:
            shards[sh] = safetensors.safe_open(os.path.join(T.MOE_MODEL_PATH, sh), framework="pt")
        return shards[sh]

    mm = np.memmap(bin_p, dtype=np.uint8, mode="w+", shape=(meta["total_bytes"],))
    E = meta["num_experts"]
    H = meta["hidden"]
    I = meta["moe_inter"]
    G = meta["group"]

    def emit(layer, part, arr):
        assert arr.flags["C_CONTIGUOUS"], part
        raw = arr.view(np.uint8).reshape(-1)
        off = layer * meta["layer_stride"] + part_off[part]
        nbytes = raw.shape[0]
        assert nbytes == int(np.prod(meta["parts"][part])) * (2 if part.endswith("_s") else 1)
        mm[off:off + nbytes] = raw

    def quant(w):
        if a.sym4:
            return bitpack.quant2_group(w, G)
        return bitpack.quant_group(w, a.bits, G)

    n_build = meta["num_layers"]
    for layer in range(n_build):
        t0 = time.time()
        gu_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        dn_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        gu = shard(gu_key).get_tensor(gu_key).reshape(E * 2 * I, H)
        dn = shard(dn_key).get_tensor(dn_key).reshape(E * H, I)

        q, sc = quant(gu)
        pq = bitpack.pack(q.numpy(), a.bits).reshape(E, 2 * I, -1)
        emit(layer, "gu_q", pq)
        emit(layer, "gu_s", sc.numpy().reshape(E, 2 * I, H // G))
        del q, pq, sc

        q, sc = quant(dn)
        pq = bitpack.pack(q.numpy(), a.bits).reshape(E, H, -1)
        emit(layer, "dn_q", pq)
        emit(layer, "dn_s", sc.numpy().reshape(E, H, I // G))
        del q, pq, sc, gu, dn

        if layer % 4 == 0 or layer == n_build - 1:
            print(f"[packed{a.bits}] layer {layer} done ({time.time()-t0:.1f}s)", flush=True)

    mm.flush()
    del mm
    json.dump(meta, open(idx_p, "w", encoding="utf-8"), indent=1)

    size = os.path.getsize(bin_p)
    info = {"bits": a.bits, "path": out_dir, "bytes": size,
            "gib": round(size / 2 ** 30, 2),
            "nibble_gib": 15.47,
            "ratio_vs_nibble": round(size / 2 ** 30 / 15.47, 3)}
    p = os.path.join(RESULTS, "packing.json")
    cur = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    cur.setdefault("sizes", {})[f"packed{a.bits}" + ("_sym4" if a.sym4 else "")] = info
    cur["elapsed_s_build"] = round(cur.get("elapsed_s_build", 0) + time.time() - t_all, 1)
    json.dump(cur, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[packed{a.bits}] saved {bin_p} {size/2**30:.2f} GiB "
          f"({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
