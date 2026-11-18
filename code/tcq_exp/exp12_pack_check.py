"""E12: 真实位宽打包的一致性/体积校验。

1) 从 BF16 重新量化 (bitpack.quant_group) 与 stores/int3|int2 的 nibble q/scale 逐项比对;
2) 读 stores/packed{3,2}/experts.bin 的实际字节, unpack 后与 q 完全一致;
3) 权重重建误差 (密排 3bit/2bit vs nibble INT3/INT2) 应完全相同;
4) pack/unpack 吞吐;
5) H2: 3bit 底座上 25% INT4 补丁的真实增量体积与占比。

用法: python exp12_pack_check.py
输出: results/packing.json
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
from moe_lib import Int4Store

N_SAMPLE = 256
SEED = 0
W_PER_EXPERT = 2 * bitpack.MOE_INTER * bitpack.HIDDEN + bitpack.HIDDEN * bitpack.MOE_INTER
N_HIGH = 2560          # 25% of 10240 (E3 的高精度名额)


def nibble_q(arr):
    """torch uint8 [rows, cols] (2 值/字节) -> numpy [rows, 2*cols] 的 +0 偏移值"""
    a = np.asarray(arr, dtype=np.uint8)
    q = np.empty((a.shape[0], a.shape[1] * 2), dtype=np.uint8)
    q[:, 0::2] = a & 0x0F
    q[:, 1::2] = a >> 4
    return q


def main():
    wmap = T.load_weight_map()
    p_json = os.path.join(T.RES_DIR, "packing.json")
    cur = json.load(open(p_json, encoding="utf-8")) if os.path.exists(p_json) else {}
    out = {"n_sample": N_SAMPLE, "groups": {}, "consistency": {},
           "throughput": {}, "patch": {}}

    rng = np.random.default_rng(SEED)
    L, E = bitpack.N_LAYERS, bitpack.N_EXPERTS
    pairs = [(int(rng.integers(0, L)), int(rng.integers(0, E))) for _ in range(N_SAMPLE)]
    pairs = sorted(set(pairs))

    for bits in (3, 2):
        nb_dir = T.STORE_INT3 if bits == 3 else T.STORE_INT2
        pk_dir = os.path.join(HERE, "stores", f"packed{bits}")
        nb = Int4Store(nb_dir, mmap_mode="r")
        meta, part_off = bitpack.layout(bits)
        pk_mm = np.memmap(os.path.join(pk_dir, "experts.bin"), dtype=np.uint8,
                          mode="r", shape=(meta["total_bytes"],))

        t0 = time.time()
        n_ok_q = n_ok_s = n_ok_pk = 0
        max_wdiff = 0.0
        mse = 0.0
        for layer, e in pairs:
            w_gu, w_dn = T.read_expert_bf16(wmap, T.MOE_MODEL_PATH, layer, e)
            v = nb.layer_views(layer)
            for tag, w in (("gu", w_gu), ("dn", w_dn)):
                q_u, sc = bitpack.quant_group(w, bits)
                q_np = q_u.numpy()
                q_signed = q_np.astype(np.int16) - 2 ** (bits - 1)
                # nibble store 的 q 是 +8 偏移
                q_nib = nibble_q(v[f"{tag}_q"][e]).astype(np.int16) - 8
                n_ok_q += int(np.array_equal(q_signed, q_nib))
                s_nib = np.asarray(v[f"{tag}_s"][e], dtype=np.float16)
                n_ok_s += int(np.array_equal(sc.numpy(), s_nib))

                # 密排文件里的实际字节
                part = f"{tag}_q"
                seg_n = int(np.prod(meta["parts"][part][1:]))
                off = (layer * meta["layer_stride"] + part_off[part] +
                       e * seg_n)
                seg = np.asarray(pk_mm[off:off + seg_n])
                q_pk = bitpack.unpack(seg, bits).reshape(q_np.shape)
                n_ok_pk += int(np.array_equal(q_pk, q_np))

                # 权重重建: 密排解包 vs nibble 反量化 (应完全相同)
                w_pk = bitpack.dequant_group(torch.from_numpy(q_np), sc, bits)
                packed4 = np.asarray(v[f"{tag}_q"][e])
                nb_t = torch.from_numpy(packed4.copy())
                w_nb = T.dequant_int4(nb_t, v[f"{tag}_s"][e], torch.float32, nb.group)
                max_wdiff = max(max_wdiff, float((w_pk - w_nb).abs().max()))
                mse += float(((w_pk - w.float()) ** 2).mean())
        n_pairs_parts = len(pairs) * 2
        out["consistency"][f"int{bits}"] = {
            "n_checked": n_pairs_parts,
            "q_equal": n_ok_q, "scale_equal": n_ok_s, "packed_file_equal": n_ok_pk,
            "weight_max_abs_diff_vs_nibble": max_wdiff,
            "weight_mse_vs_bf16": mse / n_pairs_parts,
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"[check int{bits}] q {n_ok_q}/{n_pairs_parts} scale {n_ok_s} "
              f"packed {n_ok_pk} max_wdiff={max_wdiff:.2e} "
              f"mse={mse/n_pairs_parts:.3e} ({time.time()-t0:.0f}s)", flush=True)
        del pk_mm

    # ---- pack/unpack 吞吐 (单层 gate_up 规模) ----
    for bits in (3, 2):
        q_big = rng.integers(0, 2 ** bits, size=(bitpack.N_EXPERTS * 2 * bitpack.MOE_INTER,
                                                 bitpack.HIDDEN), dtype=np.uint8)
        t0 = time.perf_counter()
        p = bitpack.pack(q_big, bits)
        t_pack = time.perf_counter() - t0
        t0 = time.perf_counter()
        q2 = bitpack.unpack(p, bits)
        t_unp = time.perf_counter() - t0
        assert np.array_equal(q2, q_big)
        in_mb = q_big.nbytes / 2 ** 20
        out["throughput"][f"int{bits}"] = {
            "pack_MBps": round(in_mb / t_pack, 1),
            "unpack_MBps": round(in_mb / t_unp, 1),
            "packed_MB": round(p.nbytes / 2 ** 20, 1),
        }
        print(f"[throughput int{bits}] pack {in_mb/t_pack:.1f} MB/s "
              f"unpack {in_mb/t_unp:.1f} MB/s", flush=True)
        del p, q2

    # ---- H2: 补丁增量体积 ----
    p3 = cur.get("sizes", {}).get("packed3", {})
    p3_gib = p3.get("gib", os.path.getsize(os.path.join(HERE, "stores", "packed3",
                                                        "experts.bin")) / 2 ** 30)
    patch_bytes = N_HIGH * W_PER_EXPERT // 8        # 1 bit/weight 增量
    out["patch"] = {
        "n_high_experts": N_HIGH,
        "weights_per_expert": W_PER_EXPERT,
        "delta_bits": 1,
        "patch_gib": round(patch_bytes / 2 ** 30, 3),
        "base3_gib": round(p3_gib, 2),
        "patch_ratio": round(patch_bytes / 2 ** 30 / p3_gib, 3),
    }
    print(f"[patch] {N_HIGH} experts, 1bit delta -> {patch_bytes/2**30:.3f} GiB "
          f"= {100*patch_bytes/2**30/p3_gib:.1f}% of 3bit base", flush=True)

    p = os.path.join(T.RES_DIR, "packing.json")
    cur.update(out)
    json.dump(cur, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[packing] saved {p}", flush=True)


if __name__ == "__main__":
    main()
