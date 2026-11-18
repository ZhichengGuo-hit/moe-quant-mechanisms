"""构建低精度专家库(INT2/INT3), 布局与 int4_store 完全一致(group=128, nibble打包)。
用于任务条件化配额实验的低位宽底座。输出:
  tcq_exp/stores/int2/{experts.bin,index.json}
  tcq_exp/stores/int3/{experts.bin,index.json}
"""
import os
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.join(os.path.dirname(HERE), "coserving")
import sys
sys.path.insert(0, sys_path)

import numpy as np
import torch
import safetensors

MODEL_PATH = r"E:\CUDA11.8\model\Qwen3.6-35B-A3B"
OUT_ROOT = os.path.join(HERE, "stores")
GROUP = 128
CHUNK_E = 32


def quantize_int(w: torch.Tensor, bits: int, group: int = GROUP):
    """w [E*ROWS, IN] fp32 -> packed uint8 [E*ROWS, IN/2], scale fp16 [E*ROWS, IN/group]"""
    maxq = 2 ** (bits - 1) - 1
    EROWS, IN = w.shape
    wg = w.reshape(EROWS, IN // group, group)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / maxq).clamp(min=1e-8)
    q = torch.clamp(torch.round(wg / scale), -maxq, maxq).to(torch.int8) + 8
    q = q.reshape(EROWS, IN).to(torch.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed, scale.squeeze(-1).to(torch.float16)


def load_index():
    with open(os.path.join(MODEL_PATH, "model.safetensors.index.json"),
              encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def build(bits: int, wmap, meta_base):
    out_dir = os.path.join(OUT_ROOT, f"int{bits}")
    os.makedirs(out_dir, exist_ok=True)
    total = meta_base["total_bytes"]
    path = os.path.join(out_dir, "experts.bin")
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(total,))
    parts = meta_base["parts"]
    stride = meta_base["layer_stride"]
    # 计算part偏移
    off = 0
    part_off = {}
    for p in ("gu_q", "gu_s", "dn_q", "dn_s"):
        part_off[p] = off
        off += int(np.prod(parts[p])) * (2 if p.endswith("_s") else 1)
    assert off == stride

    shard_cache = {}
    t0 = time.time()
    for layer in range(meta_base["num_layers"]):
        gu_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        dn_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        gu = dn = None
        for key in (gu_key, dn_key):
            shard = wmap[key]
            f = shard_cache.get(shard)
            close = False
            if f is None:
                f = safetensors.safe_open(os.path.join(MODEL_PATH, shard), framework="pt")
                shard_cache[shard] = f
            t = f.get_tensor(key)  # [E,2I,H] / [E,H,I]
            if key == gu_key:
                gu = t
            else:
                dn = t
        E, r_gu, H = gu.shape
        E2, H2, I = dn.shape
        base = layer * stride
        for c0 in range(0, E, CHUNK_E):
            c1 = min(c0 + CHUNK_E, E)
            pq, sq = quantize_int(gu[c0:c1].reshape(-1, H).float(), bits)
            pd, sd = quantize_int(dn[c0:c1].reshape(-1, I).float(), bits)
            nq = (c1 - c0) * r_gu
            nd = (c1 - c0) * H2
            # gu_q [E,2I,H/2]
            mm[base + part_off["gu_q"] + c0 * r_gu * (H // 2):
               base + part_off["gu_q"] + c1 * r_gu * (H // 2)] = pq.numpy().view(np.uint8).reshape(-1)
            mm[base + part_off["gu_s"] + c0 * r_gu * (H // GROUP) * 2:
               base + part_off["gu_s"] + c1 * r_gu * (H // GROUP) * 2] = sq.numpy().view(np.uint8).reshape(-1)
            mm[base + part_off["dn_q"] + c0 * H2 * (I // 2):
               base + part_off["dn_q"] + c1 * H2 * (I // 2)] = pd.numpy().view(np.uint8).reshape(-1)
            mm[base + part_off["dn_s"] + c0 * H2 * (I // GROUP) * 2:
               base + part_off["dn_s"] + c1 * H2 * (I // GROUP) * 2] = sd.numpy().view(np.uint8).reshape(-1)
        mm.flush()
        print(f"[int{bits}] layer {layer} done ({time.time()-t0:.0f}s)", flush=True)
    meta = dict(meta_base)
    meta["scheme"] = f"sym-int{bits}-group{GROUP}-scale-fp16"
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[int{bits}] finished {time.time()-t0:.0f}s -> {out_dir}", flush=True)


def main():
    wmap = load_index()
    with open(os.path.join(sys_path, "int4_store", "index.json"), encoding="utf-8") as f:
        meta_base = json.load(f)
    for bits in (3, 2):
        build(bits, wmap, meta_base)


if __name__ == "__main__":
    main()
