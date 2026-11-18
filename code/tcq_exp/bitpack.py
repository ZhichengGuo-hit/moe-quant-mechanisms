"""真实低位宽打包 (离线校验用, 不改 moe_lib/int4_ext 运行时)。

nibble 容器: INT2/INT3/INT4 都占 4bit/权重 (每字节 2 个槽), 文件均 15.47GB。
这里实现 3bit(8 值 -> 3 字节) 与 2bit(4 值 -> 1 字节) 的密排, 量化规则与
tcq_lib.quantize_bits 完全一致 (group=128 对称量化, fp16 scale), 只把
符号偏移改成无符号 (3bit: q+4 -> 1..7; 2bit: q+2 -> 1..3)。
"""
import numpy as np
import torch

GROUP = 128
HIDDEN = 2048
MOE_INTER = 512
N_LAYERS = 40
N_EXPERTS = 256


# ---------------------------------------------------------------- 量化
def quant_group(w, bits, group=GROUP):
    """w [ROWS,IN] (torch, 任意精度) -> (q uint8 [ROWS,IN] 无符号偏移, scale fp16 [ROWS,IN/group])"""
    maxq = 2 ** (bits - 1) - 1
    off = 2 ** (bits - 1)
    ROWS, IN = w.shape
    assert IN % group == 0
    wg = w.float().reshape(ROWS, IN // group, group)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / maxq).clamp(min=1e-8)
    q = torch.clamp(torch.round(wg / scale), -maxq, maxq) + off
    q = q.reshape(ROWS, IN).to(torch.uint8)
    return q, scale.squeeze(-1).to(torch.float16)


def dequant_group(q, scale, bits, group=GROUP):
    """q 无符号偏移 -> float32 权重 [ROWS,IN]"""
    ROWS, IN = q.shape
    off = 2 ** (bits - 1)
    x = (q.to(torch.float32) - off).reshape(ROWS, IN // group, group)
    return (x * scale.float().unsqueeze(-1)).reshape(ROWS, IN)


# ---------------------------------------------------------------- 4 级 INT2
# quant_group(bits=2) 用 maxq=1, 只有 3 个码 {-1,0,1} (步长 amax), 4 个码浪费 1 个。
# 标准 4 级对称 INT2: 码 k=0..3 -> (k-1.5)s, s=amax/1.5, 即 {-amax,-amax/3,amax/3,amax},
# 与 Q2_K 的 (min + s*q) 取 min=-max 时完全同集。需要内核按 (q-1.5) 解码 (dot_int2)。
def quant2_group(w, group=GROUP):
    ROWS, IN = w.shape
    wg = w.float().reshape(ROWS, IN // group, group)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 1.5).clamp(min=1e-8)
    q = torch.clamp(torch.round(wg / scale + 1.5), 0, 3)
    q = q.reshape(ROWS, IN).to(torch.uint8)
    return q, scale.squeeze(-1).to(torch.float16)


def dequant2_group(q, scale, group=GROUP):
    ROWS, IN = q.shape
    x = (q.to(torch.float32) - 1.5).reshape(ROWS, IN // group, group)
    return (x * scale.float().unsqueeze(-1)).reshape(ROWS, IN)


# ---------------------------------------------------------------- 打包
def pack3(q):
    """q [...,N] (0..7, N%8==0) -> [..., N*3/8]"""
    n = q.shape[-1] // 8
    v = q.reshape(*q.shape[:-1], n, 8).astype(np.uint8)
    out = np.empty((*q.shape[:-1], n, 3), dtype=np.uint8)
    out[..., 0] = v[..., 0] | (v[..., 1] << 3) | ((v[..., 2] & 0x3) << 6)
    out[..., 1] = (v[..., 2] >> 2) | (v[..., 3] << 1) | (v[..., 4] << 4) | ((v[..., 5] & 0x1) << 7)
    out[..., 2] = (v[..., 5] >> 1) | (v[..., 6] << 2) | (v[..., 7] << 5)
    return out.reshape(*q.shape[:-1], n * 3)


def unpack3(p):
    """[...,M] (M%3==0) -> [..., M*8/3] (0..7)"""
    n = p.shape[-1] // 3
    b = p.reshape(*p.shape[:-1], n, 3).astype(np.uint8)
    out = np.empty((*p.shape[:-1], n, 8), dtype=np.uint8)
    out[..., 0] = b[..., 0] & 0x7
    out[..., 1] = (b[..., 0] >> 3) & 0x7
    out[..., 2] = ((b[..., 0] >> 6) & 0x3) | ((b[..., 1] & 0x1) << 2)
    out[..., 3] = (b[..., 1] >> 1) & 0x7
    out[..., 4] = (b[..., 1] >> 4) & 0x7
    out[..., 5] = ((b[..., 1] >> 7) & 0x1) | ((b[..., 2] & 0x3) << 1)
    out[..., 6] = (b[..., 2] >> 2) & 0x7
    out[..., 7] = (b[..., 2] >> 5) & 0x7
    return out.reshape(*p.shape[:-1], n * 8)


def pack2(q):
    """q [...,N] (0..3, N%4==0) -> [..., N/4]"""
    n = q.shape[-1] // 4
    v = q.reshape(*q.shape[:-1], n, 4).astype(np.uint8)
    out = (v[..., 0] | (v[..., 1] << 2) | (v[..., 2] << 4) | (v[..., 3] << 6))
    return out.reshape(*q.shape[:-1], n)


def unpack2(p):
    n = p.shape[-1]
    b = p.astype(np.uint8)
    out = np.empty((*p.shape[:-1], n, 4), dtype=np.uint8)
    out[..., 0] = b & 0x3
    out[..., 1] = (b >> 2) & 0x3
    out[..., 2] = (b >> 4) & 0x3
    out[..., 3] = (b >> 6) & 0x3
    return out.reshape(*p.shape[:-1], n * 4)


def pack(q, bits):
    return pack3(q) if bits == 3 else pack2(q)


def unpack(p, bits):
    return unpack3(p) if bits == 3 else unpack2(p)


# ---------------------------------------------------------------- 布局
def layout(bits, num_layers=N_LAYERS, num_experts=N_EXPERTS,
           hidden=HIDDEN, moe_inter=MOE_INTER, group=GROUP):
    """返回 (meta, part_off)；parts[p] 为单专家张量形状, q 仍以 [rows, packed_cols] 表示"""
    hq = hidden * bits // 8
    iq = moe_inter * bits // 8
    parts = {
        "gu_q": [num_experts, 2 * moe_inter, hq],
        "gu_s": [num_experts, 2 * moe_inter, hidden // group],
        "dn_q": [num_experts, hidden, iq],
        "dn_s": [num_experts, hidden, moe_inter // group],
    }
    off, part_off, expert_bytes = 0, {}, 0
    for p in ("gu_q", "gu_s", "dn_q", "dn_s"):
        part_off[p] = off
        n = int(np.prod(parts[p][1:])) * (2 if p.endswith("_s") else 1)
        expert_bytes += n
        off += n * num_experts
    assert off == expert_bytes * num_experts
    layer_stride = expert_bytes * num_experts
    meta = {"version": 1, "bits": bits, "group": group, "num_layers": num_layers,
            "num_experts": num_experts, "hidden": hidden, "moe_inter": moe_inter,
            "parts": parts, "expert_bytes": expert_bytes,
            "layer_stride": layer_stride,
            "total_bytes": layer_stride * num_layers,
            "scheme": f"sym-int{bits}-packed-group{group}-scale-fp16"}
    return meta, part_off


if __name__ == "__main__":
    # 自检: pack/unpack 往返一致
    rng = np.random.default_rng(0)
    for bits, mx in ((3, 7), (2, 3)):
        q = rng.integers(0, mx + 1, size=(7, 128), dtype=np.uint8)
        assert np.array_equal(unpack(pack(q, bits), bits), q), bits
        print(f"[bitpack] bits={bits} roundtrip OK")
    for bits in (2, 3):
        meta, _ = layout(bits)
        print(f"[bitpack] int{bits}: total {meta['total_bytes']/2**30:.2f} GiB "
              f"(nibble 15.47 GiB)")
