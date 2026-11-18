"""临时检查: 补丁行 w4 能否用 "3bit 码 + 1 bit 残差" 编码 (即 packing.json 的 3.375bit 口径).
逐元素看 |w4-w3| 相对 4bit 网格步长 s4 的分布; 若绝大多数 <= 1 步 -> 1bit 可编码, 否则不可。
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import tcq_lib as T
from d1_residual_spectrum import Packed3Store, deq3, deq4

def bits_q(t, bits, group=128):
    rows, cols = t.shape
    maxq = 2 ** (bits - 1) - 1
    tg = t.reshape(rows, cols // group, group)
    amax = tg.abs().amax(-1, keepdim=True)
    scale = (amax / maxq).clamp(min=1e-8)
    q = torch.clamp(torch.round(tg / scale), -maxq, maxq)
    return q, scale

import torch
from moe_lib import Int4Store
q4 = Int4Store(T.STORE_INT4, mmap_mode="r")
E = q4.num_experts
p3 = Packed3Store()
he_gu, he_dn = [], []
for l in (0, 10, 20, 30):
    v4 = q4.layer_views(l)
    for e in range(0, 256, 32):
        for proj, N in (("gu", 2048), ("dn", 512)):
            part = f"{proj}_q"
            w3 = deq3(p3, l, e, part).float()
            w4 = deq4(v4, e, part, N).float()
            q3, s3 = bits_q(w3, 3)
            q4c, s4 = bits_q(w4, 4)
            d = (w4 - w3).abs() / s4.squeeze(-1).repeat_interleave(128, dim=1)
            d = d.reshape(-1).numpy()
            frac = [float((d <= k + 0.5).mean()) for k in (0, 1, 2, 3)]
            rec = {"proj": proj, "l": l, "e": e, "n": int(d.size),
                   "median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
                   "le_0.5": frac[0], "le_1.5": frac[1], "le_2.5": frac[2], "le_3.5": frac[3],
                   "s3_over_s4": float((s3.squeeze(-1) / s4.squeeze(-1)).median())}
            (he_gu if proj == "gu" else he_dn).append(rec)
            if l == 0 and e == 0:
                print(json.dumps(rec, ensure_ascii=False), flush=True)

out = {}
for name, rs in (("gu", he_gu), ("dn", he_dn)):
    out[name] = {k: float(np.median([r[k] for r in rs]))
                 for k in ("median", "p99", "le_0.5", "le_1.5", "le_2.5", "le_3.5", "s3_over_s4")}
    out[name]["n_samples"] = len(rs)
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
with open(os.path.join(T.RES_DIR, "d1c_delta_bits.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
