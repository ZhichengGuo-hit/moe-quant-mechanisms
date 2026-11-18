"""冒烟测试: 验证 Qwen3.6-35B-A3B INT4 引擎在本机(16GB GPU + CPU offload)可跑前向,
并测量 prefill 吞吐、显存/内存占用、路由捕获与逐专家BF16/INT4一致性。
运行:
  C:\\Users\\administered\\miniconda3\\envs\\unsloth\\python.exe smoke_probe.py
"""
import os
import sys
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
COSERVING = os.path.join(os.path.dirname(HERE), "coserving")
sys.path.insert(0, COSERVING)

import torch
from moe_lib import (MOE_MODEL_PATH, STORE_DIR, Int4Store, ExpertCache,
                     build_moe_model)


def vram():
    return round(torch.cuda.memory_allocated() / 2**30, 2)


def main():
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MOE_MODEL_PATH, trust_remote_code=True)
    print(f"[tok] loaded {time.time()-t0:.1f}s", flush=True)

    store = Int4Store(STORE_DIR, mmap_mode="r")
    cache = ExpertCache(store, device="cuda", dtype=torch.float16, slots=512,
                        pin_pool=8, trace=False, prefetch=False)
    t1 = time.time()
    model = build_moe_model("gpu", MOE_MODEL_PATH, device="cuda",
                            dtype=torch.float16, store=store, cache=cache,
                            cpu_cache=None, direct_ext=None)
    print(f"[model] built {time.time()-t1:.1f}s vram={vram()}GB", flush=True)

    text = ("Solve the following problem step by step.\n"
            "A store sells notebooks for $3 each and pens for $2 each. "
            "Mia buys 7 notebooks and 5 pens. How much does she pay in total?\n"
            "Answer: 7*3 + 5*2 = 21 + 10 = 31 dollars.\n"
            "Now: A train travels 60 km/h for 2.5 hours and 80 km/h for 1.5 hours. "
            "What is the total distance? Answer: 60*2.5 + 80*1.5 = 150 + 120 = 270 km.\n")
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    print(f"[data] {ids.shape[1]} tokens", flush=True)

    # 捕获前6层 MoE 输入
    captured = {}

    def make_hook(li):
        def hook(mod, inp, out):
            if li < 6:
                captured[li] = inp[0].detach().float().cpu()
        return hook

    hs = [model.model.language_model.layers[i].mlp.register_forward_hook(make_hook(i))
          for i in range(len(model.model.language_model.layers))]

    with torch.no_grad():
        t2 = time.time()
        out = model(input_ids=ids, use_cache=False)
        dt = time.time() - t2
    n = ids.shape[1]
    print(f"[fwd] {n} tokens in {dt:.2f}s -> {n/dt:.1f} tok/s prefill, "
          f"vram={vram()}GB", flush=True)

    logits = out.logits[:, :-1, :].float()
    labels = ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    print(f"[nll] loss={loss.item():.4f}", flush=True)

    # 路由统计 (层0)
    x0 = captured[0][0]  # [S,H]
    mlp0 = model.model.language_model.layers[0].mlp
    logits_r = torch.nn.functional.linear(x0.cuda(), mlp0.gate.weight).float()
    probs = torch.softmax(logits_r, dim=-1)
    topv, topi = torch.topk(probs, 8, dim=-1)
    uniq = torch.unique(topi).numel()
    print(f"[route] layer0 tokens={x0.shape[0]} unique_experts={uniq} "
          f"top1_share={topv[:,0].mean().item():.4f}", flush=True)

    # 逐专家 BF16 vs INT4 输出误差 (层0, 专家0)
    import safetensors
    idx_path = os.path.join(MOE_MODEL_PATH, "model.safetensors.index.json")
    wmap = json.load(open(idx_path, encoding="utf-8"))["weight_map"]
    e = 0
    key_gu = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    key_dn = "model.language_model.layers.0.mlp.experts.down_proj"
    shard_gu = wmap[key_gu]
    with safetensors.safe_open(os.path.join(MOE_MODEL_PATH, shard_gu),
                               framework="pt") as f:
        w_bf16 = f.get_slice(key_gu)[e]  # [2I,H]
    w_bf16 = w_bf16.float()

    from moe_lib import dequant_int4
    v = store.layer_views(0)
    w_int4 = dequant_int4(v["gu_q"][e], v["gu_s"][e], torch.float32, store.group)
    rel = (w_int4 - w_bf16).norm() / w_bf16.norm()
    print(f"[expert] layer0 e0 gate_up weight rel_err={rel.item():.4f}", flush=True)

    # 激活上的输出误差
    xi = x0[:16]  # 16 tokens
    y_bf = xi @ w_bf16.T   # [16,2I]
    y_q = xi @ w_int4.T
    g1, u1 = y_bf.chunk(2, dim=-1)
    g2, u2 = y_q.chunk(2, dim=-1)
    h1 = torch.nn.functional.silu(g1) * u1
    h2 = torch.nn.functional.silu(g2) * u2
    act_rel = (h2 - h1).norm(dim=-1).mean().item() / h1.norm(dim=-1).mean().item()
    print(f"[expert] layer0 e0 activation rel_err={act_rel:.4f}", flush=True)

    for h in hs:
        h.remove()
    print(f"[done] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
