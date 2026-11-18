"""实验1: 全量(40层×256专家×3域)任务条件敏感度矩阵。
信号: s(e,l,D) = sum_t w_e(t) * ||FFN_int4(x_t)-FFN_bf16(x_t)|| / ||FFN_bf16(x_t)||
即域D下被判给专家e的token上, INT4相对BF16的专家输出相对误差(路由权重加权)。
同时记录 usage(e,l,D) = 路由权重占比。
输出: results/sens_matrix.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
import safetensors

import tcq_lib as T

N_SEQ = 12


def expert_out(x, w_gu, w_dn):
    """x [T,H] fp32 -> [T,H]"""
    gu = x @ w_gu.t()
    g, u = gu.chunk(2, dim=-1)
    h = F.silu(g) * u
    return h @ w_dn.t()


def main():
    eng = T.build_engine()
    tok = eng["tok"]
    wmap = T.load_weight_map()
    t_all = time.time()

    # 1) 三域校准集前向捕获
    caps = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "calib")[:N_SEQ]
        texts = [r["text"] for r in rows]
        t0 = time.time()
        caps[dom] = T.capture_layer_stats(eng, texts, max_len=96, batch_size=6)
        print(f"[capture] {dom}: {len(texts)} seq {time.time()-t0:.1f}s", flush=True)

    n_layers = len(eng["model"].model.language_model.layers)
    n_experts = eng["store"].num_experts
    s = np.zeros((n_layers, n_experts, len(T.DOMAINS)), dtype=np.float64)
    usage = np.zeros_like(s)
    assign_cnt = np.zeros((n_layers, n_experts), dtype=np.int64)

    # 2) 逐层读取 BF16 专家 -> 计算域条件误差
    shard_files = {}
    for dom_i, dom in enumerate(T.DOMAINS):
        pass
    for layer in range(n_layers):
        t0 = time.time()
        gu_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        dn_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        for key in (gu_key, dn_key):
            sh = wmap[key]
            if sh not in shard_files:
                shard_files[sh] = safetensors.safe_open(
                    os.path.join(T.MOE_MODEL_PATH, sh), framework="pt")
        gu_all = shard_files[wmap[gu_key]].get_tensor(gu_key)
        dn_all = shard_files[wmap[dn_key]].get_tensor(dn_key)

        # 每域 token 的路由清单: {e: [(domain_idx, token_idx, weight), ...]}
        per_dom = []
        for dom in T.DOMAINS:
            x = caps[dom][layer]["x"]       # [T,H] fp16
            topi = caps[dom][layer]["topi"].long()
            topv = caps[dom][layer]["topv"].float()
            per_dom.append((x, topi, topv))

        for e in range(n_experts):
            w_gu = gu_all[e].float()
            w_dn = dn_all[e].float()
            # 反量化 INT4 (复用 store)
            v = eng["store"].layer_views(layer)
            q_gu = T.dequant_int4(v["gu_q"][e], v["gu_s"][e], torch.float32, eng["store"].group)
            q_dn = T.dequant_int4(v["dn_q"][e], v["dn_s"][e], torch.float32, eng["store"].group)

            for di, (x, topi, topv) in enumerate(per_dom):
                # 找出路由到 e 的 (token, slot)
                mask = topi == e
                tok_idx, slot = torch.nonzero(mask, as_tuple=True)
                if tok_idx.numel() == 0:
                    continue
                wts = topv[tok_idx, slot].numpy()
                assign_cnt[layer, e] += int(tok_idx.numel())
                xe = x[tok_idx].float()
                y_b = expert_out(xe, w_gu, w_dn)
                y_q = expert_out(xe, q_gu, q_dn)
                rel = ((y_q - y_b).norm(dim=-1) / (y_b.norm(dim=-1) + 1e-6)).numpy()
                s[layer, e, di] += float((wts * rel).sum())
                usage[layer, e, di] += float(wts.sum())

        if layer % 4 == 0:
            print(f"[sens] layer {layer} done {time.time()-t0:.1f}s", flush=True)

    out = {
        "n_layers": n_layers, "n_experts": n_experts, "domains": list(T.DOMAINS),
        "n_seq": N_SEQ,
        "signal": "route-weight-weighted relative expert FFN output error (INT4 vs BF16)",
        "s": s.tolist(), "usage": usage.tolist(), "assign_cnt": assign_cnt.tolist(),
        "elapsed_s": round(time.time() - t_all, 1),
    }
    p = os.path.join(T.RES_DIR, "sens_matrix.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[sens] saved {p} ({time.time()-t_all:.0f}s)", flush=True)

    # 快速统计: 每域 top-20% 集合重叠
    for dom in T.DOMAINS:
        pass
    top = {}
    frac = 0.2
    for di, dom in enumerate(T.DOMAINS):
        sel = []
        for layer in range(n_layers):
            k = max(1, int(n_experts * frac))
            idx = np.argsort(-s[layer, :, di])[:k]
            sel.append(set(int(i) for i in idx))
        top[dom] = sel
    print("[overlap] per-domain top-20% Jaccard:", flush=True)
    for a in range(len(T.DOMAINS)):
        for b in range(a + 1, len(T.DOMAINS)):
            js = []
            for layer in range(n_layers):
                A, B = top[T.DOMAINS[a]][layer], top[T.DOMAINS[b]][layer]
                js.append(len(A & B) / max(len(A | B), 1))
            print(f"  {T.DOMAINS[a]}-{T.DOMAINS[b]}: {np.mean(js):.3f}", flush=True)


if __name__ == "__main__":
    main()
