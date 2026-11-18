"""C4: 补偿型量化 (AWQ-lite) 下重跑 E5/E6 的分配对照。

问题(§4.3 caveat): E6 得到 "Δerr ≈ 1.40 × err_INT4 且比例跨专家恒定", 故 m ≡ s;
该等价性只在 RTN(无补偿) 量化下被验证过。本实验换用补偿型量化 AWQ-lite
(逐通道激活缩放的等效变换: w' = w·s, 量化后除以 s, s = mean|x|^0.5 归一), 检验:
  (a) Δerr/err_INT4 比例是否仍恒定;
  (b) 每层 top-20% 集合在 {s, m, usage} 之间是否仍等价, 且跨量化器是否稳定;
  (c) 25% 高精度配额的残余误差预算(以 AWQ 下 Δerr 为真值)是否仍由 usage 主导。

设置: 全部 40 层 × 256 专家 × 3 域, 每域 6 条 calib (64 token); BF16 权重逐层读盘。
离线代理指标(不跑端到端 NLL): 端到端 AWQ 全量换装需重建容器与 INT3 store, 见报告局限。
输出: results/c4_awq.json (+ results/c4.log)
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

N_SEQ = 6
MAX_LEN = 64
GROUP = 128
ALPHA = 0.5
P_HIGH = 0.25
FRAC_RANK = 0.20
EPS = 1e-6
SEED = 97


def expert_out(x, w_gu, w_dn):
    gu = x @ w_gu.t()
    g, u = gu.chunk(2, dim=-1)
    return (F.silu(g) * u) @ w_dn.t()


def mid_act(x, w_gu):
    gu = x @ w_gu.t()
    g, u = gu.chunk(2, dim=-1)
    return F.silu(g) * u


def qdq(w, bits, group=GROUP):
    """对称 per-group 量化后再反量化 (RTN, 与 tcq_lib.quantize_bits 同规则)"""
    maxq = 2 ** (bits - 1) - 1
    ROWS, IN = w.shape
    wg = w.reshape(ROWS, IN // group, group)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / maxq).clamp(min=1e-8)
    q = torch.clamp(torch.round(wg / scale), -maxq, maxq)
    return (q * scale).reshape(ROWS, IN)


def awq_scale(act, alpha=ALPHA, eps=1e-5):
    s = act.abs().mean(dim=0).clamp(min=eps).pow(alpha)
    s = (s / s.mean()).clamp(min=1e-3)
    return s


def qdq_awq(w, s, bits):
    return qdq(w * s, bits) / s


def rel_err(yq, yb):
    return ((yq - yb).norm(dim=-1) / (yb.norm(dim=-1) + EPS)).numpy()


def top_set(score, frac=FRAC_RANK):
    k = max(1, int(round(len(score) * frac)))
    return set(np.argsort(-score)[:k].tolist())


def jacc(a, b):
    return len(a & b) / max(len(a | b), 1)


def main():
    torch.set_num_threads(16)
    t_all = time.time()
    eng = T.build_engine()
    wmap = T.load_weight_map()
    n_layers = len(eng["model"].model.language_model.layers)
    E = eng["store"].num_experts

    caps = {}
    for dom in T.DOMAINS:
        texts = [r["text"] for r in T.load_domain(dom, "calib")[:N_SEQ]]
        t0 = time.time()
        caps[dom] = T.capture_layer_stats(eng, texts, max_len=MAX_LEN, batch_size=6)
        print(f"[capture] {dom}: {len(texts)} seq {time.time()-t0:.0f}s", flush=True)

    D = len(T.DOMAINS)
    s4r = np.zeros((n_layers, E, D)); s4a = np.zeros((n_layers, E, D))
    s3r = np.zeros((n_layers, E, D)); s3a = np.zeros((n_layers, E, D))
    usage = np.zeros((n_layers, E, D))

    shard = {}

    def get(key):
        sh = wmap[key]
        if sh not in shard:
            shard[sh] = safetensors.safe_open(os.path.join(T.MOE_MODEL_PATH, sh), framework="pt")
        return shard[sh].get_tensor(key)

    for layer in range(n_layers):
        t0 = time.time()
        gu_all = get(f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj")
        dn_all = get(f"model.language_model.layers.{layer}.mlp.experts.down_proj")
        for di, dom in enumerate(T.DOMAINS):
            x, topi, topv = caps[dom][layer]["x"], caps[dom][layer]["topi"].long(), \
                caps[dom][layer]["topv"].float()
            # 域级 AWQ 缩放(对 get_gu 用 token 输入; 对 dn 用该域所有专家中位激活近似)
            s_in = awq_scale(x.float())
            for e in range(E):
                tok_idx, slot = torch.nonzero(topi == e, as_tuple=True)
                if tok_idx.numel() == 0:
                    continue
                wts = topv[tok_idx, slot].numpy()
                xe = x[tok_idx].float()
                w_gu = gu_all[e].float(); w_dn = dn_all[e].float()
                yb = expert_out(xe, w_gu, w_dn)
                y4r = expert_out(xe, qdq(w_gu, 4), qdq(w_dn, 4))
                y3r = expert_out(xe, qdq(w_gu, 3), qdq(w_dn, 3))
                hb = mid_act(xe, w_gu)
                s_dn = awq_scale(hb)
                y4a = expert_out(xe, qdq_awq(w_gu, s_in, 4), qdq_awq(w_dn, s_dn, 4))
                y3a = expert_out(xe, qdq_awq(w_gu, s_in, 3), qdq_awq(w_dn, s_dn, 3))
                s4r[layer, e, di] += float((wts * rel_err(y4r, yb)).sum())
                s3r[layer, e, di] += float((wts * rel_err(y3r, yb)).sum())
                s4a[layer, e, di] += float((wts * rel_err(y4a, yb)).sum())
                s3a[layer, e, di] += float((wts * rel_err(y3a, yb)).sum())
                usage[layer, e, di] += float(wts.sum())
        if layer % 4 == 0 or layer == n_layers - 1:
            print(f"[c4] layer {layer} {time.time()-t0:.0f}s", flush=True)

    d_r = s3r - s4r
    d_a = s3a - s4a
    ratio_r = np.where(s4r > 1e-9, d_r / np.maximum(s4r, 1e-9), np.nan)
    ratio_a = np.where(s4a > 1e-9, d_a / np.maximum(s4a, 1e-9), np.nan)

    def stats(v):
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                "std": float(v.std()), "n": int(v.size)}

    out_ratio = {"rtn": stats(ratio_r), "awq": stats(ratio_a)}

    # (b) 每层 top-20% 集合一致性
    scores = {"s_rtn": s4r, "s_awq": s4a, "m_rtn": usage * d_r, "m_awq": usage * d_a,
              "usage": usage}
    pairs = [("s_rtn", "s_awq"), ("s_awq", "m_awq"), ("m_awq", "usage"),
             ("s_rtn", "m_rtn"), ("s_rtn", "usage"), ("m_rtn", "usage")]
    jac = {}
    for a, b in pairs:
        vals = []
        for di, dom in enumerate(T.DOMAINS):
            for l in range(n_layers):
                vals.append(jacc(top_set(scores[a][l, :, di]), top_set(scores[b][l, :, di])))
        jac[f"{a}-{b}"] = {"mean": float(np.mean(vals)), "min": float(np.min(vals))}

    # (c) 25% 配额下的残余误差预算 (真值 = AWQ 下的 Δerr)
    def residual(score, truth):
        tot, rsd = 0.0, 0.0
        for di in range(D):
            for l in range(n_layers):
                hi = top_set(score[l, :, di], P_HIGH)
                w = usage[l, :, di]; t = truth[l, :, di]
                tot += float((w * t).sum())
                rsd += float((w * t[[e for e in range(E) if e not in hi]]).sum())
        return rsd / max(tot, 1e-12)

    budget = {k: residual(v, d_a) for k, v in scores.items()}
    budget["oracle_awq"] = residual(usage * d_a, d_a)
    budget_rtn = {k: residual(v, d_r) for k, v in scores.items()}
    budget_rtn["oracle_rtn"] = residual(usage * d_r, d_r)

    out = {"n_layers": n_layers, "n_experts": E, "n_seq": N_SEQ, "max_len": MAX_LEN,
           "alpha": ALPHA, "frac_rank": FRAC_RANK, "p_high": P_HIGH,
           "domains": list(T.DOMAINS),
           "ratio_delta_over_err4": out_ratio,
           "rank_jaccard": jac,
           "residual_budget_awq": budget,
           "residual_budget_rtn": budget_rtn,
           "s4_rtn": s4r.tolist(), "s4_awq": s4a.tolist(),
           "d_rtn": d_r.tolist(), "d_awq": d_a.tolist(), "usage": usage.tolist(),
           "elapsed_s": round(time.time() - t_all, 1)}
    p = os.path.join(T.RES_DIR, "c4_awq.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[ratio] rtn {out_ratio['rtn']['mean']:.3f}±{out_ratio['rtn']['std']:.3f} | "
          f"awq {out_ratio['awq']['mean']:.3f}±{out_ratio['awq']['std']:.3f}", flush=True)
    print("[jac] " + " | ".join(f"{k}:{v['mean']:.3f}" for k, v in jac.items()), flush=True)
    print("[budget awq] " + " | ".join(f"{k}:{v:.3f}" for k, v in budget.items()), flush=True)
    print(f"[c4] saved {p} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
