"""B1: Qwen3-30B-A3B 重放路由, 离线限制 top-k ∈ {2,4,8,16,32} (K/N = 1.6%..25%)。

一次 trace 捕获每层 top-32 路由概率, 离线对每个 k:
  - 逐层重归一化直方图 -> H1 (pooled top-20% Jaccard / lift / Spearman + 200 单元 bootstrap)
  - H3 (calib 训练 / eval 测试 逻辑回归, 1000 序列 bootstrap)
  - 分配代理: 每域 score = usage -> 每层 top-25% 高精度集合; math/code 选择重叠
输出: results/b1_topk.json (+ results/b1_trace.npz 供 B5 复用)
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tcq_lib as T

MODEL = r"E:\CUDA11.8\model\Qwen3-30B-A3B"
MAX_LEN = 32
TOPK_MAX = 32
N_CALIB = 24
N_EVAL = 16
KS = (2, 4, 8, 16, 32)
THREADS = 16
N_BOOT_H1 = 200
N_BOOT_H3 = 1000
SEED = 41
RAND_BASE = 0.111111


def find_moe_blocks(model):
    out, seen = [], set()
    for name, mod in model.named_modules():
        router = None
        if name.endswith(".mlp"):
            for attr in ("gate", "router"):
                cand = getattr(mod, attr, None)
                if isinstance(cand, nn.Module):
                    router = cand
                    break
        if router is None:
            r = getattr(mod, "router", None)
            if isinstance(r, nn.Module) and isinstance(getattr(mod, "experts", None), nn.Module):
                router = r
        if router is not None and id(router) not in seen and hasattr(router, "weight"):
            seen.add(id(router))
            out.append((name, mod, router))
    return out


@torch.no_grad()
def capture(eng, texts, ks=KS):
    """-> {k: [n,L,E] float32}, 每 token 取 top-k 并重归一化后累加到逐层直方图"""
    model, tok = eng["model"], eng["tok"]
    blocks = eng["blocks"]
    L = len(blocks)
    E = eng["n_experts"]
    n = len(texts)
    out = {k: np.zeros((n, L, E), dtype=np.float32) for k in ks}
    for ti, text in enumerate(texts):
        buf = [None] * L
        hooks = []

        def make_hook(li, router):
            def hook(m, inp, out_):
                x = inp[0].detach()
                flat = x.reshape(-1, x.shape[-1]).float()
                probs = torch.softmax(F.linear(flat, router.weight.float()), dim=-1)
                tv, ti_ = torch.topk(probs, TOPK_MAX, dim=-1)
                buf[li] = (ti_.cpu().numpy(), tv.cpu().numpy())
            return hook

        for li, (_, mod, router) in enumerate(blocks):
            hooks.append(mod.register_forward_hook(make_hook(li, router)))
        try:
            enc = tok([text], return_tensors="pt", padding="longest",
                      truncation=True, max_length=MAX_LEN, add_special_tokens=False)
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                  use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for li in range(L):
            inds, vals = buf[li]                    # [T,32], [T,32]
            T = inds.shape[0]
            for k in ks:
                kk = min(k, TOPK_MAX)
                w = vals[:, :kk] / np.maximum(vals[:, :kk].sum(axis=1, keepdims=True), 1e-12)
                for t in range(T):
                    np.add.at(out[k][ti, li], inds[t, :kk], w[t])
        if (ti + 1) % 8 == 0:
            print(f"  [trace] {ti+1}/{n}", flush=True)
    for k in ks:
        out[k] /= np.maximum(out[k].sum(axis=2, keepdims=True), 1e-12)
    return out


def top_mask(h, frac=0.2):
    L, E = h.shape
    k = max(1, int(round(E * frac)))
    m = np.zeros((L, E), dtype=bool)
    for l in range(L):
        m[l, np.argsort(-h[l])[:k]] = True
    return m


def rank_corr(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def fit_logreg(X, y, classes, iters=400, lr=0.5, l2=1e-2):
    Y = np.zeros((len(y), len(classes)))
    for i, c in enumerate(y):
        Y[i, classes.index(int(c))] = 1.0
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    W = np.zeros((Z.shape[1], len(classes)))
    b = np.zeros(len(classes))
    for _ in range(iters):
        lg = Z @ W + b
        lg -= lg.max(axis=1, keepdims=True)
        P = np.exp(lg); P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / len(y)
        W -= lr * (Z.T @ G + l2 * W); b -= lr * G.sum(axis=0)
    return {"mu": mu, "sd": sd, "W": W, "b": b, "classes": classes}


def macro_f1(y, p, k):
    f1 = []
    for c in range(k):
        tp = int(((y == c) & (p == c)).sum()); fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1.append(2 * pr * rc / max(pr + rc, 1e-12))
    return float(np.mean(f1))


def main():
    torch.set_num_threads(THREADS)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {MODEL}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()
    blocks = find_moe_blocks(model)
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    E = int(getattr(cfg, "num_experts", 128))
    eng = {"model": model, "tok": tok, "blocks": blocks, "n_experts": E}
    print(f"[load] {len(blocks)} layers, E={E}, {time.time()-t0:.0f}s", flush=True)

    di = {d: i for i, d in enumerate(T.DOMAINS)}
    texts, labels, splits = [], [], []
    for d in T.DOMAINS:
        for split, n in (("calib", N_CALIB), ("eval", N_EVAL)):
            rows = T.load_domain(d, split)[:n]
            texts.extend(r["text"] for r in rows)
            labels.extend([di[d]] * len(rows))
            splits.extend([0 if split == "calib" else 1] * len(rows))
    print(f"[data] {len(texts)} texts", flush=True)
    t1 = time.time()
    hists = capture(eng, texts)
    print(f"[trace] done L={len(blocks)} ({time.time()-t1:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(T.RES_DIR, "b1_trace.npz"),
                        **{f"h{k}": hists[k] for k in KS},
                        y=np.array(labels), split=np.array(splits),
                        max_len=MAX_LEN, model="Qwen3-30B-A3B")
    del model
    import gc; gc.collect()

    y = np.array(labels); sp = np.array(splits)
    rng = np.random.default_rng(SEED)
    classes = list(range(len(T.DOMAINS)))
    L = len(blocks)
    res = {}
    for k in KS:
        h = hists[k].astype(np.float64)
        hc, he = h[sp == 0], h[sp == 1]
        yc, ye = y[sp == 0], y[sp == 1]
        mean_h = {d: hc[yc == di[d]].mean(axis=0) for d in T.DOMAINS}
        # ---- H1 ----
        h1 = {}
        for i, a in enumerate(T.DOMAINS):
            for b in T.DOMAINS[i + 1:]:
                ma, mb = top_mask(mean_h[a]), top_mask(mean_h[b])
                va, vb = mean_h[a].reshape(-1), mean_h[b].reshape(-1)
                fa, fb = ma.reshape(-1), mb.reshape(-1)
                n = fa.size
                j0 = float((fa & fb).sum()) / max(float((fa | fb).sum()), 1)
                r0 = rank_corr(va, vb)
                js, rs = np.empty(N_BOOT_H1), np.empty(N_BOOT_H1)
                for bi in range(N_BOOT_H1):
                    ii = rng.integers(0, n, size=n)
                    js[bi] = float((fa[ii] & fb[ii]).sum()) / max(float((fa[ii] | fb[ii]).sum()), 1)
                    rs[bi] = rank_corr(va[ii], vb[ii])
                h1[f"{a}-{b}"] = {
                    "jaccard": j0, "lift": j0 / RAND_BASE,
                    "jaccard_ci95": [float(np.percentile(js, 2.5)), float(np.percentile(js, 97.5))],
                    "spearman": r0,
                    "spearman_ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]}
        # ---- H3 ----
        Xc = hc.reshape(len(hc), -1); Xe = he.reshape(len(he), -1)
        lr = fit_logreg(Xc, yc, classes)
        pe = np.argmax(((Xe - lr["mu"]) / lr["sd"]) @ lr["W"] + lr["b"], axis=1)
        corr = (pe == ye).astype(float)
        boot = np.array([corr[rng.integers(0, len(corr), len(corr))].mean()
                         for _ in range(N_BOOT_H3)])
        conf = [[int(((ye == t) & (pe == q)).sum()) for q in classes] for t in classes]
        h3 = {"acc": float(corr.mean()),
              "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
              "macro_f1": macro_f1(ye, pe, len(classes)), "confusion": conf}
        # ---- 分配代理: 每域 usage -> 每层 top-25% ----
        alloc = {}
        sel = {d: top_mask(mean_h[d], 0.25) for d in T.DOMAINS}
        for i, a in enumerate(T.DOMAINS):
            for b in T.DOMAINS[i + 1:]:
                A, B = sel[a], sel[b]
                inter = float((A & B).sum()) / max(float((A | B).sum()), 1)
                # 域 b 的路由质量被 a 的高精度集合覆盖的比例
                cover = float(mean_h[b][A].sum()) / max(float(mean_h[b].sum()), 1e-12)
                alloc[f"{a}-{b}"] = {"selection_jaccard": inter,
                                     f"cover_{b}_by_{a}": cover,
                                     f"cover_{a}_by_{b}": float(mean_h[a][B].sum()) /
                                     max(float(mean_h[a].sum()), 1e-12)}
        res[str(k)] = {"k_over_N": k / E, "h1": h1, "h3": h3, "alloc": alloc}
        print(f"[k={k} K/N={k/E:.3f}] " + " | ".join(
            f"{p}: J={v['jaccard']:.3f}(lift {v['lift']:.2f}) ρ={v['spearman']:.3f}"
            for p, v in h1.items()) +
            f" | H3 acc={h3['acc']:.3f} f1={h3['macro_f1']:.3f} "
            f"| allocJ(math-code)={alloc['math-code']['selection_jaccard']:.3f}", flush=True)

    out = {"model": "Qwen3-30B-A3B", "L": L, "E": E, "ks": list(KS),
           "max_len": MAX_LEN, "n_calib": N_CALIB, "n_eval": N_EVAL,
           "domains": list(T.DOMAINS), "rand_base": RAND_BASE,
           "n_boot_h1": N_BOOT_H1, "n_boot_h3": N_BOOT_H3, "results": res}
    p = os.path.join(T.RES_DIR, "b1_topk.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[b1] saved {p}", flush=True)


if __name__ == "__main__":
    main()
