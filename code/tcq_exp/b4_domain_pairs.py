"""B4: gemma-4-26B-A4B 新增域对 (code vs 翻译 / 摘要) 的 H1 分离度。

域: math / code / general / trans / summ (24 calib + 16 eval, 前缀 32 token)
输出全部 10 个域对的 pooled top-20% Jaccard + lift + 200 次单元 bootstrap CI + Spearman,
以及 5 类 H3 准确率/混淆矩阵。
输出: results/b4_domain_pairs.json
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

import tcq_lib as T

MODEL = r"E:\CUDA11.8\model\gemma-4-26B-A4B"
DOMS = ("math", "code", "general", "trans", "summ")
MAX_LEN = 32
N_CALIB = 24
N_EVAL = 16
N_BOOT = 200
N_BOOT_H3 = 1000
SEED = 71
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
        if router is not None and id(router) not in seen:
            seen.add(id(router))
            out.append((name, mod, router))
    return out


def capture(eng, texts, max_len):
    model, tok = eng["model"], eng["tok"]
    blocks = eng["blocks"]
    L, E, K = eng["n_layers"], eng["n_experts"], eng["top_k"]
    out = np.zeros((len(texts), L, E), dtype=np.float64)
    for ti, text in enumerate(texts):
        hists = [np.zeros(E, dtype=np.float64) for _ in range(L)]
        hooks = []

        def make_hook(li):
            def hook(m, inp, output):
                if isinstance(output, (tuple, list)):
                    idx_t, w_t = output[2], output[1]
                    inds = idx_t.reshape(-1).detach().cpu().numpy()
                    vals = w_t.reshape(-1).detach().double().cpu().numpy()
                else:
                    topv, topi = torch.topk(torch.softmax(output.float(), dim=-1), K, dim=-1)
                    inds = topi.reshape(-1).cpu().numpy()
                    vals = topv.reshape(-1).double().cpu().numpy()
                np.add.at(hists[li], inds, vals)
            return hook

        for li, (_, _, router) in enumerate(blocks):
            hooks.append(router.register_forward_hook(make_hook(li)))
        try:
            with torch.inference_mode():
                enc = tok([text], return_tensors="pt", padding="longest",
                          truncation=True, max_length=max_len, add_special_tokens=False)
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                      use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        h = np.stack(hists)
        h /= np.maximum(h.sum(axis=1, keepdims=True), 1e-12)
        out[ti] = h
        if (ti + 1) % 8 == 0:
            print(f"  [route] {ti+1}/{len(texts)}", flush=True)
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
        W -= lr * (Z.T @ G + l2 * W)
        b -= lr * G.sum(axis=0)
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
    torch.set_num_threads(16)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {MODEL}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=True)
    except Exception as e:
        from transformers import AutoModelForImageTextToText
        print(f"[load] fallback ImageTextToText ({(repr(e)[:60])})", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()
    blocks = find_moe_blocks(model)
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    top_k = 8
    for key in ("num_experts_per_tok", "num_experts_per_token", "top_k_experts", "moe_top_k"):
        v = getattr(cfg, key, None)
        if v:
            top_k = int(v); break
    n_exp = 0
    for key in ("num_experts", "num_local_experts", "moe_num_experts"):
        v = getattr(cfg, key, None)
        if v:
            n_exp = int(v); break
    if n_exp == 0:
        n_exp = int(blocks[0][2].weight.shape[0])
    eng = {"model": model, "tok": tok, "blocks": blocks,
           "n_layers": len(blocks), "n_experts": n_exp, "top_k": top_k}
    print(f"[load] {len(blocks)} layers, E={n_exp}, K={top_k}, {time.time()-t0:.0f}s",
          flush=True)

    feats, labels = {}, {}
    for d in DOMS:
        di = DOMS.index(d)
        cal = [r["text"] for r in T.load_domain(d, "calib")[:N_CALIB]]
        ev = [r["text"] for r in T.load_domain(d, "eval")[:N_EVAL]]
        t1 = time.time()
        feats[(d, "calib")] = capture(eng, cal, MAX_LEN)
        feats[(d, "eval")] = capture(eng, ev, MAX_LEN)
        labels[(d, "calib")] = np.full(len(cal), di)
        labels[(d, "eval")] = np.full(len(ev), di)
        print(f"[{d}] calib {len(cal)} eval {len(ev)} ({time.time()-t1:.0f}s)", flush=True)

    L, E = eng["n_layers"], eng["n_experts"]
    mean_h = {d: feats[(d, "calib")].mean(axis=0) for d in DOMS}
    rng = np.random.default_rng(SEED)
    h1 = {}
    for i, a in enumerate(DOMS):
        for b in DOMS[i + 1:]:
            ma, mb = top_mask(mean_h[a]), top_mask(mean_h[b])
            va, vb = mean_h[a].reshape(-1), mean_h[b].reshape(-1)
            fa, fb = ma.reshape(-1), mb.reshape(-1)
            n = fa.size
            j = float((fa & fb).sum()) / max(float((fa | fb).sum()), 1)
            r = rank_corr(va, vb)
            js, rs = np.empty(N_BOOT), np.empty(N_BOOT)
            for bi in range(N_BOOT):
                ii = rng.integers(0, n, size=n)
                js[bi] = float((fa[ii] & fb[ii]).sum()) / max(float((fa[ii] | fb[ii]).sum()), 1)
                rs[bi] = rank_corr(va[ii], vb[ii])
            h1[f"{a}-{b}"] = {"jaccard": j, "lift": j / RAND_BASE,
                              "jaccard_ci95": [float(np.percentile(js, 2.5)),
                                               float(np.percentile(js, 97.5))],
                              "spearman": r,
                              "spearman_ci95": [float(np.percentile(rs, 2.5)),
                                                float(np.percentile(rs, 97.5))]}

    Xc = np.vstack([feats[(d, "calib")].reshape(N_CALIB, -1) for d in DOMS])
    Xe = np.vstack([feats[(d, "eval")].reshape(N_EVAL, -1) for d in DOMS])
    yc = np.concatenate([labels[(d, "calib")] for d in DOMS])
    ye = np.concatenate([labels[(d, "eval")] for d in DOMS])
    lr = fit_logreg(Xc, yc, list(range(len(DOMS))))
    pe = np.argmax(((Xe - lr["mu"]) / lr["sd"]) @ lr["W"] + lr["b"], axis=1)
    corr = (pe == ye).astype(float)
    boot = np.array([corr[rng.integers(0, len(corr), len(corr))].mean()
                     for _ in range(N_BOOT_H3)])
    conf = [[int(((ye == t) & (pe == q)).sum()) for q in range(len(DOMS))]
            for t in range(len(DOMS))]
    h3 = {"acc": float(corr.mean()),
          "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
          "macro_f1": macro_f1(ye, pe, len(DOMS)), "confusion": conf}

    out = {"model": "gemma-4-26b-a4b", "L": L, "E": E, "K": eng["top_k"],
           "max_len": MAX_LEN, "domains": list(DOMS), "n_calib": N_CALIB,
           "n_eval": N_EVAL, "rand_base": RAND_BASE, "n_boot": N_BOOT,
           "h1": h1, "h3": h3,
           "mean_hist": {d: mean_h[d].tolist() for d in DOMS}}
    p = os.path.join(T.RES_DIR, "b4_domain_pairs.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[b4] saved {p}", flush=True)
    for k, v in h1.items():
        print(f"[h1] {k}: J={v['jaccard']:.3f} lift={v['lift']:.2f} ρ={v['spearman']:.3f}",
              flush=True)
    print(f"[h3] acc={h3['acc']:.3f} f1={h3['macro_f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
