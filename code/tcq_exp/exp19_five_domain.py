"""E19: 五域结构验证 —— 主模型 H1 矩阵与 gemma-4-26B-A4B 同构性。

在 Qwen3.6-35B-A3B (INT4 运行时引擎) 上复刻 b4 的五域协议:
  math / code / general / trans / summ, 24 calib + 16 eval, 前缀 32 token
  - 逐条路由直方图 [L,E] (top-8 权重按专家聚合, 逐层归一化)
  - H1: 10 个域对的 pooled top-20% Jaccard + lift + 200 次单元 bootstrap + Spearman
  - H3: 5 类逻辑回归
  - 与 b4_domain_pairs.json (gemma4) 的同构性: 10 对 Jaccard 的秩相关、
    "代码 vs NL" 与 "NL vs NL" 两组均值差 + 置换检验

    python exp19_five_domain.py --threads 8 --cpus 16-31
输出: results/five_domain.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F

import tcq_lib as T

DOMS = ("math", "code", "general", "trans", "summ")
MAX_LEN = 32
N_CALIB = 24
N_EVAL = 16
N_BOOT = 200
N_BOOT_H3 = 1000
TOP_K = 8
SEED = 71
RAND_BASE = 0.111111


def capture(eng, texts, batch_size=8):
    """批处理路由直方图 -> [n, L, E]。

    批内右侧 padding, 只对 attention_mask=1 的位置累加 (与逐条前向等价:
    因果注意力下真实 token 看不到后续 padding)。
    """
    model, tok = eng["model"], eng["tok"]
    tok.padding_side = "right"
    n_layers = len(model.model.language_model.layers)
    E = eng["store"].num_experts
    out = np.zeros((len(texts), n_layers, E), dtype=np.float64)

    for b0 in range(0, len(texts), batch_size):
        chunk = texts[b0:b0 + batch_size]
        B = len(chunk)
        acc = [torch.zeros(B, E, dtype=torch.float32, device="cuda")
               for _ in range(n_layers)]
        ctx = {"mask": None}

        def make_hook(li):
            def hook(m, inp, _):
                x = inp[0].detach()
                bb, ss, hh = x.shape
                logits = F.linear(x.reshape(-1, hh), m.gate.weight).float()
                topv, topi = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
                # 只用真实 token (mask=1) 的路由权重
                topv = topv.reshape(bb, ss * TOP_K) * \
                    ctx["mask"].repeat_interleave(TOP_K, dim=1).to(topv.dtype)
                topi = topi.reshape(bb, ss * TOP_K)
                acc[li].scatter_add_(1, topi, topv)
            return hook

        hooks = [model.model.language_model.layers[li].mlp.register_forward_hook(make_hook(li))
                 for li in range(n_layers)]
        try:
            with torch.inference_mode():
                enc = tok(chunk, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LEN,
                          add_special_tokens=False)
                ids = enc["input_ids"].cuda()
                ctx["mask"] = enc["attention_mask"].cuda()
                model(input_ids=ids, attention_mask=ctx["mask"], use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        stacked = torch.stack(acc).cpu().numpy()  # [L, B, E]
        for i in range(B):
            out[b0 + i] = stacked[:, i, :]
        print(f"  [route] {b0 + B}/{len(texts)}", flush=True)
    out /= np.maximum(out.sum(axis=2, keepdims=True), 1e-12)
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
    rx -= rx.mean()
    ry -= ry.mean()
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
        P = np.exp(lg)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / len(y)
        W -= lr * (Z.T @ G + l2 * W)
        b -= lr * G.sum(axis=0)
    return {"mu": mu, "sd": sd, "W": W, "b": b}


def macro_f1(y, p, k):
    f1 = []
    for c in range(k):
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1.append(2 * pr * rc / max(pr + rc, 1e-12))
    return float(np.mean(f1))


def h1_matrix(mean_h):
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
    return h1


def group_gap(h1):
    """代码(NL) 与 NL-NL 两组 Jaccard 均值差 + 置换检验"""
    code_pairs = [k for k in h1 if "code" in k]
    nl_pairs = [k for k in h1 if "code" not in k]
    jc = np.array([h1[k]["jaccard"] for k in code_pairs])
    jn = np.array([h1[k]["jaccard"] for k in nl_pairs])
    diff = float(jc.mean() - jn.mean())
    rng = np.random.default_rng(SEED + 1)
    allj = np.concatenate([jc, jn])
    nc = len(jc)
    cnt = 0
    for _ in range(20000):
        idx = rng.permutation(len(allj))
        d = allj[idx[:nc]].mean() - allj[idx[nc:]].mean()
        if abs(d) >= abs(diff):
            cnt += 1
    return {"code_pairs": code_pairs, "nl_pairs": nl_pairs,
            "code_mean_j": float(jc.mean()), "nl_mean_j": float(jn.mean()),
            "diff": diff, "perm_p": (cnt + 1) / 20001}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpus", default=None)
    ap.add_argument("--slots", type=int, default=512)
    a = ap.parse_args(argv)
    if a.cpus:
        from pin import pin_self
        print(f"[pin] {pin_self(cpus=a.cpus)}", flush=True)
    torch.set_num_threads(a.threads)
    t_all = time.time()
    eng = T.build_engine(slots=a.slots, cpu_threads=a.threads)

    feats, labels = {}, {}
    for d in DOMS:
        cal = [r["text"] for r in T.load_domain(d, "calib")[:N_CALIB]]
        ev = [r["text"] for r in T.load_domain(d, "eval")[:N_EVAL]]
        t0 = time.time()
        feats[(d, "calib")] = capture(eng, cal)
        feats[(d, "eval")] = capture(eng, ev)
        labels[(d, "calib")] = np.full(len(cal), DOMS.index(d))
        labels[(d, "eval")] = np.full(len(ev), DOMS.index(d))
        print(f"[{d}] calib {len(cal)} eval {len(ev)} ({time.time()-t0:.0f}s)", flush=True)

    L, E = eng["store"].num_layers, eng["store"].num_experts
    mean_h = {d: feats[(d, "calib")].mean(axis=0) for d in DOMS}
    h1 = h1_matrix(mean_h)

    Xc = np.vstack([feats[(d, "calib")].reshape(N_CALIB, -1) for d in DOMS])
    Xe = np.vstack([feats[(d, "eval")].reshape(N_EVAL, -1) for d in DOMS])
    yc = np.concatenate([labels[(d, "calib")] for d in DOMS])
    ye = np.concatenate([labels[(d, "eval")] for d in DOMS])
    lr = fit_logreg(Xc, yc, list(range(len(DOMS))))
    pe = np.argmax(((Xe - lr["mu"]) / lr["sd"]) @ lr["W"] + lr["b"], axis=1)
    corr = (pe == ye).astype(float)
    rng = np.random.default_rng(SEED)
    boot = np.array([corr[rng.integers(0, len(corr), len(corr))].mean()
                     for _ in range(N_BOOT_H3)])
    conf = [[int(((ye == t) & (pe == q)).sum()) for q in range(len(DOMS))]
            for t in range(len(DOMS))]
    h3 = {"acc": float(corr.mean()),
          "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
          "macro_f1": macro_f1(ye, pe, len(DOMS)), "confusion": conf}

    gap = group_gap(h1)
    out = {"model": "qwen3.6-35b-a3b", "L": L, "E": E, "K": TOP_K,
           "max_len": MAX_LEN, "domains": list(DOMS), "n_calib": N_CALIB,
           "n_eval": N_EVAL, "rand_base": RAND_BASE, "n_boot": N_BOOT,
           "h1": h1, "h3": h3, "gap": gap,
           "mean_hist": {d: mean_h[d].tolist() for d in DOMS},
           "elapsed_s": round(time.time() - t_all, 1)}

    # ---- 与 gemma4 (b4) 的同构性 ----
    gp = os.path.join(T.RES_DIR, "b4_domain_pairs.json")
    if os.path.exists(gp):
        with open(gp, encoding="utf-8") as f:
            g = json.load(f)
        pairs = sorted(set(h1) & set(g["h1"]))
        jm = np.array([h1[k]["jaccard"] for k in pairs])
        jg = np.array([g["h1"][k]["jaccard"] for k in pairs])
        out["compare"] = {
            "pairs": pairs,
            "jaccard_main": jm.tolist(), "jaccard_gemma": jg.tolist(),
            "pearson": float(np.corrcoef(jm, jg)[0, 1]),
            "rank_corr": rank_corr(jm, jg),
            "max_abs_delta": float(np.max(np.abs(jm - jg))),
            "mean_abs_delta": float(np.mean(np.abs(jm - jg))),
            "gap_main": gap, "gap_gemma": group_gap(g["h1"]),
            "same_order": bool(np.array_equal(np.argsort(-jm), np.argsort(-jg))),
        }
    p = os.path.join(T.RES_DIR, "five_domain.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[five_domain] saved {p}", flush=True)
    for k, v in h1.items():
        print(f"[h1] {k}: J={v['jaccard']:.3f} lift={v['lift']:.2f} rho={v['spearman']:.3f}",
              flush=True)
    print(f"[h3] acc={h3['acc']:.3f} f1={h3['macro_f1']:.3f}", flush=True)
    print(f"[gap] code={gap['code_mean_j']:.3f} nl={gap['nl_mean_j']:.3f} "
          f"diff={gap['diff']:.3f} p={gap['perm_p']:.4f}", flush=True)
    if "compare" in out:
        print(f"[compare] pearson={out['compare']['pearson']:.3f} "
              f"rho={out['compare']['rank_corr']:.3f} "
              f"max|dJ|={out['compare']['max_abs_delta']:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
