"""C2: 同源题库 N-token 分类复测 (关闭 E14 的"数据源格式" caveat)。

两组同源二分类:
  A) Alpaca 数学题 vs 非数学散文 (同一语料来源)
  B) HumanEval 代码 vs 同一题的题面描述 (同一题面来源)
每域 24 条 calib 训练, 32 条 test; T ∈ {1,2,4,8,16,32}; 1000 次序列 bootstrap。

实现: 每条文本只做一次 32-token 前向(右填充), 因果注意力保证 position t 只依赖前缀,
因此一次前向即可同时得到全部前缀长度的路由直方图 (等价于对每个 T 单独截断重跑,
但省掉 6 倍前向)。
输出: results/c2_same_source.json
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

import tcq_lib as T

LENGTHS = (1, 2, 4, 8, 16, 32)
MAX_LEN = 32
PAIRS = {
    "alpaca_math_vs_prose": (("samemath", "prose"), "数学题 vs 散文(Alpaca)"),
    "humaneval_code_vs_spec": (("hecode", "codespec"), "代码 vs 题面描述(HumanEval)"),
}
N_BOOT = 1000
SEED = 53


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


def prefix_hists(eng, texts, lengths=LENGTHS, max_len=MAX_LEN, batch_size=8):
    """一次 max_len 右填充前向 -> {L: [n, layers, E]} 前缀路由直方图"""
    model, tok = eng["model"], eng["tok"]
    n_layers = len(model.model.language_model.layers)
    E = eng["store"].num_experts
    n = len(texts)
    out = {L: np.zeros((n, n_layers, E), dtype=np.float64) for L in lengths}
    buf = {}
    hooks = []

    def make_hook(li, mod):
        def hook(m, inp, out_):
            x = inp[0].detach()
            B, S, H = x.shape
            flat = x.reshape(-1, H)
            probs = torch.softmax(F.linear(flat, m.gate.weight).float(), dim=-1)
            tv, ti = torch.topk(probs, T.TOPK, dim=-1)
            buf[li] = (ti.cpu().numpy().reshape(B, S, -1),
                       tv.double().cpu().numpy().reshape(B, S, -1))
        return hook

    for li in range(n_layers):
        mod = model.model.language_model.layers[li].mlp
        hooks.append(mod.register_forward_hook(make_hook(li, mod)))
    try:
        with torch.no_grad():
            for b0 in range(0, n, batch_size):
                chunk = texts[b0:b0 + batch_size]
                ids, mask = T._batch_encode(tok, chunk, max_len, "longest")
                lens = mask.sum(dim=1).tolist()
                model(input_ids=ids, attention_mask=mask, use_cache=False)
                for li in range(n_layers):
                    idx, w = buf[li]
                    for bi in range(idx.shape[0]):
                        ni = b0 + bi
                        for L in lengths:
                            k = min(L, int(lens[bi]))
                            np.add.at(out[L][ni, li], idx[bi, :k].reshape(-1),
                                      w[bi, :k].reshape(-1))
    finally:
        for h in hooks:
            h.remove()
    for L in lengths:
        out[L] /= np.maximum(out[L].sum(axis=2, keepdims=True), 1e-12)
    return out


def load(name, split):
    return [r["text"] for r in T.load_domain(name, split)]


def main():
    eng = T.build_engine()
    out = {"lengths": list(LENGTHS), "max_len": MAX_LEN, "n_boot": N_BOOT, "pairs": {}}
    t_all = time.time()
    for tag, ((ga, gb), desc) in PAIRS.items():
        ra_c, rb_c = load(ga, "calib"), load(gb, "calib")
        ra_t, rb_t = load(ga, "test"), load(gb, "test")
        print(f"[{tag}] calib {len(ra_c)}+{len(rb_c)} test {len(ra_t)}+{len(rb_t)}", flush=True)
        texts = ra_c + rb_c + ra_t + rb_t
        n1 = len(ra_c); n2 = n1 + len(rb_c); n3 = n2 + len(ra_t)
        t0 = time.time()
        hist = prefix_hists(eng, texts)
        print(f"  [route] {len(texts)} 条 / 一次前向 {time.time()-t0:.0f}s", flush=True)
        y_cal = np.array([0] * len(ra_c) + [1] * len(rb_c))
        y_test = np.array([0] * len(ra_t) + [1] * len(rb_t))
        res = {}
        rng = np.random.default_rng(SEED)
        for L in LENGTHS:
            h = hist[L]
            Xc = np.vstack([h[:n1].reshape(n1, -1), h[n1:n2].reshape(n2 - n1, -1)])
            Xt = np.vstack([h[n2:n3].reshape(n3 - n2, -1), h[n3:].reshape(len(rb_t), -1)])
            lr = fit_logreg(Xc, y_cal, [0, 1])
            pe = np.argmax(((Xt - lr["mu"]) / lr["sd"]) @ lr["W"] + lr["b"], axis=1)
            corr = (pe == y_test).astype(float)
            boot = np.array([corr[rng.integers(0, len(corr), len(corr))].mean()
                             for _ in range(N_BOOT)])
            conf = [[int(((y_test == t) & (pe == q)).sum()) for q in (0, 1)] for t in (0, 1)]
            res[str(L)] = {"acc": float(corr.mean()),
                           "ci95": [float(np.percentile(boot, 2.5)),
                                    float(np.percentile(boot, 97.5))],
                           "macro_f1": macro_f1(y_test, pe, 2), "confusion": conf}
            print(f"[{tag} T={L}] acc={res[str(L)]['acc']:.3f} "
                  f"ci={res[str(L)]['ci95']} f1={res[str(L)]['macro_f1']:.3f}", flush=True)
        out["pairs"][tag] = {"desc": desc, "groups": [ga, gb],
                             "n_calib": [len(ra_c), len(rb_c)],
                             "n_test": [len(ra_t), len(rb_t)], "matched": res}

    out["elapsed_s"] = round(time.time() - t_all, 1)
    p = os.path.join(T.RES_DIR, "c2_same_source.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[c2] saved {p}", flush=True)


if __name__ == "__main__":
    main()
