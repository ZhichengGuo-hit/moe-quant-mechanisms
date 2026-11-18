"""E14: N-token 前缀分类滑动曲线 —— "前缀越短越准"图。

对每个前缀长度 T ∈ {1,2,4,8,16,32,64}:
  1) 用 24 条/域 calib 的逐条路由直方图训练 (逻辑回归 / 类心), 在 96 条 eval 上测;
  2) 另做"只在 T=64 训练, 在全部 T 上测"的跨长度迁移曲线;
  3) 1000 次序列级 bootstrap 给出准确率 95% CI。
输出: results/prefix_curve.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T

LENGTHS = (1, 2, 4, 8, 16, 32, 64)
N_CALIB = 24
N_TEST = 32        # 每域 32 条, 共 96
N_BOOT = 1000
SEED = 17


def featurize(h):
    return h.reshape(-1)


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
    return {"mu": mu, "sd": sd, "W": W, "b": b, "classes": classes}


def pred_logreg(model, X):
    Z = (X - model["mu"]) / model["sd"]
    return np.argmax(Z @ model["W"] + model["b"], axis=1)


def macro_f1(y, p, k):
    f1s = []
    for c in range(k):
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(f1s))


def seq_hists(eng, texts, max_len):
    out = []
    for t in texts:
        out.append(T.capture_route_hist(eng, [t], max_len=max_len,
                                        batch_size=1, padding="longest").reshape(1, -1))
    return np.vstack(out)


def main():
    eng = T.build_engine()
    di = {d: i for i, d in enumerate(T.DOMAINS)}

    calib_texts, test_texts, y_cal, y_test = [], [], [], []
    for d in T.DOMAINS:
        calib_texts.append([r["text"] for r in T.load_domain(d, "calib")[:N_CALIB]])
        test_texts.append([r["text"] for r in T.load_domain(d, "eval64")[:N_TEST]])
        y_cal.extend([di[d]] * N_CALIB)
        y_test.extend([di[d]] * N_TEST)
    all_calib = sum(calib_texts, [])
    all_test = sum(test_texts, [])
    y_cal, y_test = np.array(y_cal), np.array(y_test)
    classes = list(range(len(T.DOMAINS)))
    k = len(classes)

    feats_cal, feats_test = {}, {}
    t_all = time.time()
    for L in LENGTHS:
        t0 = time.time()
        feats_cal[L] = seq_hists(eng, all_calib, L)
        feats_test[L] = seq_hists(eng, all_test, L)
        print(f"[route] T={L}: calib {feats_cal[L].shape} test {feats_test[L].shape} "
              f"({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)

    def boot_ci(correct):
        n = len(correct)
        vals = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            vals[b] = correct[idx].mean()
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    def evaluate(model, X, y):
        p = pred_logreg(model, X)
        correct = (p == y).astype(float)
        return {"acc": float(correct.mean()), "ci95": boot_ci(correct),
                "macro_f1": macro_f1(y, p, k),
                "confusion": [[int(((y == t) & (p == q)).sum()) for q in classes]
                              for t in classes]}

    # ---- 曲线 1: 同长度训练/测试 ----
    matched = {}
    models = {}
    for L in LENGTHS:
        models[L] = fit_logreg(feats_cal[L], y_cal, classes)
        matched[str(L)] = evaluate(models[L], feats_test[L], y_test)
        print(f"[matched T={L}] acc={matched[str(L)]['acc']:.3f} "
              f"f1={matched[str(L)]['macro_f1']:.3f} ci={matched[str(L)]['ci95']}", flush=True)

    # ---- 曲线 2: 只在 T=64 训练, 跨长度测试 ----
    model64 = models[64]
    transfer = {str(L): evaluate(model64, feats_test[L], y_test) for L in LENGTHS}
    print("[transfer from T=64] " + " ".join(
        f"{L}:{transfer[str(L)]['acc']:.3f}" for L in LENGTHS), flush=True)

    out = {"lengths": list(LENGTHS), "n_calib_per_domain": N_CALIB,
           "n_test_per_domain": N_TEST, "matched": matched,
           "transfer_from_64": transfer, "n_boot": N_BOOT,
           "elapsed_s": round(time.time() - t_all, 1)}
    p = os.path.join(T.RES_DIR, "prefix_curve.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[prefix_curve] saved {p} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
