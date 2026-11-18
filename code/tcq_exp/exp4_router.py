"""实验4: 路由特征的任务域分类 (H3 验证)。
特征: 每层 MoE 门控 top-8 路由权重按专家聚合(256维) -> 逐层 L2 归一化 -> 拼接 (40*256)
数据: 三域校准集各 N_SEQ 条
分类器: (a) 最近类心(余弦) (b) 多项式逻辑回归, 二者均用留一交叉验证(LOO)
指标: 准确率 / 每类 F1 (macro)
输出: results/router.json
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

N_SEQ = 24
MAX_LEN = 96
BATCH = 8
TOPK = 8


def capture_route(eng, texts):
    """返回 [L, E] 路由权重直方图 (每层归一化)"""
    return T.capture_route_hist(eng, texts, max_len=MAX_LEN, batch_size=BATCH)


def featurize(h):
    """[L,E] -> [L*E] 逐层归一化后拼接"""
    return h.reshape(-1)


def loo_centroid(X, y):
    classes = sorted(set(y.tolist()))
    pred = []
    for i in range(len(y)):
        train = [j for j in range(len(y)) if j != i]
        best, bestc = -1e9, None
        for c in classes:
            idx = [j for j in train if y[j] == c]
            centroid = X[idx].mean(axis=0)
            sim = float(np.dot(X[i], centroid) /
                        (np.linalg.norm(X[i]) * np.linalg.norm(centroid) + 1e-12))
            if sim > best:
                best, bestc = sim, c
        pred.append(bestc)
    return np.array(pred)


def loo_logreg(X, y, iters=400, lr=0.5, l2=1e-2):
    classes = sorted(set(y.tolist()))
    Y = np.zeros((len(y), len(classes)))
    for i, c in enumerate(y):
        Y[i, classes.index(c)] = 1.0
    pred = []
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        Xtr, Ytr = X[tr], Y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Ztr = (Xtr - mu) / sd
        zi = (X[i] - mu) / sd
        W = np.zeros((Ztr.shape[1], len(classes)))
        b = np.zeros(len(classes))
        for _ in range(iters):
            lg = Ztr @ W + b
            lg -= lg.max(axis=1, keepdims=True)
            P = np.exp(lg)
            P /= P.sum(axis=1, keepdims=True)
            G = (P - Ytr) / len(tr)
            W -= lr * (Ztr.T @ G + l2 * W)
            b -= lr * G.sum(axis=0)
        pred.append(classes[int(np.argmax(zi @ W + b))])
    return np.array(pred), classes


def f1_macro(y, pred):
    classes = sorted(set(y.tolist()))
    f1s = []
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1s.append(2 * p * r / max(p + r, 1e-12))
    return float(np.mean(f1s)), [round(f, 3) for f in f1s]


def main():
    eng = T.build_engine()
    X, y = [], []
    t0 = time.time()
    for di, dom in enumerate(T.DOMAINS):
        rows = T.load_domain(dom, "calib")[:N_SEQ]
        texts = [r["text"] for r in rows]
        for t in texts:
            h = capture_route(eng, [t])
            X.append(featurize(h))
            y.append(di)
        print(f"[router] {dom}: {len(texts)} seq ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(X)
    y = np.array(y)

    out = {"n_seq": int(len(y)), "n_features": int(X.shape[1]),
           "domains": list(T.DOMAINS),
           "feature": "per-layer route-weight histogram (256 experts), L2-normalized per layer"}

    pc = loo_centroid(X, y)
    f1c, per_c = f1_macro(y, pc)
    out["centroid"] = {"acc": float((pc == y).mean()), "f1_macro": f1c,
                       "per_class_f1": per_c}
    print(f"[router] centroid LOO acc={out['centroid']['acc']:.3f} "
          f"F1={f1c:.3f} per={per_c}", flush=True)

    pl, classes = loo_logreg(X, y)
    f1l, per_l = f1_macro(y, pl)
    out["logreg"] = {"acc": float((pl == y).mean()), "f1_macro": f1l,
                     "per_class_f1": per_l}
    print(f"[router] logreg LOO acc={out['logreg']['acc']:.3f} "
          f"F1={f1l:.3f} per={per_l}", flush=True)

    p = os.path.join(T.RES_DIR, "router.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[router] saved {p}", flush=True)


if __name__ == "__main__":
    main()
