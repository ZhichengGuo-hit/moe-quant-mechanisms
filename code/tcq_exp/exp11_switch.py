"""E11: 在线切换端到端 —— 用前 8 token 路由识别域, 换装对应高精度补丁。

流程:
  1) 24 条/域 calib 的 8-token 路由直方图 -> 类心 / 逻辑回归分类器
  2) 三域 eval64 前 32 条 (96 条) 用 8-token 路由分类, 按预测域分桶
  3) 对照: base_int4 / global / oracle(真实域) / predicted(预测域)
  4) 记录每次补丁安装耗时与写入字节
输出: results/switch.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T
from moe_lib import Int4Store

P_HIGH = 0.25
N_EVAL_SEQ = 32
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")


def select_per_layer(score, frac):
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        high[l, np.argsort(-score[l])[:k]] = True
    return high


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
    return np.array([model["classes"][int(i)] for i in
                     np.argmax(Z @ model["W"] + model["b"], axis=1)])


def fit_centroid(X, y, classes):
    return {c: X[y == c].mean(0) for c in classes}


def pred_centroid(cents, X):
    classes = sorted(cents)
    out = []
    for x in X:
        sims = [np.dot(x, cents[c]) / (np.linalg.norm(x) * np.linalg.norm(cents[c]) + 1e-12)
                for c in classes]
        out.append(classes[int(np.argmax(sims))])
    return np.array(out)


def main():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])
    doms = sm["domains"]
    di = {d: i for i, d in enumerate(doms)}

    eng = T.build_engine()
    tok = eng["tok"]
    patcher = T.ExpertPatcher(eng["store"])
    store3 = Int4Store(T.STORE_INT3, mmap_mode="r")
    v3 = {l: store3.layer_views(l) for l in range(store3.num_layers)}
    n_layers, n_experts = store3.num_layers, store3.num_experts
    expert_bytes = int(store3.expert_bytes())

    high = {d: select_per_layer(s[:, :, di[d]], P_HIGH) for d in doms}
    high["global"] = select_per_layer(s.sum(-1), P_HIGH)

    def apply_patch(name):
        t0 = time.perf_counter()
        n = 0
        for l in range(n_layers):
            for e in np.nonzero(~high[name][l])[0]:
                patcher.set_expert(l, int(e),
                                   {p: v3[l][p][int(e)] for p in LAYER_PARTS})
                n += 1
        dt = time.perf_counter() - t0
        return dt, n * expert_bytes

    def restore_patch(name):
        for l in range(n_layers):
            for e in np.nonzero(~high[name][l])[0]:
                patcher.restore_expert(l, int(e))

    def seq_hists(texts, max_len):
        """逐条文本的路由直方图 [n_texts, L*E]"""
        out = []
        for t in texts:
            out.append(T.capture_route_hist(eng, [t], max_len=max_len,
                                            batch_size=1).reshape(1, -1))
        return np.vstack(out)

    # ---------- 1) 分类器训练 ----------
    Xc, yc = [], []
    for d in T.DOMAINS:
        rows = T.load_domain(d, "calib")[:24]
        Xc.append(seq_hists([r["text"] for r in rows], 8))
        yc.extend([di[d]] * 24)
        print(f"[calib] {d}: 24 seq", flush=True)
    Xc = np.vstack(Xc)
    yc = np.array(yc)
    classes = list(range(len(doms)))
    lr = fit_logreg(Xc, yc, classes)
    cents = fit_centroid(Xc, yc, classes)

    # ---------- 2) eval 前 32 条: 8-token 与 96-token 路由 ----------
    evalsets = {}
    X8, X96, ytrue = [], [], []
    for d in T.DOMAINS:
        rows = T.load_domain(d, "eval64")[:N_EVAL_SEQ]
        texts = [r["text"] for r in rows]
        evalsets[d] = texts
        X8.append(seq_hists(texts, 8))
        X96.append(seq_hists(texts, 96))
        ytrue.extend([di[d]] * len(texts))
        print(f"[route] {d}: {len(texts)} seq (8/96 token)", flush=True)
    X8 = np.vstack(X8)
    X96 = np.vstack(X96)
    ytrue = np.array(ytrue)

    cls = {
        "logreg_acc_8": float((pred_logreg(lr, X8) == ytrue).mean()),
        "centroid_acc_8": float((pred_centroid(cents, X8) == ytrue).mean()),
        "logreg_acc_96": float((pred_logreg(lr, X96) == ytrue).mean()),
        "centroid_acc_96": float((pred_centroid(cents, X96) == ytrue).mean()),
    }
    pred = pred_logreg(lr, X8)
    conf = np.zeros((len(doms), len(doms)), dtype=int)
    for t, p in zip(ytrue, pred):
        conf[int(t), int(p)] += 1
    cls["confusion_8"] = conf.tolist()
    print(f"[cls] {cls}", flush=True)

    # ---------- 3) 配置评测 ----------
    out = {"n_eval_per_domain": N_EVAL_SEQ, "classifier": cls,
           "nll": {}, "per_seq": {}, "switch": {}}

    def eval_domains(tag):
        res, seqs = {}, {}
        for d in T.DOMAINS:
            tot, n, per, ps = T.eval_nll(eng["model"], tok, evalsets[d],
                                         max_len=64, batch_size=8,
                                         return_per_seq=True)
            res[d] = per
            seqs[d] = ps
        out["nll"][tag] = res
        out["per_seq"][tag] = seqs
        return res

    t0 = time.time()
    out["nll"]["base"] = {}
    out["per_seq"]["base"] = {}
    eval_domains("base")
    print(f"[base] " + " ".join(f"{d}:{out['nll']['base'][d]:.4f}"
                                for d in T.DOMAINS), flush=True)

    dt, nb = apply_patch("global")
    out["switch"]["global"] = {"install_s": round(dt, 2), "bytes": nb}
    eval_domains("global")
    print(f"[global] " + " ".join(f"{d}:{out['nll']['global'][d]:.4f}"
                                  for d in T.DOMAINS) + f" install={dt:.1f}s", flush=True)
    restore_patch("global")

    out["nll"]["oracle"] = {}
    out["per_seq"]["oracle"] = {}
    out["switch"]["oracle"] = {}
    for d in T.DOMAINS:
        dt, nb = apply_patch(d)
        out["switch"]["oracle"][d] = {"install_s": round(dt, 2), "bytes": nb}
        tot, n, per, ps = T.eval_nll(eng["model"], tok, evalsets[d],
                                     max_len=64, batch_size=8, return_per_seq=True)
        out["nll"]["oracle"][d] = per
        out["per_seq"]["oracle"][d] = ps
        print(f"[oracle] {d}: {per:.4f} install={dt:.1f}s", flush=True)
        restore_patch(d)

    out["nll"]["predicted"] = {}
    out["per_seq"]["predicted"] = {}
    out["switch"]["predicted"] = {}
    # 按预测域分桶: 每桶装一次补丁, 桶内可能混有其它真实域
    buckets = {cd: [] for cd in doms}
    for i, p in enumerate(pred):
        buckets[doms[int(p)]].append(i)
    pred_seq_by_dom = {d: None for d in doms}
    for cd, idxs in buckets.items():
        if not idxs:
            continue
        dt, nb = apply_patch(cd)
        out["switch"]["predicted"][cd] = {"install_s": round(dt, 2),
                                          "bytes": nb, "n_seq": len(idxs)}
        # 从 flat 序列池取对应文本
        flat_texts = sum([evalsets[d] for d in T.DOMAINS], [])
        tot, n, per, ps = T.eval_nll(eng["model"], tok, [flat_texts[i] for i in idxs],
                                     max_len=64, batch_size=8, return_per_seq=True)
        restore_patch(cd)
        print(f"[predicted] bucket={cd} n={len(idxs)} nll={per:.4f} install={dt:.1f}s",
              flush=True)
        for j, i in enumerate(idxs):
            d_true = doms[int(ytrue[i])]
            if pred_seq_by_dom[d_true] is None:
                pred_seq_by_dom[d_true] = [None] * len(evalsets[d_true])
            # 该序列在真实域文本列表中的位置
            base_pos = i - sum(len(evalsets[x]) for x in T.DOMAINS[:ytrue[i]])
            pred_seq_by_dom[d_true][base_pos] = ps[j]
    for d in T.DOMAINS:
        ps = pred_seq_by_dom[d]
        if ps is None or any(x is None for x in ps):
            out["per_seq"]["predicted"][d] = None
            out["nll"]["predicted"][d] = None
            continue
        out["per_seq"]["predicted"][d] = ps
        out["nll"]["predicted"][d] = sum(x[0] for x in ps) / sum(x[1] for x in ps)

    out["predicted_minus_oracle"] = {
        d: (None if (out["nll"]["predicted"][d] is None or out["nll"]["oracle"][d] is None)
            else out["nll"]["predicted"][d] - out["nll"]["oracle"][d])
        for d in T.DOMAINS}

    xor = {}
    for a in T.DOMAINS:
        for b in T.DOMAINS:
            if a < b:
                n_xor = int(np.logical_xor(high[a], high[b]).sum())
                xor[f"{a}-{b}"] = {"n_experts": n_xor,
                                   "bytes": n_xor * expert_bytes}
    out["switch"]["delta_between_patches"] = xor
    out["switch"]["expert_bytes"] = expert_bytes
    out["elapsed_s"] = round(time.time() - t0, 1)

    p = os.path.join(T.RES_DIR, "switch.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[switch] saved {p}", flush=True)


if __name__ == "__main__":
    main()
