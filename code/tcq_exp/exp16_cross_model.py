"""E16: 跨模型普适性对照 —— H1(路由域可分) + H3(前缀路由分类)。

在 Qwen3-30B-A3B 与 gemma-4-26B-A4B 上复刻主模型的协议:
  - 每域 24 条 calib / 16 条 eval, 前缀 32 token (CPU bf16 前向)
  - 逐条路由直方图 [L,E] (top-k 权重按专家聚合, 逐层归一化)
  - H1: 域对 top-20% Jaccard + Spearman
  - H3: calib 训练 / eval 测试 的逻辑回归准确率 + macro-F1 + 混淆矩阵
输出: results/cross_model.json
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

MODELS = {
    "qwen3-30b-a3b": r"E:\CUDA11.8\model\Qwen3-30B-A3B",
    "gemma4-26b-a4b": r"E:\CUDA11.8\model\gemma-4-26B-A4B",
}
MAX_LEN = 32
N_CALIB = 24
N_EVAL = 16
THREADS = 16


def find_moe_blocks(model):
    """通用: 返回 [(layer_name, block_mod, router_mod)]。

    两种结构:
      Qwen3 风格: <layer>.mlp 持有 gate 路由器, forward 返回 (probs, weights, idx);
      Gemma4 风格: 解码层直接持有 router + experts, router forward 返回同形三元组。
    """
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
    """逐条路由直方图 -> np [n_texts, L*E]"""
    model, tok = eng["model"], eng["tok"]
    blocks = eng["blocks"]
    L, E, K = eng["n_layers"], eng["n_experts"], eng["top_k"]
    out = np.zeros((len(texts), L, E), dtype=np.float64)

    def run_one(text, ti):
        hists = [np.zeros(E, dtype=np.float64) for _ in range(L)]
        hooks = []

        def make_hook(li):
            def hook(m, inp, output):
                if isinstance(output, (tuple, list)):
                    # (probs, top_k_weights, top_k_index): Qwen3 / Gemma4 统一
                    idx_t, w_t = output[2], output[1]
                    inds = idx_t.reshape(-1).detach().cpu().numpy()
                    vals = w_t.reshape(-1).detach().double().cpu().numpy()
                else:
                    logits = output.float()
                    topv, topi = torch.topk(torch.softmax(logits, dim=-1), K, dim=-1)
                    inds = topi.reshape(-1).cpu().numpy()
                    vals = topv.reshape(-1).double().cpu().numpy()
                np.add.at(hists[li], inds, vals)
            return hook

        for li, (_, mod, router) in enumerate(blocks):
            hooks.append(router.register_forward_hook(make_hook(li)))
        try:
            with torch.inference_mode():
                enc = tok([text], return_tensors="pt", padding="longest",
                          truncation=True, max_length=max_len,
                          add_special_tokens=False)
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                      use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        h = np.stack(hists)
        h /= np.maximum(h.sum(axis=1, keepdims=True), 1e-12)
        out[ti] = h

    for i, t in enumerate(texts):
        run_one(t, i)
        if (i + 1) % 8 == 0:
            print(f"  [route] {i+1}/{len(texts)}", flush=True)
    return out.reshape(len(texts), -1)


def topk_sets(hists, frac=0.2):
    """hists [L,E] 均值 -> bool [L,E] top-frac"""
    L, E = hists.shape
    k = max(1, int(E * frac))
    m = np.zeros((L, E), dtype=bool)
    for l in range(L):
        m[l, np.argsort(-hists[l])[:k]] = True
    return m


def jacc(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def spearman(x, y):
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
    return {"mu": mu, "sd": sd, "W": W, "b": b, "classes": classes}


def run_model(tag, path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {tag}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=True)
    except Exception as e:
        from transformers import AutoModelForImageTextToText
        print(f"[load] {tag}: causal-LM 加载失败({repr(e)[:80]}), 改用 ImageTextToText", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()
    blocks = find_moe_blocks(model)
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    top_k = 8
    for key in ("num_experts_per_tok", "num_experts_per_token", "top_k_experts", "moe_top_k"):
        v = getattr(cfg, key, None)
        if v:
            top_k = int(v)
            break
    n_exp = 0
    for key in ("num_experts", "num_local_experts", "moe_num_experts"):
        v = getattr(cfg, key, None)
        if v:
            n_exp = int(v)
            break
    if n_exp == 0:
        n_exp = int(blocks[0][2].weight.shape[0])
    eng = {"model": model, "tok": tok, "blocks": blocks,
           "n_layers": len(blocks), "n_experts": n_exp, "top_k": top_k}
    print(f"[load] {tag}: {len(blocks)} MoE layers, E={n_exp}, top_k={top_k}, "
          f"{time.time()-t0:.0f}s", flush=True)

    feats, labels, split = {}, {}, {}
    for d in T.DOMAINS:
        di = list(T.DOMAINS).index(d)
        cal = [r["text"] for r in T.load_domain(d, "calib")[:N_CALIB]]
        ev = [r["text"] for r in T.load_domain(d, "eval")[:N_EVAL]]
        t1 = time.time()
        feats[(d, "calib")] = capture(eng, cal, MAX_LEN)
        feats[(d, "eval")] = capture(eng, ev, MAX_LEN)
        labels[(d, "calib")] = np.full(len(cal), di)
        labels[(d, "eval")] = np.full(len(ev), di)
        print(f"[{tag}] {d}: calib {len(cal)} eval {len(ev)} "
              f"({time.time()-t1:.0f}s)", flush=True)

    Xc = np.vstack([feats[(d, "calib")] for d in T.DOMAINS])
    yc = np.concatenate([labels[(d, "calib")] for d in T.DOMAINS])
    Xe = np.vstack([feats[(d, "eval")] for d in T.DOMAINS])
    ye = np.concatenate([labels[(d, "eval")] for d in T.DOMAINS])

    # ---- H1: 域平均直方图 ----
    L, E = eng["n_layers"], eng["n_experts"]
    mean_h = {}
    for d in T.DOMAINS:
        h = feats[(d, "calib")].reshape(-1, L, E).mean(axis=0)
        mean_h[d] = h
    h1 = {}
    for i, a in enumerate(T.DOMAINS):
        for b in T.DOMAINS[i + 1:]:
            ma, mb = topk_sets(mean_h[a]), topk_sets(mean_h[b])
            h1[f"{a}-{b}"] = {
                "jaccard_top20": jacc(ma, mb),
                "spearman_all_units": spearman(mean_h[a].reshape(-1),
                                               mean_h[b].reshape(-1))}
    print(f"[{tag}] H1 " + json.dumps(h1, ensure_ascii=False), flush=True)

    # ---- H3: 分类 ----
    k = len(T.DOMAINS)
    lr = fit_logreg(Xc, yc, list(range(k)))
    Z = (Xe - lr["mu"]) / lr["sd"]
    pe = np.argmax(Z @ lr["W"] + lr["b"], axis=1)
    acc = float((pe == ye).mean())
    f1s = []
    for c in range(k):
        tp = int(((ye == c) & (pe == c)).sum())
        fp = int(((ye != c) & (pe == c)).sum())
        fn = int(((ye == c) & (pe != c)).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1s.append(2 * pr * rc / max(pr + rc, 1e-12))
    conf = [[int(((ye == t) & (pe == q)).sum()) for q in range(k)] for t in range(k)]
    h3 = {"acc": acc, "macro_f1": float(np.mean(f1s)), "confusion": conf,
          "n_train": len(yc), "n_test": len(ye), "max_len": MAX_LEN}
    print(f"[{tag}] H3 acc={acc:.3f} f1={np.mean(f1s):.3f} conf={conf}", flush=True)

    return {"model": tag, "n_layers": L, "n_experts": E, "top_k": eng["top_k"],
            "max_len": MAX_LEN, "n_calib": N_CALIB, "n_eval": N_EVAL,
            "h1": h1, "h3": h3,
            "mean_hist": {d: mean_h[d].tolist() for d in T.DOMAINS}}


def main():
    torch.set_num_threads(THREADS)
    out = {}
    for tag, path in MODELS.items():
        if not os.path.isdir(path) or not any(
                f.endswith(".safetensors") for f in os.listdir(path)):
            print(f"[skip] {tag}: weights not ready at {path}", flush=True)
            out[tag] = {"status": "missing"}
            continue
        t0 = time.time()
        out[tag] = run_model(tag, path)
        out[tag]["elapsed_s"] = round(time.time() - t0, 1)
        import gc
        gc.collect()
    p = os.path.join(T.RES_DIR, "cross_model.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[cross_model] saved {p}", flush=True)


if __name__ == "__main__":
    main()
