#!/usr/bin/env python
"""E-H5 前置机制检查（零机时）：低频/长文档 token 的 expert 曝光偏离 GSM8K 校准分布？

数据来源: results/sens_matrix.json（E1，40 层 × 256 专家 × 3 域 usage/s）。
  usage(layer, expert, domain) = 域校准文本(12 条, max_len 96)上的路由权重占比。
代理: math 域 ≈ GSM8K 校准分布；general 域 ≈ 长文档正文（低频 token 占比更高）。
先验假设(H5): 长上下文中段 token 路由到与 GSM8K 校准不同的专家 →
  imatrix 校准覆盖不足 → TCQ 的 FP16 注意力/更高质量在 ≥16k 显现。
若 math 与 general 的专家分布几乎相同，则该先验减弱，NIAH 更可能平坦。

指标(逐层 + 汇总):
  cos      = 两域 usage 向量余弦
  js_bits  = Jensen-Shannon 散度(bit)
  tv       = 总变差距离 0.5*Σ|p-q|
  top20_jac= 各域 top-20% 专家集合 Jaccard
  oot20    = general 路由权重落在 math top-20% 之外的份额（校准覆盖缺口）

预注册判定:
  支撑   : 中位 cos ≤ 0.70 且 中位 js_bits ≥ 0.30 且 中位 oot20 ≥ 0.35
  减弱   : 中位 cos ≥ 0.90 或 中位 js_bits ≤ 0.10
  其余   : 混合

可选补充: 用 HF tokenizer 统计 general vs math 的低频 token 占比（纯 CPU）。

用法: python e_h5_mech_check.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SENS = os.path.join(RES, "sens_matrix.json")
OUT = os.path.join(RES, "e_h5_mech_check.json")
KFRAC = 0.2
TH = {"cos_support_max": 0.70, "js_support_min": 0.30, "oot20_support_min": 0.35,
      "cos_weak_min": 0.90, "js_weak_max": 0.10}


def kl_bits(p, q):
    m = (p > 0) & (q > 0)
    return float(np.sum(p[m] * np.log2(p[m] / q[m])))


def js_bits(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_bits(p, m) + 0.5 * kl_bits(q, m)


def pair_metrics(p, q, k):
    cos = float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-12))
    jac = len(set(np.argsort(-p)[:k]) & set(np.argsort(-q)[:k])) / len(set(np.argsort(-p)[:k]) | set(np.argsort(-q)[:k]))
    oot = float(q[np.argsort(-p)[k:]].sum())
    tv = float(0.5 * np.abs(p - q).sum())
    return {"cos": cos, "js_bits": js_bits(p, q), "tv": tv, "top20_jaccard": jac,
            "oot20": oot}


def _whitespace_stats():
    """tokenizer 不可用时的词级近似: 词形多样性/TTR/hapax 占比"""
    import re
    out = {}
    corp = {}
    for dom in ("math", "general"):
        rows = []
        with open(os.path.join(HERE, "data", "%s_calib.jsonl" % dom), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line)["text"])
        words = re.findall(r"\w+", "\n\n".join(rows[:12]).lower())
        corp[dom] = words
        out[dom] = {"words": len(words), "types": len(set(words)),
                    "ttr": round(len(set(words)) / max(1, len(words)), 4),
                    "hapax_share": round(sum(1 for w in words if words.count(w) == 1) / max(1, len(words)), 4)}
    allc = {w: corp["math"].count(w) + corp["general"].count(w) for w in set(corp["math"]) | set(corp["general"])}
    for dom in ("math", "general"):
        out[dom]["rare_share(global<=2)"] = round(
            sum(1 for w in corp[dom] if allc[w] <= 2) / max(1, len(corp[dom])), 4)
    return {"ok": True, "method": "whitespace-word (tokenizer 不可用)", **out}


def rare_token_stats():
    """general vs math 的低频 token 占比（纯 CPU；优先 HF tokenizer，回退词级）"""
    try:
        from transformers import AutoTokenizer
        import glob
        cands = [d for d in glob.glob(r"E:\CUDA11.8\model\*\*", recursive=False) if os.path.isdir(d)]
        tok = None
        for d in cands:
            if os.path.exists(os.path.join(d, "tokenizer_config.json")) and "Qwen3.6" in d:
                tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
                break
        if tok is None:
            return _whitespace_stats()
        out = {}
        for dom in ("math", "general"):
            rows = []
            with open(os.path.join(HERE, "data", "%s_calib.jsonl" % dom), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line)["text"])
            ids = tok("\n\n".join(rows[:12]))["input_ids"]
            cnt = {}
            for i in ids:
                cnt[i] = cnt.get(i, 0) + 1
            rare = sum(1 for i in ids if cnt[i] <= 2)
            out[dom] = {"tokens": len(ids), "vocab_types": len(cnt),
                        "rare_share(<=2 occurrences)": round(rare / max(1, len(ids)), 4)}
        return {"ok": True, "method": "hf-tokenizer", **out}
    except Exception as e:  # 任何失败都回退词级近似
        r = _whitespace_stats()
        r["tokenizer_error"] = str(e)[:120]
        return r


def main():
    s = json.load(open(SENS))
    doms = s["domains"]
    u = np.array(s["usage"], dtype=np.float64)
    n_layers, n_exp, n_dom = u.shape
    k = max(1, int(round(n_exp * KFRAC)))
    i_math, i_gen = doms.index("math"), doms.index("general")

    per_layer = {}
    pairs = {}
    for name, (a, b) in {"math_vs_general": (i_math, i_gen),
                         "math_vs_code": (i_math, doms.index("code"))}.items():
        rows = []
        for L in range(n_layers):
            p = u[L, :, a]
            q = u[L, :, b]
            if p.sum() <= 0 or q.sum() <= 0:
                continue
            p, q = p / p.sum(), q / q.sum()
            rows.append({"layer": L, **pair_metrics(p, q, k)})
        pairs[name] = {
            "n_layers": len(rows),
            "median": {m: float(np.median([r[m] for r in rows])) for m in ("cos", "js_bits", "tv", "top20_jaccard", "oot20")},
            "mean": {m: float(np.mean([r[m] for r in rows])) for m in ("cos", "js_bits", "tv", "top20_jaccard", "oot20")},
            "per_layer": rows,
        }

    mg = pairs["math_vs_general"]["median"]
    if mg["cos"] <= TH["cos_support_max"] and mg["js_bits"] >= TH["js_support_min"] and mg["oot20"] >= TH["oot20_support_min"]:
        verdict = "support"
    elif mg["cos"] >= TH["cos_weak_min"] or mg["js_bits"] <= TH["js_weak_max"]:
        verdict = "weak"
    else:
        verdict = "mixed"

    concl = {
        "support": "先验支撑：长文档正文(mid-doc 低频 token)路由到与 GSM8K 校准显著不同的专家集合，"
                   "校准缺口大 → H5 预测偏向'TCQ 在 ≥16k 出现优势'；NIAH 曲线值得期待非平坦。",
        "mixed": "混合：两域分布有偏移但强度不足以断言校准失效 → H5 期望中性，"
                 "按'平坦即报饱和带覆盖至 64k'的零假设处理。",
        "weak": "先验减弱：general 与 math 的专家分布几乎相同 → 校准覆盖缺口小，"
                "H5 更可能平坦；NIAH 请以'未检测到长度效应'为主要预期。",
    }[verdict]

    out = {
        "source": SENS,
        "domains": doms,
        "n_layers": n_layers, "n_experts": n_exp, "k_top": k,
        "k_frac": KFRAC,
        "proxy_note": ("math=GSM8K 类校准代理；general=长文档正文代理。"
                       "sens_matrix 的 usage 来自每域 12 条校准文本(max_len 96)，"
                       "无法按文档位置/词频分层，故本检查是域级粗代理，非逐 token 曝光。"),
        "thresholds": TH,
        "pairs": pairs,
        "rare_token_stats": rare_token_stats(),
        "verdict": verdict,
        "conclusion": concl,
    }
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT)
    for name in ("math_vs_general", "math_vs_code"):
        m = pairs[name]["median"]
        print("%-16s cos=%.3f js=%.3f bits tv=%.3f top20jac=%.3f oot20=%.3f" % (
            name, m["cos"], m["js_bits"], m["tv"], m["top20_jaccard"], m["oot20"]))
    print("verdict:", verdict)
    print(concl)
    rt = out["rare_token_stats"]
    if rt.get("ok"):
        key = "rare_share(<=2 occurrences)" if "rare_share(<=2 occurrences)" in rt.get("math", {}) else "rare_share(global<=2)"
        print("rare tokens (%s): math %s | general %s" % (
            rt.get("method"), rt["math"][key], rt["general"][key]))
    else:
        print("rare_token_stats skipped:", rt.get("error"))


if __name__ == "__main__":
    main()
