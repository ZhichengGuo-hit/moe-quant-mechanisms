"""E-R2 分析：三配置路由一致性 + 翻转 token 的 NLL 代价 + 预注册判定。

用法:
    python r2_analyze.py
产物:
    results/r2_summary.json
    results/r2_by_layer.csv
"""
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WORK = os.path.join(HERE, "work")
SEED = 20261118
NBOOT = 1000
TOPK = 8
WARMUP_SKIP = 8
SYSTEM_DNLL = 0.01
MAJORITY = 20          # 40 层中 >=20 层集合不同 = 翻转 token（主口径）


def iter_nll(path):
    data = open(path, "rb").read()
    o = 0
    while o < len(data):
        magic, idx, n, nv, nl = struct.unpack_from("<5i", data, o)
        o += 20
        toks = np.frombuffer(data, dtype=np.int32, count=n, offset=o).copy()
        o += 4 * n
        nll = np.frombuffer(data, dtype=np.float32, count=n, offset=o).copy()
        o += 4 * n
        yield idx, toks, nll


def iter_probs(path):
    data = open(path, "rb").read()
    o = 0
    while o < len(data):
        magic, n = struct.unpack_from("<2i", data, o)
        o += 8
        toks = np.frombuffer(data, dtype=np.int32, count=n, offset=o).copy()
        o += 4 * n
        ne = struct.unpack_from("<i", data, o)[0]
        o += 4
        layers = []
        for _ in range(ne):
            il, ne0, ne1 = struct.unpack_from("<3i", data, o)
            o += 12
            a = np.frombuffer(data, dtype=np.float32, count=ne0 * ne1, offset=o).reshape(ne1, ne0).copy()
            o += 4 * ne0 * ne1
            layers.append((il, a))
        yield toks, layers


def topk(P):
    return np.sort(np.argpartition(-P, TOPK - 1, axis=1)[:, :TOPK], axis=1)


def jaccard(A, B):
    inter = (A[:, :, None] == B[:, None, :]).any(axis=2).sum(axis=1)
    return inter / (2.0 * TOPK - inter)


def js_bits(P, Q):
    m = 0.5 * (P + Q)
    e = 1e-12
    a = np.sum(P * (np.log(np.maximum(P, e)) - np.log(np.maximum(m, e))), axis=1)
    b = np.sum(Q * (np.log(np.maximum(Q, e)) - np.log(np.maximum(m, e))), axis=1)
    return 0.5 * (a + b) / np.log(2.0)


def load_meta():
    with open(os.path.join(WORK, "texts_meta.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    meta = load_meta()
    arms = ["tcq", "iq2xxs"]
    paths = {a: (os.path.join(RES, "probs_%s.bin" % a), os.path.join(RES, "nll_%s.bin" % a)) for a in
             ["bf16"] + arms}
    for a, (p, q) in paths.items():
        if not (os.path.exists(p) and os.path.exists(q)):
            print("missing", a)
            return 1

    n_text = len(meta)
    n_layers = 40
    rec = [{"domain": m["domain"], "n": 0,
            "arms": {a: {"exact": np.zeros(n_layers), "jac": np.zeros(n_layers),
                         "js": np.zeros(n_layers), "sev": None,
                         "flip": [], "keep": [], "mid": []} for a in arms}} for m in meta]

    its = {a: (iter_probs(paths[a][0]), iter_nll(paths[a][1])) for a in paths}
    hard_ref = np.zeros((n_layers, 256))
    soft_ref = np.zeros((n_layers, 256))
    hard_arm = {a: np.zeros((n_layers, 256)) for a in arms}
    soft_arm = {a: np.zeros((n_layers, 256)) for a in arms}
    for ti in range(n_text):
        d = {}
        for a in paths:
            tp, layers = next(its[a][0])
            _, tn, nll = next(its[a][1])
            if not np.array_equal(tp, tn):
                raise RuntimeError("%s text %d: probs/nll mismatch" % (a, ti))
            d[a] = (tp, layers, nll)
        for a in arms:
            if not np.array_equal(d[a][0], d["bf16"][0]):
                raise RuntimeError("text %d: token mismatch %s" % (ti, a))
        n = len(d["bf16"][0])
        rec[ti]["n"] = n
        P_ref = dict(d["bf16"][1])
        for a in arms:
            P_arm = dict(d[a][1])
            sev = np.zeros(n, dtype=np.int32)
            for il in range(n_layers):
                A, B = topk(P_ref[il]), topk(P_arm[il])
                eq = np.all(A == B, axis=1)
                rec[ti]["arms"][a]["exact"][il] += eq.sum()
                rec[ti]["arms"][a]["jac"][il] += jaccard(A, B).sum()
                rec[ti]["arms"][a]["js"][il] += js_bits(P_ref[il], P_arm[il]).sum()
                sev += (~eq).astype(np.int32)
                hard_arm[a][il] += np.bincount(B.reshape(-1), minlength=256)
                soft_arm[a][il] += P_arm[il].sum(axis=0)
                if a == arms[0]:
                    hard_ref[il] += np.bincount(A.reshape(-1), minlength=256)
                    soft_ref[il] += P_ref[il].sum(axis=0)
            rec[ti]["arms"][a]["sev"] = sev
            js_tok = np.array([js_bits(P_ref[il], P_arm[il]) for il in range(n_layers)]).T
            dnll = d[a][2] - d["bf16"][2]
            mask = np.arange(n) >= WARMUP_SKIP
            rec[ti]["arms"][a]["tok"] = {
                "sev": sev.astype(np.float64), "js": js_tok.mean(axis=1),
                "dnll": dnll.astype(np.float64), "mask": mask,
            }
            flip = (sev >= MAJORITY) & mask
            keep = (sev == 0) & mask
            mid = (sev > 0) & (sev < MAJORITY) & mask
            rec[ti]["arms"][a]["flip"].extend(dnll[flip].tolist())
            rec[ti]["arms"][a]["keep"].extend(dnll[keep].tolist())
            rec[ti]["arms"][a]["mid"].extend(dnll[mid].tolist())

    rng = np.random.default_rng(SEED)
    domains = ["math", "code", "general"]

    def agg(idxs, a):
        ntok = sum(rec[i]["n"] for i in idxs)
        S = [rec[i]["arms"][a] for i in idxs]
        fl = sum(len(s["flip"]) for s in S)
        kp = sum(len(s["keep"]) for s in S)
        md = sum(len(s["mid"]) for s in S)
        sfl = sum(float(np.sum(s["flip"])) for s in S)
        skp = sum(float(np.sum(s["keep"])) for s in S)
        smd = sum(float(np.sum(s["mid"])) for s in S)
        return {
            "n_text": len(idxs), "n_token": int(ntok),
            "exact_match": float(sum(s["exact"].sum() for s in S) / (ntok * n_layers)),
            "jaccard": float(sum(s["jac"].sum() for s in S) / (ntok * n_layers)),
            "js_bits": float(sum(s["js"].sum() for s in S) / (ntok * n_layers)),
            "flip_any_rate": float((fl + md + kp - kp) / max(fl + md + kp, 1)),
            "flip_majority_n": int(fl), "mid_n": int(md), "stable_n": int(kp),
            "dNLL_flip": float(sfl / fl) if fl else None,
            "dNLL_mid": float(smd / md) if md else None,
            "dNLL_stable": float(skp / kp) if kp else None,
            "dNLL_diff": float(sfl / fl - skp / kp) if fl and kp else None,
        }

    all_idx = list(range(n_text))
    out_arms = {}
    for a in arms:
        o = agg(all_idx, a)
        boots = {"exact": [], "js": [], "diff": []}
        for _ in range(NBOOT):
            idxs = list(rng.integers(0, n_text, n_text))
            x = agg(idxs, a)
            boots["exact"].append(x["exact_match"])
            boots["js"].append(x["js_bits"])
            if x["dNLL_diff"] is not None:
                boots["diff"].append(x["dNLL_diff"])
        o["exact_ci95"] = [float(np.percentile(boots["exact"], 2.5)),
                           float(np.percentile(boots["exact"], 97.5))]
        o["js_ci95"] = [float(np.percentile(boots["js"], 2.5)),
                        float(np.percentile(boots["js"], 97.5))]
        o["dNLL_diff_ci95"] = ([float(np.percentile(boots["diff"], 2.5)),
                                float(np.percentile(boots["diff"], 97.5))]
                               if boots["diff"] else None)
        o["by_domain"] = {dm: agg([i for i, r in enumerate(rec) if r["domain"] == dm], a)
                          for dm in domains}
        out_arms[a] = o

    # (c) 替代归因：翻转是普遍现象（每 token 至少 20 层不同），预注册的 flip-vs-stable 对照为空。
    # 改用"翻转严重度 / 路由 JS"与 ΔNLL 的关联，文本级 cluster bootstrap。
    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r

    def spearman(x, y):
        rx = ranks(x) - (len(x) - 1) / 2
        ry = ranks(y) - (len(y) - 1) / 2
        d = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
        return float(np.sum(rx * ry) / d) if d > 0 else 0.0

    def attribution(a):
        sev, js, dn = [], [], []
        for r in rec:
            t = r["arms"][a]["tok"]
            m = t["mask"]
            sev.append(t["sev"][m])
            js.append(t["js"][m])
            dn.append(t["dnll"][m])
        sev, js, dn = np.concatenate(sev), np.concatenate(js), np.concatenate(dn)
        bstats = []
        for lo, hi in [(20, 24), (25, 29), (30, 34), (35, 40)]:
            m = (sev >= lo) & (sev <= hi)
            bstats.append({"sev": [lo, hi], "n": int(m.sum()),
                           "dNLL": float(dn[m].mean()) if m.any() else None})
        med = float(np.median(sev))
        hi_m, lo_m = sev > med, sev <= med
        diff = float(dn[hi_m].mean() - dn[lo_m].mean())
        boots = []
        for _ in range(NBOOT):
            idxs = rng.integers(0, n_text, n_text)
            hs, ls = [], []
            for i in idxs:
                t = rec[i]["arms"][a]["tok"]
                m = t["mask"]
                s, d = t["sev"][m], t["dnll"][m]
                hm = s > med
                if hm.any():
                    hs.append(d[hm])
                if (~hm).any():
                    ls.append(d[~hm])
            if hs and ls:
                boots.append(float(np.concatenate(hs).mean() - np.concatenate(ls).mean()))
        return {
            "spearman_sev_dnll": spearman(sev, dn),
            "spearman_js_dnll": spearman(js, dn),
            "median_sev": med,
            "dnll_hi_minus_lo": diff,
            "dnll_hi_minus_lo_ci95": ([float(np.percentile(boots, 2.5)),
                                       float(np.percentile(boots, 97.5))] if boots else None),
            "bins": bstats,
            "note": "预注册的 flip(>=20 层) vs stable(0 层) 对照为空（stable_n=0），以严重度分层替代",
        }

    # s = usage x q 直接依赖的"聚合 usage 排序"是否稳定
    def usage_stats(a):
        per_layer = []
        for il in range(n_layers):
            hr = hard_ref[il] / max(hard_ref[il].sum(), 1)
            ha = hard_arm[a][il] / max(hard_arm[a][il].sum(), 1)
            sr = soft_ref[il] / max(soft_ref[il].sum(), 1)
            sa = soft_arm[a][il] / max(soft_arm[a][il].sum(), 1)
            tr = set(np.argsort(-hr)[:64].tolist())
            ta = set(np.argsort(-ha)[:64].tolist())
            per_layer.append({
                "layer": il,
                "tv_hard": float(0.5 * np.abs(hr - ha).sum()),
                "tv_soft": float(0.5 * np.abs(sr - sa).sum()),
                "top64_overlap": len(tr & ta) / 64.0,
                "pearson_hard": float(np.corrcoef(hr, ha)[0, 1]),
            })
        return {
            "per_layer": per_layer,
            "mean_tv_hard": float(np.mean([x["tv_hard"] for x in per_layer])),
            "mean_tv_soft": float(np.mean([x["tv_soft"] for x in per_layer])),
            "mean_top64_overlap": float(np.mean([x["top64_overlap"] for x in per_layer])),
            "min_top64_overlap": float(np.min([x["top64_overlap"] for x in per_layer])),
        }

    # 逐层与翻转集中度（主臂 tcq）
    by_layer = []
    for il in range(n_layers):
        ntok = sum(r["n"] for r in rec)
        S = [r["arms"]["tcq"] for r in rec]
        by_layer.append({
            "layer": il,
            "exact": float(sum(s["exact"][il] for s in S) / ntok),
            "jaccard": float(sum(s["jac"][il] for s in S) / ntok),
            "js_bits": float(sum(s["js"][il] for s in S) / ntok),
        })

    main = out_arms["tcq"]
    exact = main["exact_match"]
    systematic = (main["dNLL_diff_ci95"] is not None
                  and (main["dNLL_diff_ci95"][0] > 0 or main["dNLL_diff_ci95"][1] < 0)
                  and abs(main["dNLL_diff"] or 0) >= SYSTEM_DNLL)
    conc = None
    if exact >= 0.95 and not systematic:
        verdict = "P1_premise_holds"
    elif 0.85 <= exact < 0.95:
        verdict = "P2_routing_shift"
    else:
        verdict = "P3_routing_not_negligible"

    out = {
        "prereg": "PREREG.md", "seed": SEED, "nboot": NBOOT,
        "configs": {"ref": "bf16 f16 gguf (CPU)", "main": "tcq nl_global (fork CPU)",
                    "secondary": "iq2xxs (CPU)"},
        "definitions": {
            "flip_majority": "40 层中 >=%d 层的 top-8 集合与 BF16 不同" % MAJORITY,
            "flip_any": "任一层的 top-8 集合不同（预注册口径，几乎覆盖全部 token，仅作参考）",
            "warmup_skip": WARMUP_SKIP,
            "deviation": "预注册把\"翻转 token\"定义为任一层不同；实测该口径覆盖 100%% token，"
                         "无对照组。故主口径改为 >=20/40 层，并保留 mid/stable 分层；"
                         "此偏离在 PROGRESS.md 中记录。",
        },
        "arms": out_arms,
        "by_layer": by_layer,
        "attribution": {a: attribution(a) for a in arms},
        "usage_stability": {a: usage_stats(a) for a in arms},
        "systematic_direction": bool(systematic),
        "verdict": verdict,
        "verdict_text": {
            "P1_premise_holds": "精确匹配率>=95% 且无系统方向：s=usage*q 前提成立，M5 关闭",
            "P2_routing_shift": "匹配率 85-95%：量化诱导路由偏移，机制地图第四层",
            "P3_routing_not_negligible": "匹配率<85%：归因章必须加路由偏移限定语",
        }[verdict],
    }
    with open(os.path.join(RES, "r2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RES, "r2_by_layer.csv"), "w", encoding="utf-8") as f:
        f.write("layer,exact,jaccard,js_bits\n")
        for r in by_layer:
            f.write("%d,%.6f,%.6f,%.6f\n" % (r["layer"], r["exact"], r["jaccard"], r["js_bits"]))
    print(json.dumps({"tcq": {k: main[k] for k in
                              ["exact_match", "exact_ci95", "jaccard", "js_bits",
                               "flip_majority_n", "mid_n", "stable_n",
                               "dNLL_flip", "dNLL_mid", "dNLL_stable", "dNLL_diff",
                               "dNLL_diff_ci95"]},
                      "iq2xxs": {k: out_arms["iq2xxs"][k] for k in
                                 ["exact_match", "exact_ci95", "jaccard", "js_bits"]},
                      "attribution_tcq": out["attribution"]["tcq"],
                      "verdict": verdict, "systematic": systematic},
                     ensure_ascii=False, indent=1))
    print("by_domain tqc:", json.dumps(main["by_domain"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
