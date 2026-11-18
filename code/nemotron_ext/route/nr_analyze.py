"""E-R2 Nemotron 臂分析：Q3_K_M vs Q6_K 的路由一致性。

用法: python nr_analyze.py
产物: results/nr_summary.json
"""
import json
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEED = 20261118
NBOOT = 1000
TOPK = 8
SKIP = 8


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


def jac(A, B):
    inter = (A[:, :, None] == B[:, None, :]).any(axis=2).sum(axis=1)
    return inter / (2.0 * TOPK - inter)


def js(P, Q):
    m = 0.5 * (P + Q)
    e = 1e-12
    a = np.sum(P * (np.log(np.maximum(P, e)) - np.log(np.maximum(m, e))), axis=1)
    b = np.sum(Q * (np.log(np.maximum(Q, e)) - np.log(np.maximum(m, e))), axis=1)
    return 0.5 * (a + b) / np.log(2.0)


def main():
    ref_p, ref_n = os.path.join(RES, "probs_nemotron_q6k.bin"), os.path.join(RES, "nll_nemotron_q6k.bin")
    arm_p, arm_n = os.path.join(RES, "probs_nemotron_q3km.bin"), os.path.join(RES, "nll_nemotron_q3km.bin")
    ip_ref, in_ref = iter_probs(ref_p), iter_nll(ref_n)
    ip_arm, in_arm = iter_probs(arm_p), iter_nll(arm_n)
    per_text = []
    n_layers = None
    while True:
        try:
            t_ref, l_ref = next(ip_ref)
            _, tn_ref, nll_ref = next(in_ref)
        except StopIteration:
            break
        t_arm, l_arm = next(ip_arm)
        _, tn_arm, nll_arm = next(in_arm)
        if not np.array_equal(t_ref, tn_ref) or not np.array_equal(t_arm, tn_arm):
            raise RuntimeError("probs/nll token mismatch")
        if not np.array_equal(t_ref, t_arm):
            raise RuntimeError("token mismatch between configs")
        if n_layers is None:
            n_layers = len(l_ref)
        R = {i: p for i, (il, p) in enumerate(l_ref)}
        A = {i: p for i, (il, p) in enumerate(l_arm)}
        n = len(t_ref)
        exact = np.zeros(n_layers)
        jacc = np.zeros(n_layers)
        jsd = np.zeros(n_layers)
        sev = np.zeros(n, dtype=np.int32)
        for i in range(n_layers):
            tr, ta = topk(R[i]), topk(A[i])
            eq = np.all(tr == ta, axis=1)
            exact[i] = eq.sum()
            jacc[i] = jac(tr, ta).sum()
            jsd[i] = js(R[i], A[i]).sum()
            sev += (~eq).astype(np.int32)
        per_text.append({"n": n, "exact": exact, "jac": jacc, "js": jsd, "sev": sev,
                         "dnll": (nll_arm - nll_ref).astype(np.float64),
                         "mask": np.arange(n) >= SKIP})
    rng = np.random.default_rng(SEED)

    def agg(idxs):
        ntok = sum(per_text[i]["n"] for i in idxs)
        sev = np.concatenate([per_text[i]["sev"][per_text[i]["mask"]] for i in idxs])
        dn = np.concatenate([per_text[i]["dnll"][per_text[i]["mask"]] for i in idxs])
        half = (n_layers + 1) // 2
        bstats = []
        step = max(1, n_layers // 4)
        for lo in range(0, n_layers + 1, step):
            hi = min(n_layers, lo + step - 1)
            m = (sev >= lo) & (sev <= hi)
            bstats.append({"sev": [lo, hi], "n": int(m.sum()),
                           "dNLL": float(dn[m].mean()) if m.any() else None})
        return {
            "n_text": len(idxs), "n_token": int(ntok),
            "exact_match": float(sum(per_text[i]["exact"].sum() for i in idxs) / (ntok * n_layers)),
            "jaccard": float(sum(per_text[i]["jac"].sum() for i in idxs) / (ntok * n_layers)),
            "js_bits": float(sum(per_text[i]["js"].sum() for i in idxs) / (ntok * n_layers)),
            "median_sev": float(np.median(sev)),
            "flip_majority_n": int((sev >= half).sum()), "stable_n": int((sev == 0).sum()),
            "dNLL_mean": float(dn.mean()),
            "dNLL_majority": float(dn[sev >= half].mean()) if (sev >= half).any() else None,
            "dNLL_stable": float(dn[sev == 0].mean()) if (sev == 0).any() else None,
            "bins": bstats,
        }

    overall = agg(list(range(len(per_text))))
    boots = {"exact": [], "js": []}
    for _ in range(NBOOT):
        a = agg(list(rng.integers(0, len(per_text), len(per_text))))
        boots["exact"].append(a["exact_match"])
        boots["js"].append(a["js_bits"])
    overall["exact_ci95"] = [float(np.percentile(boots["exact"], 2.5)),
                             float(np.percentile(boots["exact"], 97.5))]
    overall["js_ci95"] = [float(np.percentile(boots["js"], 2.5)),
                          float(np.percentile(boots["js"], 97.5))]
    e = overall["exact_match"]
    verdict = "P1_premise_holds" if e >= 0.95 else ("P2_routing_shift" if e >= 0.85
                                                     else "P3_routing_not_negligible")
    out = {"prereg": "PREREG.md", "n_layers": n_layers, "overall": overall,
           "verdict": verdict,
           "note": "参照为 Q6_K（非 BF16），一致率是下界；Q6_K 在本模型 GSM8K 上并不优于 Q3_K_M（C4: 0.97 vs 0.98），故 ΔNLL 不以 Q6_K 为真值"}
    with open(os.path.join(RES, "nr_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
