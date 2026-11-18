"""实验6: marginal-value 对照 (usage × Δerr)。

动机: 3.3/3.5 的配额分数用的是 s = usage × err(INT4 vs BF16)，那是"总误差"；
     真正决定"某个专家该不该保留 INT4"的是**边际误差增量** Δerr = err(INT3) − err(INT4)，
       marginal m(e,l,D) = usage(e,l,D) × Δerr(e,l,D)。

阶段 A (measure): 复用 E1 的捕获与逐专家前向, 额外算 INT3 的专家输出误差
  输出 results/marginal.json: {m, s4(=s), s3, usage, delta_err}
阶段 B (alloc): 用 m 作为分数做 25% 高精度配额, 三域 NLL, 对照 usage-only / 完整 s

用法: python exp6_marginal.py --part measure   (约 20-30 min)
      python exp6_marginal.py --part alloc      (约 4 min)
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
import safetensors

import tcq_lib as T

N_SEQ = 12
P_HIGH = 0.25
N_EVAL_SEQ = 8
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")


def expert_out(x, w_gu, w_dn):
    gu = x @ w_gu.t()
    g, u = gu.chunk(2, dim=-1)
    return (F.silu(g) * u) @ w_dn.t()


def part_measure():
    from moe_lib import Int4Store
    eng = T.build_engine()
    tok = eng["tok"]
    wmap = T.load_weight_map()
    st4 = eng["store"]
    st3 = Int4Store(T.STORE_INT3, mmap_mode="r")

    caps = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "calib")[:N_SEQ]
        texts = [r["text"] for r in rows]
        t0 = time.time()
        caps[dom] = T.capture_layer_stats(eng, texts, max_len=96, batch_size=6)
        print(f"[capture] {dom}: {len(texts)} seq {time.time()-t0:.1f}s", flush=True)

    L = len(eng["model"].model.language_model.layers)
    E = st4.num_experts
    D = len(T.DOMAINS)
    s4 = np.zeros((L, E, D))
    s3 = np.zeros((L, E, D))
    u = np.zeros((L, E, D))

    shard_files = {}
    t_all = time.time()
    for layer in range(L):
        t0 = time.time()
        gu_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        dn_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        for key in (gu_key, dn_key):
            sh = wmap[key]
            if sh not in shard_files:
                shard_files[sh] = safetensors.safe_open(
                    os.path.join(T.MOE_MODEL_PATH, sh), framework="pt")
        gu_all = shard_files[wmap[gu_key]].get_tensor(gu_key)
        dn_all = shard_files[wmap[dn_key]].get_tensor(dn_key)
        v4 = st4.layer_views(layer)
        v3 = st3.layer_views(layer)

        per_dom = []
        for dom in T.DOMAINS:
            per_dom.append((caps[dom][layer]["x"],
                            caps[dom][layer]["topi"].long(),
                            caps[dom][layer]["topv"].float()))

        for e in range(E):
            w_gu = gu_all[e].float()
            w_dn = dn_all[e].float()
            q4_gu = T.dequant_int4(v4["gu_q"][e], v4["gu_s"][e], torch.float32, st4.group)
            q4_dn = T.dequant_int4(v4["dn_q"][e], v4["dn_s"][e], torch.float32, st4.group)
            q3_gu = T.dequant_int4(v3["gu_q"][e], v3["gu_s"][e], torch.float32, st3.group)
            q3_dn = T.dequant_int4(v3["dn_q"][e], v3["dn_s"][e], torch.float32, st3.group)
            for di, (x, topi, topv) in enumerate(per_dom):
                tok_idx, slot = torch.nonzero(topi == e, as_tuple=True)
                if tok_idx.numel() == 0:
                    continue
                wts = topv[tok_idx, slot].numpy()
                xe = x[tok_idx].float()
                y_b = expert_out(xe, w_gu, w_dn)
                den = (y_b.norm(dim=-1) + 1e-6).numpy()
                rel4 = ((expert_out(xe, q4_gu, q4_dn) - y_b).norm(dim=-1)).numpy() / den
                rel3 = ((expert_out(xe, q3_gu, q3_dn) - y_b).norm(dim=-1)).numpy() / den
                s4[layer, e, di] += float((wts * rel4).sum())
                s3[layer, e, di] += float((wts * rel3).sum())
                u[layer, e, di] += float(wts.sum())
        if layer % 8 == 0:
            print(f"[marginal] layer {layer} done ({time.time()-t0:.1f}s)", flush=True)

    m = s3 - s4
    out = {"n_seq": N_SEQ, "domains": list(T.DOMAINS),
           "definition": "m = usage * (err_INT3 - err_INT4), err = w-weighted relative expert FFN output error vs BF16",
           "m": m.tolist(), "s4": s4.tolist(), "s3": s3.tolist(), "usage": u.tolist(),
           "elapsed_s": round(time.time() - t_all, 1)}
    p = os.path.join(T.RES_DIR, "marginal.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[marginal] saved {p} ({time.time()-t_all:.0f}s)", flush=True)

    for di, dom in enumerate(T.DOMAINS):
        rel = np.where(u[..., di] > 0, m[..., di] / np.maximum(s4[..., di], 1e-12), np.nan)
        print(f"[marginal] {dom}: s3/s4 ratio median={np.nanmedian(rel):.2f} "
              f"p10={np.nanpercentile(rel, 10):.2f} p90={np.nanpercentile(rel, 90):.2f}",
              flush=True)


def select_per_layer(score, frac):
    L, E = score.shape
    k = max(1, int(E * frac))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        high[l, np.argsort(-score[l])[:k]] = True
    return high


def part_alloc():
    from moe_lib import Int4Store
    mg = json.load(open(os.path.join(T.RES_DIR, "marginal.json"), encoding="utf-8"))
    m = np.array(mg["m"])
    doms = mg["domains"]
    al = json.load(open(os.path.join(T.RES_DIR, "allocation.json"), encoding="utf-8"))
    ref = {c["name"]: c["nll"] for c in al["configs"]}

    eng = T.build_engine()
    tok = eng["tok"]
    store = eng["store"]
    patcher = T.ExpertPatcher(store)
    s3 = Int4Store(T.STORE_INT3, mmap_mode="r")
    v3 = {l: s3.layer_views(l) for l in range(s3.num_layers)}

    evalsets = {}
    for dom in ("math", "code"):
        rows = T.load_domain(dom, "eval")[:N_EVAL_SEQ]
        evalsets[dom] = [r["text"] for r in rows]

    def run(name, score):
        t0 = time.time()
        high = select_per_layer(score, P_HIGH)
        for l in range(store.num_layers):
            for e in np.nonzero(~high[l])[0]:
                patcher.set_expert(l, int(e), {p: v3[l][p][int(e)] for p in LAYER_PARTS})
        nll = {}
        for dom in ("math", "code"):
            tot, n, per = T.eval_nll(eng["model"], tok, evalsets[dom],
                                     max_len=64, batch_size=8)
            nll[dom] = per
        for l in range(store.num_layers):
            for e in np.nonzero(~high[l])[0]:
                patcher.restore_expert(l, int(e))
        gain = {d: float(ref["global"][d] - nll[d]) for d in nll}
        print(f"[{name}] nll=" + " ".join(f"{d}:{nll[d]:.4f}" for d in nll) +
              " gain=" + " ".join(f"{d}:{gain[d]:+.4f}" for d in gain) +
              f" ({time.time()-t0:.0f}s)", flush=True)
        return {"name": name, "nll": nll, "gain_vs_global": gain}

    configs = []
    for i, dom in enumerate(doms):
        if dom == "general":
            continue
        configs.append(run(f"{dom}_marginal", m[..., i]))

    p = os.path.join(T.RES_DIR, "marginal.json")
    mg["alloc"] = {"p_high": P_HIGH, "configs": configs, "ref_nll": ref,
                   "n_eval_seq": N_EVAL_SEQ}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mg, f)
    print(f"[marginal] updated {p} with alloc results")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["measure", "alloc"], required=True)
    a = ap.parse_args()
    part_measure() if a.part == "measure" else part_alloc()
