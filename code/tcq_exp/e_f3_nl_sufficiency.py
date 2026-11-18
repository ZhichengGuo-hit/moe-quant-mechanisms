"""E-F3: 专家重要性排序键的充分性判定 —— usage 单独是否足够 (论文 §5.2 边界)。

问题: 每层 25% 专家配额 (与 E9 / E13 / E18 同口径) 升到 INT4 整行补丁时,
      "选谁进配额" 能否只用 usage (路由命中量), 还是必须用完整敏感度
      s = usage × q̂ (q̂ 为单次命中的量化误差 / 敏感度代理)。

三档排序键 (只改排序键, 其余配置完全一致):
  usage  K(e,l) = Σ_d usage[e,l,d]
  s      K(e,l) = Σ_d s[e,l,d] = Σ_d usage[e,l,d]·q̂[e,l,d]   (sens_matrix["s"] 求和口径)
  qhat   K(e,l) = Σ_d q̂[e,l,d],  q̂ = s / usage (usage=0 时置 0)   (去掉 usage 权重)

底座: stores/packed2_sym4 (2bit 密排专家容器, 7.97 GiB) + 每层 top-64 (25%) 专家
      INT4 整行补丁 (双容器运行时, LayerMixed2Ext 混合内核, 与 E18 相同)。

评测: eval64 (每域 64 序列), 三域 NLL; 序列级 paired bootstrap (默认 1000 次,
      seed 20261104) 给出三档两两 NLL 差的 95% CI。
判定: diff = NLL(s) − NLL(usage)
      CI 覆盖 0          -> "usage_close"  (§5.2 边界关闭)
      CI 完全在 0 以下    -> "s_better"     (升格为发现)
      其它               -> "inconclusive"

用法:
    python e_f3_nl_sufficiency.py --dry-run          # 只读 JSON + 容器元数据, 打印重合率
    python e_f3_nl_sufficiency.py --n-boot 1000 --seed 20261104
输出: results/e_f3_nl_sufficiency.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch

import tcq_lib as T
from a1_two_bit_runtime import Packed2Store, LayerMixed2Ext, select_per_layer

P_HIGH = 0.25
BASE_STORE_NAME = "packed2_sym4"
PACKED2_DIR = os.path.join(HERE, "stores", BASE_STORE_NAME)
LLAMA_DIR = os.path.abspath(os.path.join(HERE, "..", "llama.cpp"))
KEYS = ("usage", "s", "qhat")
PAIRS = (("s", "usage"), ("qhat", "usage"), ("qhat", "s"))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def store_meta(path):
    """只读 index.json (不 mmap experts.bin)"""
    return load_json(os.path.join(path, "index.json"))


def git_commit(path):
    try:
        r = subprocess.run(["git", "-C", path, "log", "-1", "--format=%H %ci"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or f"unknown (exit {r.returncode})"
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def build_rank_keys(sm):
    """返回 (s, usage, qhat, keys, sens_info)。

    sens_matrix 只有 s 与 usage 两个分量, q̂ 未单独存盘;
    由 q̂ = s / usage 逐元素反推 (usage=0 的单元 q̂ 与 s 均为 0)。
    """
    if "s" not in sm or "usage" not in sm:
        raise SystemExit("sens_matrix.json 缺少 s 或 usage 分量, 无法构造三档排序键")
    s = np.asarray(sm["s"], dtype=np.float64)
    usage = np.asarray(sm["usage"], dtype=np.float64)
    qhat = np.zeros_like(s)
    np.divide(s, usage, out=qhat, where=usage > 0)
    keys = {"usage": usage.sum(-1),
            "s": s.sum(-1),
            "qhat": qhat.sum(-1)}
    info = {
        "fields_present": [k for k in ("s", "usage", "assign_cnt") if k in sm],
        "fields_used": ["s", "usage"],
        "shape": list(s.shape),
        "zero_usage_cells": int((usage == 0).sum()),
        "qhat_derivation": ("q̂ 未单独存盘, 由 s/usage 逐元素反推 "
                            "(q̂ 即单次路由命中的平均相对量化误差); "
                            "usage==0 的单元 q̂=0 且 s=0"),
    }
    return s, usage, qhat, keys, info


def make_selections(keys):
    return {name: select_per_layer(score, P_HIGH) for name, score in keys.items()}


def overlap_stats(sel):
    L = sel["usage"].shape[0]
    k = int(sel["usage"][0].sum())
    out = {"k_per_layer": k, "pairs": {}}
    for a, b in PAIRS:
        inter = np.array([(sel[a][l] & sel[b][l]).sum() for l in range(L)],
                         dtype=np.float64)
        kk = np.array([sel[a][l].sum() for l in range(L)], dtype=np.float64)
        union = np.array([(sel[a][l] | sel[b][l]).sum() for l in range(L)],
                         dtype=np.float64)
        ov = inter / kk
        jac = inter / union
        out["pairs"][f"{a}_vs_{b}"] = {
            "mean_overlap": round(float(ov.mean()), 4),
            "min_overlap": round(float(ov.min()), 4),
            "max_overlap": round(float(ov.max()), 4),
            "mean_jaccard": round(float(jac.mean()), 4),
            "per_layer_overlap": [round(float(v), 4) for v in ov]}
    return out


def print_dry(sm, sinfo, keys, sel, ov, p2meta, p4meta):
    L, E = keys["usage"].shape
    print(f"[sens] {BASE_STORE_NAME}: shape={tuple(keys['usage'].shape)} "
          f"fields={sinfo['fields_present']} n_seq={sm.get('n_seq')} "
          f"zero_usage_cells={sinfo['zero_usage_cells']}", flush=True)
    print(f"[sens] signal={sm.get('signal')}", flush=True)
    print(f"[sens] {sinfo['qhat_derivation']}", flush=True)
    print(f"[store] base={BASE_STORE_NAME} bits={p2meta['bits']} "
          f"total={p2meta['total_bytes'] / 2 ** 30:.2f}GiB "
          f"layers={p2meta['num_layers']} experts={p2meta['num_experts']}", flush=True)
    print(f"[store] patch=int4_store scheme={p4meta.get('scheme')} "
          f"total={p4meta['total_bytes'] / 2 ** 30:.2f}GiB", flush=True)
    print(f"[quota] frac={P_HIGH} k={ov['k_per_layer']}/{E} per layer", flush=True)
    print("layer  usage∩s  usage∩q̂  s∩q̂", flush=True)
    ps, pq, sq = (ov["pairs"]["s_vs_usage"]["per_layer_overlap"],
                  ov["pairs"]["qhat_vs_usage"]["per_layer_overlap"],
                  ov["pairs"]["qhat_vs_s"]["per_layer_overlap"])
    for l in range(L):
        print(f"{l:5d}  {ps[l]:7.4f}  {pq[l]:8.4f}  {sq[l]:6.4f}", flush=True)
    for name, rec in ov["pairs"].items():
        print(f"[dry] {name}: mean={rec['mean_overlap']:.4f} "
              f"min={rec['min_overlap']:.4f} max={rec['max_overlap']:.4f} "
              f"jaccard={rec['mean_jaccard']:.4f}", flush=True)


def build_env(args):
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu_threads": args.threads,
        "llama_cpp": {
            "path": LLAMA_DIR,
            "commit": git_commit(LLAMA_DIR),
            "used": False,
            "note": "本实验不走 llama.cpp; NLL 走 coserving torch 直接 2bit/INT4 混合内核",
        },
    }


def install(mask, model, ext, p2s, p4s, n_layers, E):
    for l in range(n_layers):
        pm = np.full(E, -1, dtype=np.int32)
        pm[mask[l]] = np.nonzero(mask[l])[0]
        model.model.language_model.layers[l].mlp.direct_ext = LayerMixed2Ext(
            ext, p2s.layer_views(l), p4s[l], torch.from_numpy(pm))


def agg_at(sums, cnts, idx):
    return float(sums[idx].sum() / max(cnts[idx].sum(), 1))


def ci_summary(vals, point):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point_estimate": round(float(point), 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "boot_mean": round(float(vals.mean()), 6),
            "boot_sd": round(float(vals.std()), 6)}


def bootstrap_diff(per_seq, nll, n_boot, seed, doms):
    """序列级 paired bootstrap: 同一重采样索引作用于三档, 返回 (diff, n_seqs)。"""
    sums = {name: {dom: np.array([x[0] for x in per_seq[name][dom]], dtype=np.float64)
                   for dom in doms} for name in KEYS}
    cnts = {name: {dom: np.array([x[1] for x in per_seq[name][dom]], dtype=np.int64)
                   for dom in doms} for name in KEYS}
    n_seqs = {}
    for dom in doms:
        n0 = len(per_seq["usage"][dom])
        n_seqs[dom] = n0
        for name in KEYS:
            assert len(per_seq[name][dom]) == n0, (name, dom)
            assert np.array_equal(cnts[name][dom], cnts["usage"][dom]), (name, dom)

    rng = np.random.default_rng(seed)
    acc = {f"{x}_minus_{y}": {"domain_mean": np.empty(n_boot),
                              **{dom: np.empty(n_boot) for dom in doms}}
           for x, y in PAIRS}
    for b in range(n_boot):
        for dom in doms:
            n = n_seqs[dom]
            idx = rng.integers(0, n, n)
            vals = {name: agg_at(sums[name][dom], cnts[name][dom], idx)
                    for name in KEYS}
            for x, y in PAIRS:
                acc[f"{x}_minus_{y}"][dom][b] = vals[x] - vals[y]
        for x, y in PAIRS:
            k = f"{x}_minus_{y}"
            acc[k]["domain_mean"][b] = float(np.mean([acc[k][dom][b] for dom in doms]))

    diff = {"sign_convention": "diff = NLL(first) − NLL(second); 负值表示 first 的 NLL 更低 (first 更优)",
            "pairs": {}}
    for x, y in PAIRS:
        name = f"{x}_minus_{y}"
        rec = {"per_domain": {}}
        for dom in doms:
            rec["per_domain"][dom] = ci_summary(
                acc[name][dom], nll[x][dom] - nll[y][dom])
        point_mean = float(np.mean([nll[x][dom] - nll[y][dom] for dom in doms]))
        rec["domain_mean"] = ci_summary(acc[name]["domain_mean"], point_mean)
        diff["pairs"][name] = rec
    return diff, n_seqs


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只读 sens_matrix 与容器元数据, 打印三档 top-25% 重合率")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpus", default=None)
    ap.add_argument("--slots", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(T.RES_DIR, "e_f3_nl_sufficiency.json"))
    a = ap.parse_args(argv)

    sm = load_json(os.path.join(T.RES_DIR, "sens_matrix.json"))
    s, usage, qhat, keys, sinfo = build_rank_keys(sm)
    sel = make_selections(keys)
    ov = overlap_stats(sel)

    p2meta = store_meta(PACKED2_DIR)
    p4meta = store_meta(T.STORE_INT4)

    if a.dry_run:
        print(f"[dry] n_boot={a.n_boot} seed={a.seed} (dry-run 不跑模型)", flush=True)
        print_dry(sm, sinfo, keys, sel, ov, p2meta, p4meta)
        return

    if a.cpus:
        from pin import pin_self
        print(f"[pin] {pin_self(cpus=a.cpus)}", flush=True)

    import int2_ext
    ext = int2_ext.load_ext(threads=a.threads)
    print("[ext] int2_ext ready", flush=True)
    eng = T.build_engine(slots=a.slots, cpu_threads=a.threads)
    tok, model = eng["tok"], eng["model"]
    n_layers = len(model.model.language_model.layers)
    E = eng["store"].num_experts
    assert p2meta["num_layers"] == n_layers and p2meta["num_experts"] == E, \
        (p2meta["num_layers"], p2meta["num_experts"], n_layers, E)

    p2s = Packed2Store(PACKED2_DIR)
    p4s = {l: eng["store"].layer_views(l) for l in range(n_layers)}

    evalsets = {}
    for dom in T.DOMAINS:
        rows = T.load_domain(dom, "eval64")
        evalsets[dom] = [r["text"] for r in rows]
        print(f"[data] {dom}: {len(evalsets[dom])} seq", flush=True)

    per_seq, nll, timing = {}, {}, {}
    t_all = time.time()
    for name in KEYS:
        t0 = time.time()
        install(sel[name], model, ext, p2s, p4s, n_layers, E)
        per_seq[name], nll[name] = {}, {}
        for dom in T.DOMAINS:
            tot, n, per, ps = T.eval_nll(model, tok, evalsets[dom], max_len=64,
                                         batch_size=8, return_per_seq=True)
            per_seq[name][dom] = ps
            nll[name][dom] = per
        timing[name] = round(time.time() - t0, 1)
        print(f"[{name}] int4={int(sel[name].sum())} nll=" +
              " ".join(f"{d}:{nll[name][d]:.4f}" for d in T.DOMAINS) +
              f" ({timing[name]}s)", flush=True)

    # ---- 序列级 paired bootstrap (同一重采样索引作用于三档) ----
    diff, n_seqs = bootstrap_diff(per_seq, nll, a.n_boot, a.seed, T.DOMAINS)

    dec = diff["pairs"]["s_minus_usage"]["domain_mean"]
    lo, hi = dec["ci95"]
    pt = dec["point_estimate"]
    if hi < 0:
        decision = "s_better"
        basis = (f"s−usage 三域均值 NLL 差 {pt:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] "
                 f"完全低于 0, s 的 NLL 显著更低 → 升格为发现")
    elif lo <= 0 <= hi:
        decision = "usage_close"
        basis = (f"s−usage 三域均值 NLL 差 {pt:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] "
                 f"覆盖 0, 两档在序列级噪声内等价 → §5.2 边界关闭")
    else:
        decision = "inconclusive"
        basis = (f"s−usage 三域均值 NLL 差 {pt:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] "
                 f"完全高于 0, usage 反而更低或方向不一致 → 需复核")

    out = {
        "experiment": "E-F3",
        "metadata": {
            "question": "每层 25% INT4 配额的选择是否可只用 usage 排序 (usage ≈ s), 还是必须用 s = usage × q̂",
            "env": build_env(a),
            "base_store": {
                "name": BASE_STORE_NAME,
                "dir": PACKED2_DIR,
                "bits": p2meta["bits"],
                "scheme": p2meta["scheme"],
                "total_gib": round(p2meta["total_bytes"] / 2 ** 30, 3),
                "num_layers": p2meta["num_layers"],
                "num_experts": p2meta["num_experts"],
            },
            "patch_store": {
                "name": os.path.basename(T.STORE_INT4),
                "dir": T.STORE_INT4,
                "scheme": p4meta["scheme"],
                "total_gib": round(p4meta["total_bytes"] / 2 ** 30, 3),
            },
            "quota": {
                "frac": P_HIGH,
                "k_per_layer": ov["k_per_layer"],
                "rule": "每层按排序键取 top-k, k = max(1, int(E * frac)), 其余专家保持 2bit",
                "runtime": "LayerMixed2Ext (2bit 密排底座 + INT4 整行补丁, 双容器)",
            },
            "rank_keys": {
                "usage": "K_usage(e,l) = Σ_d usage[e,l,d]",
                "s": "K_s(e,l) = Σ_d s[e,l,d] = Σ_d usage[e,l,d]·q̂[e,l,d] (sens_matrix['s'] 求和口径)",
                "qhat": "K_qhat(e,l) = Σ_d q̂[e,l,d], q̂ = s/usage (usage==0 时 q̂=0); 去掉 usage 权重",
            },
            "sens_matrix": {
                "path": os.path.join(T.RES_DIR, "sens_matrix.json"),
                "source_experiment": "exp1_sens.py",
                "signal": sm.get("signal"),
                "n_seq_calib": sm.get("n_seq"),
                "usage_semantics": "路由 top-8 softmax 权重按专家聚合的三域分量 (非原始命中计数)",
                **sinfo,
            },
            "bootstrap": {
                "n": a.n_boot,
                "seed": a.seed,
                "unit": "sequence",
                "paired": True,
                "resample_size": n_seqs,
                "ci": "percentile 2.5 / 97.5",
                "comparison": "diff = NLL(s) − NLL(usage); 另附 qhat 两两对比",
            },
            "eval": {
                "split": "eval64",
                "max_len": 64,
                "batch_size": 8,
                "domains": list(T.DOMAINS),
                "n_seq": n_seqs,
            },
        },
        "selection_overlap": ov,
        "nll": nll,
        "timing": timing,
        "per_seq": per_seq,
        "diff": diff,
        "decision": decision,
        "decision_basis": basis,
        "elapsed_s": round(time.time() - t_all, 1),
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[boot] {a.n_boot} 次 paired | s−usage 域均值 {pt:+.4f} "
          f"CI [{lo:+.4f}, {hi:+.4f}]", flush=True)
    print(f"[decision] {decision}: {basis}", flush=True)
    print(f"[e_f3] saved {a.out} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
