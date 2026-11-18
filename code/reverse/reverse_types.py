#!/usr/bin/env python
"""E-R1: Reverse 名单臂 —— usage 最低 10 层专家升档（与 C1.2 G_rand 同模板、同预算）。

模板与 G2/G_rand 完全一致:
  experts 基础 IQ2_XXS / 名单层专家 IQ4_XS / 非专家 Q5_K, imatrix = im_global2.gguf,
  预算 11.45 ± 0.50 GiB。
唯一改动: 名单从"随机 10 层 (seed 20261104)"换成 "全局 usage 最低的 10 层"。

usage 来自 E1 sens_matrix.json: score(l) = Σ_e Σ_d usage[l,e,d], 升序前 10 层。
规格中的"每层每域 usage 最低 64 专家"写入 meta.per_domain_bottom64 作为记录;
GGUF 专家张量为每层一个三维张量, --tensor-type-file 无法逐专家指定, 故构建在层粒度
完成 —— 与 C1.2 G_rand 的限制和口径完全相同, 对照仍是单变量 (只换名单)。

用法:
  python reverse_types.py --write
  python reverse_types.py --build --write
输出: reverse/h1_types/c1_reverse.txt, reverse/results/c1_reverse_meta.json,
      reverse/out/c1_reverse_im.gguf
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tcq_exp"))

import e_h1_typefiles as T  # noqa: E402
import e_h1_build as E1B  # noqa: E402

ARM = "c2_reverse"
TYPE_PATH = os.path.join(HERE, "h1_types", "c1_reverse.txt")
META = os.path.join(HERE, "results", "c1_reverse_meta.json")
OUT_DIR = os.path.join(HERE, "out")
OUT_GGUF = os.path.join(OUT_DIR, "c1_reverse_im.gguf")
QUANT_BIN = os.path.join(ROOT, "llama-upstream", "build-cuda", "bin",
                         "llama-quantize.exe")
IMATRIX = r"E:\CUDA11.8\model\e20\im_global2.gguf"
SENS = os.path.join(ROOT, "tcq_exp", "results", "sens_matrix.json")
TARGET, TOL = 11.45, 0.50
N_LAYER, N_EXPERT, FRAC = 40, 256, 0.25


def bottom_lists(usage, domains, k):
    """每域每层 usage 最低 k 个专家 + global(三域之和) 最低 k 个。"""
    out = {d: [sorted(np.argsort(usage[l, :, di])[:k].tolist()) for l in range(N_LAYER)]
           for di, d in enumerate(domains)}
    out["global"] = [sorted(np.argsort(usage[l].sum(-1))[:k].tolist())
                     for l in range(N_LAYER)]
    return out


def build_assign(ts, base, top, layers):
    want = set(layers)
    a = {}
    for t in ts:
        if not T.allows(t):
            continue
        name = t["name"]
        if T.EXPERT_RE.search(name):
            m = T.LAYER_RE.match(name)
            lay = int(m.group(1)) if m else -1
            a[name] = top if lay in want else base
        else:
            a[name] = "Q5_K"
    return a


def build(typefile, base):
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [QUANT_BIN, "--tensor-type-file", typefile, "--imatrix", IMATRIX,
           T.SRC, OUT_GGUF, base, "16"]
    log = os.path.join(HERE, "results", "c1_reverse_build.log")
    t0 = time.time()
    print("[build] " + " ".join(cmd), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    dur = (time.time() - t0) / 60
    sig = E1B.src_sig()
    ok = rc == 0 and os.path.exists(OUT_GGUF) and E1B.valid_gguf(OUT_GGUF, sig)
    size = os.path.getsize(OUT_GGUF) / 2**30 if os.path.exists(OUT_GGUF) else 0.0
    res = {"cmd": cmd, "log": log, "rc": rc, "dur_min": round(dur, 1),
           "size_gib": round(size, 3), "src_tensors": sig[0], "src_elements": sig[1],
           "tensor_match": bool(ok),
           "within_tol": bool(ok and abs(size - TARGET) <= TOL), "path": OUT_GGUF}
    print("[build] rc=%d %.1f min %.3f GiB match=%s within_tol=%s" % (
        rc, dur, size, res["tensor_match"], res["within_tol"]), flush=True)
    return res


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)

    h1 = json.load(open(T.META, encoding="utf-8"))
    g2 = h1["arms"]["g2"]
    base, top = g2["base_type"], g2["params"]["top"]

    ts = T.load(T.SRC)
    sens = json.load(open(SENS, encoding="utf-8"))
    usage = np.asarray(sens["usage"], dtype=np.float64)      # [40,256,3]
    domains = sens["domains"]
    score = usage.sum(axis=(1, 2))
    k = max(1, int(FRAC * N_EXPERT))
    layers = sorted(np.argsort(score)[:max(1, int(FRAC * N_LAYER))].tolist())

    assign = build_assign(ts, base, top, layers)
    est = T.total_gib(ts, assign)
    dev = est - g2["est_gib"]
    lists = bottom_lists(usage, domains, k)

    print("[c2_reverse] base=%s top=%s layers=%s" % (base, top, layers))
    print("[c2_reverse] est=%.3f GiB (G2 %.3f, dev %+.3f) 名单 %d 层 x %d 专家/域(仅记录)"
          % (est, g2["est_gib"], dev, len(layers), k))
    print("[c2_reverse] layer score: " + ", ".join(
        "l%d=%.0f" % (l, score[l]) for l in layers))

    meta = {
        "arm": ARM,
        "selection": "usage_ascending",
        "template": "e_h1_typefiles.py::assign_g2 (同预算 11.45±0.3 GiB 流程)",
        "reference_arm": "g2 (blk.0-9) / c1_grand (随机 10 层, seed 20261104)",
        "base_type": base, "top_type": top,
        "usage_source": SENS,
        "usage_metric": "score(l)=sum_e sum_d usage[l,e,d]",
        "reverse_layers": layers,
        "n_listed_layers": len(layers),
        "listed_expert_tensors": len(layers) * 3,
        "layer_scores": {str(l): float(score[l]) for l in layers},
        "est_gib": est, "dev_vs_g2_gib": dev, "target_gib": TARGET, "tol": TOL,
        "budget_check": {"est_gib": est, "target": TARGET, "tol": TOL,
                         "within_tol": bool(abs(est - TARGET) <= TOL),
                         "reference_est_gib": g2["est_gib"],
                         "ref_build_size_gib": 11.652},
        "per_domain_bottom64": lists,
        "per_domain_bottom64_note": ("每域每层 usage 最低 64/256 专家 (global = 三域 usage 之和); "
                                     "GGUF 专家张量为每层一个三维张量, --tensor-type-file 无法"
                                     "逐专家指定, 故构建按层粒度 (10/40 层) 实现, 与 G_rand 同构。"),
        "typefile": TYPE_PATH,
        "imatrix": IMATRIX,
        "note": ("G2/G_rand/reverse 三臂类型模板、imatrix、预算完全相同, 只换名单; "
                 "reverse = usage 最低 10 层。"),
    }
    if a.write:
        T.write_typefile(assign, base, TYPE_PATH)
        json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[write] %s\n[write] %s" % (TYPE_PATH, META))
    if a.build:
        meta["build"] = build(TYPE_PATH, base)
        json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[meta] ->", META)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
