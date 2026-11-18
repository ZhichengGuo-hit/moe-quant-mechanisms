#!/usr/bin/env python
"""C1.2: G_rand 随机名单臂 —— 类型文件生成 + 量化构建 + H1a 同款校验。

模板 = e_h1_typefiles.py 的 G2 (experts 基础 IQ2_XXS / 名单层 IQ4_XS / 非专家 Q5_K),
即与 G1 同预算 11.45±0.3 GiB 流程; 只把"专家补丁名单"从 blk.0-9 换成随机 10/40 层
(25%, seed 20261104) —— 名单张量数 (30) 与 G2 完全相同, 故体积预算严格对齐。

关键限制 (为什么名单落在层粒度):
  GGUF 每层专家是单个三维张量 blk.N.ffn_{gate,up,down}_exps.weight,
  llama-quantize 的 --tensor-type-file 只能按张量名做正则匹配, 无法逐专家 (64/256)
  指定类型。规格里的"每域每层 64/256"随机专家清单仍按 seed=20261104 生成并写入 meta
  (供 tcmoe 补丁表复用), 但本次构建无法表达, 因此实际随机名单 = 10/40 层。

用法:
  python c1_rand_types.py --write                 # 生成 typefile + meta
  python c1_rand_types.py --build --write          # 再串行调用 llama-quantize 并校验
输出: h1_types/c1_grand.txt, results/c1_grand_meta.json, out/c1_grand_im.gguf
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e_h1_typefiles as T  # noqa: E402  复用 load/allows/EXPERT_RE/LAYER_RE/total_gib/write_typefile
import e_h1_build as E1B  # noqa: E402  复用 valid_gguf / src_sig

SEED = 20261104
ARM = "c1_grand"
TYPE_PATH = os.path.join(HERE, "h1_types", "c1_grand.txt")
META = os.path.join(HERE, "results", "c1_grand_meta.json")
OUT_DIR = os.path.join(HERE, "out")
OUT_GGUF = os.path.join(OUT_DIR, "c1_grand_im.gguf")
QUANT_BIN = os.path.join(HERE, "..", "llama-upstream", "build-cuda", "bin",
                         "llama-quantize.exe")
IMATRIX = r"E:\CUDA11.8\model\e20\im_global2.gguf"
SENS = os.path.join(HERE, "results", "sens_matrix.json")
TARGET, TOL = 11.45, 0.50


def random_expert_lists(domains, n_layer=40, n_expert=256, frac=0.25, seed=SEED):
    """每域每层随机 top-k 专家清单 (k = ceil(frac*n_expert) = 64)。"""
    k = max(1, int(frac * n_expert))
    rng = random.Random(seed)
    return {d: [sorted(rng.sample(range(n_expert), k)) for _ in range(n_layer)]
            for d in domains}, k


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
    log = os.path.join(HERE, "results", "c1_grand_build.log")
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
    n_layer = 40  # 名单只覆盖主层 (blk.0-39), blk.40/MTP 与 G2 一样保持基础类型

    ts = T.load(T.SRC)
    rng = random.Random(SEED)
    layers = sorted(rng.sample(range(n_layer), max(1, int(0.25 * n_layer))))
    assign = build_assign(ts, base, top, layers)
    est = T.total_gib(ts, assign)
    dev = est - g2["est_gib"]

    domains = json.load(open(SENS, encoding="utf-8"))["domains"]
    lists, k = random_expert_lists(domains)

    print("[c1_grand] base=%s top=%s random_layers=%s" % (base, top, layers))
    print("[c1_grand] est=%.3f GiB (G2 %.3f, dev %+.3f) 专家清单 %d 层 x %d/256 x %d 域(不可表达)"
          % (est, g2["est_gib"], dev, n_layer, k, len(domains)))

    meta = {
        "arm": ARM,
        "seed": SEED,
        "template": "e_h1_typefiles.py::assign_g2 (同预算 11.45±0.3 GiB 流程)",
        "reference_arm": "g2 (blk.0-9)",
        "reference_arm_reason": (
            "参照臂 = G2, 不是 G1: 五臂里只有 G2/G3/G4 带'专家补丁名单'。G1 = 角色臂 "
            "(expert 全 IQ2_S / 非 expert Q5_K, est 11.512 GiB), 没有可替换的名单, 对其做"
            "'随机名单'会得到与 G1 逐字节相同的类型文件; G2 = G1 + blk.0-9 (25% 层) 专家升档, "
            "名单张量数 30, 与本次随机名单同预算。故类型组合 (base/top/非专家) 原样取自 G2, "
            "只把层号名单换成 seed=20261104 随机抽的 10 层。"),
        "budget_check": {"est_gib": est, "target": TARGET, "tol": TOL,
                         "within_tol": bool(abs(est - TARGET) <= TOL),
                         "reference_est_gib": g2["est_gib"],
                         "ref_build_size_gib": 11.652},
        "base_type": base, "top_type": top,
        "random_layers": layers,
        "n_listed_layers": len(layers),
        "listed_expert_tensors": len(layers) * 3,
        "est_gib": est, "dev_vs_g2_gib": dev, "target_gib": TARGET, "tol": TOL,
        "per_domain_top64": lists,
        "per_domain_top64_note": ("每域每层随机 64/256 专家 (seed=%d); GGUF 专家张量为每层一个 "
                                  "三维张量, --tensor-type-file 无法逐专家指定, 故本次构建在层粒度"
                                  "实现 (10/40 层)。" % SEED),
        "typefile": TYPE_PATH,
        "imatrix": IMATRIX,
        "note": ("G_rand 与 G2 的名单张量数同为 30, 体积预算相同; 唯一差别是名单由 blk.0-9 "
                 "换成随机 10 层, 用于检验'名单选择'是否优于随机。"),
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
