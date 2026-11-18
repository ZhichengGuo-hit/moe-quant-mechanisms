#!/usr/bin/env python
"""E-H1: 同预算受控组件消融 —— 五臂张量类型文件生成 + 体积对齐（11.45 ± 0.3 GiB）

臂定义（全部基于 qwen36_f16，41 层含 MTP）：
  G0  角色/层/张量无关：按张量名种子哈希在 IQ2_S / IQ3_XXS 间混合，二分调到 11.45 GiB
  G1  +角色：expert IQ2_S，非 expert Q5_K
  G2  +per-layer：G1 + blk.0-9 expert 升一档（对齐 C_layer 结构）
  G3  +per-tensor：G2 + ffn_down 比 gate/up 高一档（UD 式张量角色差异）
  G4  UD 实际清单：逐张量复刻 UD-Q2_K_XL 的 733 个类型；blk.40/nextn 取 blk.0 同角色类型

输出：h1_types/e_h1_g{0..4}.txt 与 results/e_h1_types_meta.json
"""
import argparse
import hashlib
import json
import os
import re
import sys

GGUF_PY = r"E:\CUDA11.8\train\llama.cpp\gguf-py"
sys.path.insert(0, GGUF_PY)
from gguf import GGUFReader

SRC = r"E:\CUDA11.8\model\e20\qwen36_f16.gguf"
UD = r"E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"
OUT_DIR = r"E:\CUDA11.8\train\tcq_exp\h1_types"
META = r"E:\CUDA11.8\train\tcq_exp\results\e_h1_types_meta.json"

TARGET = 11.45
TOL = 0.30
SEED = "20261108"

# (block, bytes per block)
QUANT = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q5_0": (32, 22), "Q5_1": (32, 24), "Q8_0": (32, 34),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
    "IQ1_S": (256, 50), "IQ1_M": (256, 50),
    "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82),
    "IQ3_XXS": (256, 98), "IQ3_S": (256, 110), "IQ4_XS": (256, 136), "IQ4_NL": (32, 18),
}
FALLBACK = {"IQ1_S": "IQ4_NL", "IQ1_M": "IQ4_NL", "IQ2_XXS": "IQ4_NL", "IQ2_XS": "IQ4_NL",
            "IQ2_S": "IQ4_NL", "IQ3_XXS": "IQ4_NL", "IQ3_S": "IQ4_NL", "IQ4_XS": "IQ4_NL",
            "Q2_K": "Q4_0", "Q3_K": "Q4_0", "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0"}
LADDER = ["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K"]
EXCL_SUB = ["_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight", "engram_gate.weight",
            "altup", "laurel", "per_layer_model_proj", "ssm_conv1d", "shortconv.conv.weight",
            "conv2d_spatial.weight"]
EXPERT_RE = re.compile(r"\.ffn_(gate|up|down)_exps\.weight$")
LAYER_RE = re.compile(r"blk\.(\d+)\.")


def load(path):
    r = GGUFReader(path)
    ts = []
    for t in r.tensors:
        ts.append({"name": t.name, "shape": tuple(int(x) for x in t.shape),
                   "n": int(t.n_elements), "srctype": t.tensor_type.name})
    return ts


def allows(t):
    name = t["name"]
    if len(t["shape"]) < 2:
        return False
    if not name.endswith("weight"):
        return False
    if any(s in name for s in EXCL_SUB):
        return False
    return True


def eff(q, ncols):
    blk, _ = QUANT[q]
    if ncols % blk != 0:
        q = FALLBACK.get(q, "F16")
    return q


def nbytes(q, n, ncols):
    q = eff(q, ncols)
    blk, sz = QUANT[q]
    return (n // blk) * sz


def total_gib(ts, assign):
    b = 0
    for t in ts:
        q = assign.get(t["name"])
        b += nbytes(q, t["n"], t["shape"][0]) if q else nbytes(t["srctype"], t["n"], t["shape"][0])
    return b / 2**30


def core_gib(ts, assign):
    b = 0
    for t in ts:
        if re.match(r"blk\.40\.|nextn\.", t["name"]):
            continue
        q = assign.get(t["name"])
        b += nbytes(q, t["n"], t["shape"][0]) if q else nbytes(t["srctype"], t["n"], t["shape"][0])
    return b / 2**30


def hname(name):
    return int(hashlib.sha256((SEED + "|" + name).encode()).hexdigest()[:8], 16) / 2**32


def up(q, k=1):
    return LADDER[min(len(LADDER) - 1, LADDER.index(q) + k)] if q in LADDER else q


def assign_g0(ts):
    names = [t["name"] for t in ts if allows(t)]
    fixed = sum(nbytes(t["srctype"], t["n"], t["shape"][0]) for t in ts if not allows(t))
    hs = {n: hname(n) for n in names}

    def tot2(p):
        b = fixed
        for t in ts:
            if not allows(t):
                continue
            q = "IQ3_XXS" if hs[t["name"]] < p else "IQ2_S"
            b += nbytes(q, t["n"], t["shape"][0])
        return b
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if tot2(mid) / 2**30 < TARGET:
            lo = mid
        else:
            hi = mid
    p = (lo + hi) / 2
    a = {t["name"]: ("IQ3_XXS" if hs[t["name"]] < p else "IQ2_S") for t in ts if allows(t)}
    frac = sum(1 for t in ts if allows(t) and a[t["name"]] == "IQ3_XXS") / max(1, len(a))
    return a, {"p": p, "frac_iq3xxs": frac}


def assign_g1(ts):
    a = {}
    for t in ts:
        if not allows(t):
            continue
        a[t["name"]] = "IQ2_S" if EXPERT_RE.search(t["name"]) else "Q5_K"
    return a, {}


DEBUG = False


def best_fit(ts, cands):
    best = None
    for label, a, extra in cands:
        g = total_gib(ts, a)
        d = abs(g - TARGET)
        if DEBUG:
            print("   cand %-18s est=%6.3f dev=%+.3f" % (label, g, g - TARGET))
        if best is None or d < best[0]:
            best = (d, g, label, a, extra)
    return best[3], dict(best[4], label=best[2], est_gib=best[1], dev=best[0])


def assign_g2(ts):
    cands = []
    for b in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "Q2_K"):
        for u in LADDER:
            if QUANT[u][1] / QUANT[u][0] <= QUANT[b][1] / QUANT[b][0] + 0.2:
                continue
            a = {}
            for t in ts:
                if not allows(t):
                    continue
                if EXPERT_RE.search(t["name"]):
                    m = LAYER_RE.match(t["name"])
                    a[t["name"]] = u if (m and int(m.group(1)) <= 9) else b
                else:
                    a[t["name"]] = "Q5_K"
            cands.append(("B=%s/U=%s" % (b, u), a, {"base": b, "top": u}))
    return best_fit(ts, cands)


def assign_g3(ts):
    cands = []
    for b in ("IQ2_XXS", "IQ2_XS", "IQ2_S"):
        for down_k in (1, 2):
            for top_k in (1, 2):
                a = {}
                for t in ts:
                    if not allows(t):
                        continue
                    m = EXPERT_RE.search(t["name"])
                    if not m:
                        a[t["name"]] = "Q5_K"
                        continue
                    kind = m.group(1)
                    lay = int(LAYER_RE.match(t["name"]).group(1))
                    top = lay <= 9
                    if kind == "down":
                        q = up(b, down_k + (top_k if top else 0))
                    else:
                        q = up(b, top_k if top else 0)
                    a[t["name"]] = q
                cands.append(("B=%s/d=%d/t=%d" % (b, down_k, top_k), a,
                              {"base": b, "down_boost": down_k, "top_boost": top_k}))
    return best_fit(ts, cands)


def assign_g4(ts):
    rud = GGUFReader(UD)
    udt = {t.name: t.tensor_type.name for t in rud.tensors}
    a = {}
    for t in ts:
        if not allows(t):
            continue
        name = t["name"]
        if name in udt:
            a[name] = udt[name]
            continue
        # blk.40/nextn：取 blk.0 同角色类型
        alt = re.sub(r"^blk\.40\.", "blk.0.", name)
        if alt in udt:
            a[name] = udt[alt]
        else:
            a[name] = "Q4_K"
    return a, {"ud_tensors": len(udt), "replicated": sum(1 for n in a if n in udt)}


def write_typefile(assign, base, path):
    lines = []
    for name in sorted(assign):
        lines.append("^%s$=%s" % (re.escape(name), assign[name]))
    lines.append(".*=%s" % base)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rc = argparse.ArgumentParser()
    global DEBUG
    rc.add_argument("--write", action="store_true")
    rc.add_argument("--debug", action="store_true")
    args = rc.parse_args()
    DEBUG = args.debug
    os.makedirs(OUT_DIR, exist_ok=True)

    ts = load(SRC)
    print("src tensors=%d  total=%.3fB  %.2f GiB" % (
        len(ts), sum(t["n"] for t in ts) / 1e9, os.path.getsize(SRC) / 2**30))
    ndiv = [t["name"] for t in ts if allows(t) and t["shape"][0] % 256]
    print("ncols 不能被 256 整除的可量化张量: %d %s" % (len(ndiv), ndiv[:5]))

    arms = {}
    a0, e0 = assign_g0(ts)
    arms["g0"] = (a0, dict(e0, base="IQ2_S"))
    a1, e1 = assign_g1(ts)
    arms["g1"] = (a1, dict(e1, base="IQ2_S"))
    a2, e2 = assign_g2(ts)
    arms["g2"] = (a2, dict(e2, base=e2["base"]))
    a3, e3 = assign_g3(ts)
    arms["g3"] = (a3, dict(e3, base=e3["base"]))
    a4, e4 = assign_g4(ts)
    arms["g4"] = (a4, dict(e4, base="Q2_K"))

    meta = {"target_gib": TARGET, "tol": TOL, "src": SRC, "ud": UD, "arms": {}}
    for k, (a, ex) in arms.items():
        g = total_gib(ts, a)
        c = core_gib(ts, a)
        ok = abs(g - TARGET) <= TOL
        base = ex.pop("base")
        print("%-3s est=%6.3f GiB core=%6.3f GiB dev=%+.3f %s  base=%-7s %s" % (
            k, g, c, g - TARGET, "OK" if ok else "OUT", base, ex))
        path = os.path.join(OUT_DIR, "e_h1_%s.txt" % k)
        if args.write:
            write_typefile(a, base, path)
        meta["arms"][k] = {"est_gib": g, "core_gib": c, "dev": g - TARGET, "within_tol": ok,
                           "base_type": base, "params": ex, "typefile": path,
                           "types": {n: a[n] for n in a}}
    if args.write:
        os.makedirs(os.path.dirname(META), exist_ok=True)
        json.dump(meta, open(META, "w"), indent=1, ensure_ascii=False)
        print("wrote meta ->", META)


if __name__ == "__main__":
    main()
