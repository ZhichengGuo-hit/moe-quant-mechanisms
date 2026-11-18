"""E-F10a: UD 三档 vs IQ2_XXS 的 per-tensor 类型/精度归因 (纯元数据, 不加载权重数据)。

读 GGUF 头部的张量清单 (name / ggml type / elements / bytes), 计算每个 (层, 角色) 的
实际 bpw, 与 iq2xxs 做逐张量差分, 定位 UD-Q2 的代码侧损失是否来自"特定张量被更激进地
量化", 以及 UD-Q3/Q4 相对 UD-Q2 升级了哪些张量。

输出: results/e_f10a_tensor_types.json
      results/e_f10a_per_layer.csv
用法: python e_f10a_tensor_types.py [--models iq2xxs,ud_q2kxl,ud_q3kxl,ud_q4kxl,q4km]
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, r"E:\CUDA11.8\train\llama.cpp\gguf-py")
from gguf import GGUFReader            # noqa: E402
from gguf.quants import GGML_QUANT_SIZES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

MODELS = {
    "iq2xxs": r"E:\CUDA11.8\model\baselines\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
    "iq3xxs": r"E:\CUDA11.8\model\baselines\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    "ud_q2kxl": r"E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",
    "ud_q3kxl": r"E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
    "ud_q4kxl": r"E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    "q4km": r"E:\CUDA11.8\model\Qwen3.6-35B-A3B\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
}

ROLES = [
    ("attn_q", "attn"), ("attn_k", "attn"), ("attn_v", "attn"),
    ("attn_output", "attn"), ("attn_qkv", "attn"), ("attn_gate", "attn"),
    ("ffn_gate_inp", "router"), ("ffn_gate_exps", "expert"), ("ffn_up_exps", "expert"),
    ("ffn_down_exps", "expert"), ("ffn_gate_shexp", "shared"), ("ffn_up_shexp", "shared"),
    ("ffn_down_shexp", "shared"), ("ffn_gate", "ffn_dense"), ("ffn_up", "ffn_dense"),
    ("ffn_down", "ffn_dense"), ("attn_norm", "norm"), ("ffn_norm", "norm"),
]


def role_of(name):
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    layer = int(m.group(1)) if m else -1
    tail = (m.group(2) if m else name).replace(".weight", "")
    for key, role in ROLES:
        if tail == key:
            return layer, key, role
    if name in ("token_embd.weight",):
        return -1, name, "embd"
    if name.startswith("output"):
        return -1, name, "output"
    return layer, tail, "other"


def bpw_of(t):
    blk, size = GGML_QUANT_SIZES[t.tensor_type]
    return size * 8.0 / blk


def scan(path):
    r = GGUFReader(path)
    rows = []
    for t in r.tensors:
        layer, key, role = role_of(t.name)
        rows.append({"name": t.name, "type": t.tensor_type.name, "bpw": bpw_of(t),
                     "n": int(t.n_elements), "bytes": int(t.n_bytes),
                     "layer": layer, "key": key, "role": role})
    meta = {}
    for k in ("general.architecture", "general.file_type", "general.quantization_version"):
        try:
            f = r.get_field(k)
            if f is not None:
                meta[k] = f.contents()
        except Exception:  # noqa: BLE001
            pass
    return rows, meta


def agg(rows, key_fn):
    out = {}
    for x in rows:
        k = key_fn(x)
        a = out.setdefault(k, {"bits": 0.0, "n": 0, "types": {}})
        a["bits"] += x["bpw"] * x["n"]
        a["n"] += x["n"]
        a["types"][x["type"]] = a["types"].get(x["type"], 0) + x["n"]
    for a in out.values():
        a["bpw"] = round(a["bits"] / a["n"], 4) if a["n"] else 0.0
        a["types"] = dict(sorted(a["types"].items(), key=lambda kv: -kv[1]))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="iq2xxs,ud_q2kxl,ud_q3kxl,ud_q4kxl,q4km")
    ap.add_argument("--base", default="iq2xxs")
    a = ap.parse_args(argv)
    tags = [t.strip() for t in a.models.split(",") if t.strip()]

    per = {}
    meta = {}
    for tag in tags:
        p = MODELS[tag]
        if not os.path.exists(p):
            print(f"[skip] {tag}: {p} 不存在")
            continue
        rows, m = scan(p)
        per[tag] = rows
        meta[tag] = {"path": p, "size_bytes": os.path.getsize(p), **m}
        print(f"[read] {tag}: {len(rows)} tensors, {os.path.getsize(p)/2**30:.2f} GiB")

    out = {"models": {}, "deltas_vs_base": {}, "deltas_q3q4_vs_q2": {}}
    for tag, rows in per.items():
        by_role = agg(rows, lambda x: x["role"])
        by_layer_role = agg(rows, lambda x: (x["layer"], x["role"]))
        hist = {}
        for x in rows:
            hist[x["type"]] = hist.get(x["type"], 0) + 1
        overall_bpw = sum(x["bpw"] * x["n"] for x in rows) / sum(x["n"] for x in rows)
        out["models"][tag] = {
            "meta": meta[tag],
            "overall_bpw": round(overall_bpw, 4),
            "n_tensors": len(rows),
            "type_hist_tensors": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
            "by_role": {k: {"bpw": v["bpw"], "types": v["types"]} for k, v in by_role.items()},
            "by_layer_role": {f"{k[0]}|{k[1]}": {"bpw": v["bpw"], "types": v["types"]}
                              for k, v in by_layer_role.items()},
        }

    base = a.base
    if base in per:
        base_map = {x["name"]: x for x in per[base]}
        for tag, rows in per.items():
            if tag == base:
                continue
            d = []
            for x in rows:
                b = base_map.get(x["name"])
                if not b:
                    continue
                d.append({"name": x["name"], "layer": x["layer"], "role": x["role"],
                          "type": x["type"], "base_type": b["type"],
                          "bpw": round(x["bpw"], 3), "base_bpw": round(b["bpw"], 3),
                          "n": x["n"], "delta_bits_per_weight": round(x["bpw"] - b["bpw"], 4),
                          "delta_mb": round((x["bytes"] - b["bytes"]) / 2**20, 3)})
            d.sort(key=lambda r: r["delta_bits_per_weight"])
            down = [r for r in d if r["delta_bits_per_weight"] < 0]
            lay = agg([{"bpw": r["bpw"], "n": r["n"], "type": r["type"],
                        "layer": r["layer"], "role": r["role"]} for r in d], lambda x: x["role"])
            codedown = {}
            for r in down:
                k = (r["layer"], r["role"])
                codedown.setdefault(str(k), []).append([r["name"], r["base_type"], r["type"],
                                                         r["delta_bits_per_weight"]])
            out["deltas_vs_base"][tag] = {
                "n_shared_tensors": len(d),
                "n_downgraded": len(down),
                "n_upgraded": sum(1 for r in d if r["delta_bits_per_weight"] > 0),
                "role_bpw": {k: v["bpw"] for k, v in lay.items()},
                "down_by_role": {r: sum(1 for x in down if x["role"] == r)
                                 for r in sorted({x["role"] for x in down})},
                "worst_downgrades": down[:12],
                "top_upgrades": d[-12:][::-1],
                "down_detail": codedown,
            }

    for tag in ("ud_q3kxl", "ud_q4kxl"):
        if tag not in per or "ud_q2kxl" not in per:
            continue
        q2 = {x["name"]: x for x in per["ud_q2kxl"]}
        d = []
        for x in per[tag]:
            b = q2.get(x["name"])
            if not b or x["bpw"] == b["bpw"]:
                continue
            d.append({"name": x["name"], "layer": x["layer"], "role": x["role"],
                      "from": b["type"], "to": x["type"],
                      "delta_bits_per_weight": round(x["bpw"] - b["bpw"], 3),
                      "delta_mb": round((x["bytes"] - b["bytes"]) / 2**20, 1)})
        by_role = {}
        for r in d:
            a2 = by_role.setdefault(r["role"], {"n": 0, "delta_mb": 0.0})
            a2["n"] += 1
            a2["delta_mb"] += r["delta_mb"]
        lay = agg([{"bpw": r["delta_bits_per_weight"], "n": 1,
                    "type": r["to"], "layer": r["layer"], "role": r["role"]} for r in d],
                  lambda x: x["layer"])
        out["deltas_q3q4_vs_q2"][tag] = {
            "n_changed": len(d),
            "by_role": {k: {"n": v["n"], "delta_mb": round(v["delta_mb"], 1)}
                        for k, v in by_role.items()},
            "by_layer_delta_bpw": {str(k): v["bpw"] for k, v in sorted(lay.items())},
            "examples": sorted(d, key=lambda r: -r["delta_mb"])[:10],
        }

    p_out = os.path.join(RES, "e_f10a_tensor_types.json")
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    p_csv = os.path.join(RES, "e_f10a_per_layer.csv")
    tags_n = [t for t in tags if t in per]
    idx = {}
    for t in tags_n:
        for k, v in out["models"][t]["by_layer_role"].items():
            idx[(t, k)] = v["bpw"]
    keys = sorted({k for (_, k) in idx},
                  key=lambda s: (int(s.split("|")[0]) if s.split("|")[0].lstrip("-").isdigit()
                                 else -99, s))
    with open(p_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "role"] + tags_n)
        for k in keys:
            lay, role = k.split("|", 1)
            w.writerow([lay, role] + [idx.get((t, k), "") for t in tags_n])
    print(f"\n[saved] {p_out}\n[saved] {p_csv}")

    print(f"\n{'tag':10s} {'GiB':>6s} {'bpw':>7s}  角色 bpw")
    for t in tags_n:
        m = out["models"][t]
        br = m["by_role"]
        s = " ".join(f"{r}={br[r]['bpw']:.2f}" for r in
                     ("attn", "router", "expert", "shared", "norm", "embd", "output") if r in br)
        print(f"{t:10s} {meta[t]['size_bytes']/2**30:6.2f} {m['overall_bpw']:7.3f}  {s}")
        print(f"{'':10s} types: {m['type_hist_tensors']}")

    for t, v in out["deltas_vs_base"].items():
        print(f"\n[{t} vs {base}] 降档 {v['n_downgraded']} / 升档 {v['n_upgraded']} (共 {v['n_shared_tensors']})")
        print(f"  降档角色统计: {v['down_by_role']}")
        print("  最差 6 个降档张量:")
        for r in v["worst_downgrades"][:6]:
            print(f"    blk.{r['layer']:<3d} {r['role']:8s} {r['base_type']:>8s}->{r['type']:>8s} "
                  f"d={r['delta_bits_per_weight']:+.3f} bpp  n={r['n']}")
        print("  最大 4 个升档张量:")
        for r in v["top_upgrades"][:4]:
            print(f"    blk.{r['layer']:<3d} {r['role']:8s} {r['base_type']:>8s}->{r['type']:>8s} "
                  f"d={r['delta_bits_per_weight']:+.3f} bpp  ({r['delta_mb']:+.1f} MB)")

    for t, v in out["deltas_q3q4_vs_q2"].items():
        print(f"\n[{t} vs ud_q2kxl] 变更 {v['n_changed']} 个张量, 按角色 {v['by_role']}")


if __name__ == "__main__":
    main(sys.argv[1:])
