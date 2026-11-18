"""E-G2/G3: 由 GGUF 张量清单生成 llama-quantize 的 --tensor-type-file 并估算体积。

张量清单来自 e_g_gguf_header.py --dump (真实 GGUF 头, 不是猜的):
  results/e_g2_tensors.txt  MiniCPM-V-4.7-35B-A3B (arch=qwen35moe, 40 层, 256 专家, 融合三维专家张量)
  results/e_g3_tensors.txt  Xing4.0-29B-A4B      (arch=xing4_0,  64 专家, blk.0-1 dense)

配置 (对齐 G1 的角色模板):
  C      : 专家 Q4_K + 非专家 Q5_K
  C_layer: 路由统计 top-k 层的专家 Q4_K, 其余层 Q3_K + 非专家 Q5_K  (仅 G2)
  D      : C 的类型文件 + 命令行 --imatrix (同一份文件, 不另写)
  R      : 专家 Q3_K + 非专家 Q5_K (+ --imatrix)
另外 G2 的 R_official(A 版) 直接下载现成 mix_q3gud_im, 不在本脚本生成。

用法:
  python e_g_typefiles.py --model g2 --layers 3,7,11,...   # C_layer 顶层名单
  python e_g_typefiles.py --model g2 --usage results/e_g2_route_usage.json --k 10
  python e_g_typefiles.py --model g3
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

DUMP = {"g2": os.path.join(RES, "e_g2_tensors.txt"),
        "g3": os.path.join(RES, "e_g3_tensors.txt"),
        "g5": os.path.join(RES, "e_g5_tensors.txt"),
        "g6": os.path.join(RES, "e_g6_tensors.txt"),
        "g7": os.path.join(RES, "e_g7_tensors.txt")}

# GGML tensor type id -> (block, type_size)
QUANT = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
}
BY_NAME = {v[0]: (v[1], v[2]) for v in QUANT.values()}
FALLBACK = {"Q2_K": "Q4_0", "Q3_K": "Q4_0", "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0"}


def load_tensors(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("gguf") or line.startswith("kv "):
                continue
            name, dims, _tt = line.rstrip("\n").split("\t")
            dims = [int(x) for x in dims.strip("[]").split(",")] if dims.strip("[]") else []
            rows.append((name, dims))
    return rows


def effective(qtype, ncols):
    block, _ = BY_NAME[qtype]
    if block == 1 or ncols % block == 0:
        return qtype
    return FALLBACK.get(qtype, "F16")


def bits(qtype, ncols):
    t = effective(qtype, ncols)
    block, size = BY_NAME[t]
    return size * 8.0 / block


def rules_g2(expert, layers_q4=None):
    """MiniCPM: attn 分 linear(attn_qkv/attn_gate/ssm_*) 与 full(attn_q/k/v)。"""
    out = [
        ("_norm", "F32"),
        (r"ffn_gate_inp\.weight", "F32"),
        (r"\.bias", "F32"),
        (r"ssm_conv1d", "F32"),
        (r"ssm_a", "F32"),
        (r"token_embd\.weight", "Q4_K"),
        (r"^output\.weight", "Q4_K"),
    ]
    if layers_q4:
        alt = "|".join(str(i) for i in layers_q4)
        out.append((r"blk\.(%s)\.ffn_(gate|up|down)_exps\.weight" % alt, "Q4_K"))
    out += [
        (r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", expert),
        (r"blk\.\d+\.ffn_(gate|up|down)_shexp\.weight", "Q5_K"),
        (r"blk\.\d+\.attn_(qkv|q|k|v|gate|output)\.weight", "Q5_K"),
        (r"blk\.\d+\.ssm_(out|alpha|beta)\.weight", "Q5_K"),
    ]
    return out


def rules_g3(expert):
    """Xing: MLA 注意力 + blk.0-1 dense FFN + hc_* 超连接。"""
    return [
        ("_norm", "F32"),
        (r"ffn_gate_inp\.weight", "F32"),
        (r"exp_probs_b\.bias", "F32"),
        (r"hc_", "F16"),
        (r"token_embd\.weight", "Q4_K"),
        (r"^output\.weight", "Q4_K"),
        (r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", expert),
        (r"blk\.\d+\.ffn_(gate|up|down)_shexp\.weight", "Q5_K"),
        (r"blk\.\d+\.ffn_(gate|up|down)\.weight", "Q5_K"),
        (r"blk\.\d+\.attn_(q_a|q_b|kv_a_mqa|output|k_b|v_b)\.weight", "Q5_K"),
    ]


def rules_common(expert):
    """G5/G6/G7 通用角色模板, 与 G2/G3 同义: 专家 Q4_K/Q3_K + 非专家 Q5_K。

    三者的专家都是融合三维张量 (deepseek2 / nemotron_h_moe / raon1), 只能整张量分档,
    所以不再有 G2 的 C_layer 逐层提档。
    """
    return [
        ("_norm", "F32"),
        (r"ffn_gate_inp\.weight", "F32"),
        (r"exp_probs_b\.bias", "F32"),
        (r"token_embd\.weight", "Q4_K"),
        (r"^output\.weight", "Q4_K"),
        (r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", expert),
        (r"blk\.\d+\.ffn_(gate|up|down)_shexp\.weight", "Q5_K"),
    ]


def rules_g5(expert):
    """GLM-4.7-Flash (deepseek2): MLA + 融合专家 + 共享专家。"""
    return rules_common(expert)


def rules_g6(expert):
    """Nemotron-Cascade-2 (nemotron_h_moe): Mamba2+Attention 混合, ssm 小张量保 F16/F32。"""
    r = rules_common(expert)
    return [
        ("_norm", "F32"),
        (r"ffn_gate_inp\.weight", "F32"),
        (r"exp_probs_b\.bias", "F32"),
        (r"ssm_conv1d", "F16"),
        (r"ssm_a", "F32"),
    ] + r[3:]


def rules_g7(expert):
    """Raon-1-mini (raon1): MLA + MoE + mHC/RDA, 按 fork 的 llama-quant 保留清单。"""
    r = rules_common(expert)
    return [
        ("_norm", "F32"),
        (r"ffn_gate_inp\.weight", "F32"),
        (r"exp_probs_b\.bias", "F32"),
        (r"hc_", "F16"),
        (r"ssm_f_a", "F16"),
        (r"ssm_f_b", "F16"),
        (r"ssm_beta", "F16"),
        (r"ssm_rda", "F16"),
        (r"attn_kv_a_mqa", "Q8_0"),
        (r"attn_k_b", "Q8_0"),
        (r"attn_v_b", "Q8_0"),
    ] + r[3:]


def forced_f32(name, dims):
    """llama-quantize 的硬排除: *_norm / ffn_gate_inp / 一维张量 一律 F32。"""
    low = name.lower()
    return len(dims) <= 1 or "_norm" in low or "ffn_gate_inp.weight" in low


def estimate(rows, rules, base):
    total, per_role = 0, {}
    for name, dims in rows:
        n = 1
        for d in dims:
            n *= d
        ncols = dims[0] if dims else 1
        if forced_f32(name, dims):
            q = "F32"
        else:
            q = base
            for pat, t in rules:
                if re.search(pat, name.lower()):
                    q = t
                    break
        nb = n / BY_NAME[effective(q, ncols)][0] * BY_NAME[effective(q, ncols)][1]
        total += nb
        per_role[q] = per_role.get(q, 0) + nb
    return int(total), per_role


def gib(x):
    return x / float(1 << 30)


def top_layers_from_usage(path, k):
    with open(path, encoding="utf-8") as f:
        u = json.load(f)
    counts = u["combined"]["counts"]
    ranked = sorted(range(len(counts)), key=lambda i: (-sum(counts[i]), i))
    return ranked[:k], [sum(counts[i]) for i in ranked[:k]]


def write_rules(path, rules):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join("%s=%s" % (p, t) for p, t in rules) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["g2", "g3", "g5", "g6", "g7"])
    ap.add_argument("--usage", default="", help="路由统计 json (取层聚合 usage 排序 top-k)")
    ap.add_argument("--k", type=int, default=10, help="C_layer 提档层数 (G1 对齐: 40 层的 25%% = 10)")
    ap.add_argument("--layers", default="", help="直接给出 top 层号 (逗号分隔), 优先于 --usage")
    ap.add_argument("--out-dir", default=RES)
    args = ap.parse_args()

    dump = DUMP[args.model]
    if not os.path.exists(dump):
        raise SystemExit("缺张量清单 %s (先跑 e_g_gguf_header.py --dump)" % dump)
    rows = load_tensors(dump)
    print("[dump] %s 张量 %d 个" % (args.model, len(rows)))

    if args.model == "g2":
        variants = {"c": (rules_g2("Q4_K"), "Q4_K"),
                    "r": (rules_g2("Q3_K"), "Q3_K")}
        if args.layers:
            layers = [int(x) for x in args.layers.split(",") if x.strip()]
        elif args.usage:
            layers, usage_sum = top_layers_from_usage(args.usage, args.k)
            print("[layer] top-%d: %s  (层聚合 usage=%s)"
                  % (len(layers), layers, usage_sum))
        else:
            layers = []
            print("[layer] 未给路由统计, 跳过 C_layer")
        if layers:
            variants["clayer"] = (rules_g2("Q3_K", layers), "Q3_K")
    elif args.model == "g3":
        variants = {"c": (rules_g3("Q4_K"), "Q4_K"),
                    "r": (rules_g3("Q3_K"), "Q3_K")}
    else:
        rf = {"g5": rules_g5, "g6": rules_g6, "g7": rules_g7}[args.model]
        variants = {"c": (rf("Q4_K"), "Q5_K"),
                    "r": (rf("Q3_K"), "Q5_K")}

    meta = {"model": args.model, "dump": dump, "variants": {}}
    for key, (rules, base) in variants.items():
        est, per_role = estimate(rows, rules, base)
        out = os.path.join(args.out_dir, "e_%s_types_%s.txt" % (args.model, key))
        write_rules(out, rules)
        meta["variants"][key] = {"file": out, "base": base, "est_bytes": est,
                                 "est_gib": round(gib(est), 2), "per_role_bytes": per_role}
        print("[%s] %s %.2f GiB  (base=%s)" % (args.model, key, gib(est), base))
        for role, nb in sorted(per_role.items(), key=lambda kv: -kv[1]):
            print("      %-6s %.2f GiB" % (role, gib(nb)))
    meta["note"] = ("C_layer = 路由统计 top-k 层专家 Q4_K, 其余层 Q3_K; "
                    "D 复用 C 类型文件 + --imatrix; G2 的 R_official 为官方 mix_q3gud_im(现成下载)")
    with open(os.path.join(args.out_dir, "e_%s_typefiles_meta.json" % args.model), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[write] %s/e_%s_typefiles_meta.json" % (args.out_dir, args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
