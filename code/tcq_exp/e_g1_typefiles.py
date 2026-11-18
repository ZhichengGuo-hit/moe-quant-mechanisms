"""E-G1: 为 gemma-4-26B-A4B 生成 llama-quantize 的 --tensor-type-file (C/R) 并估算体积。

张量名已逐条核对 llama.cpp 源码, 不是猜的:
  src/llama-arch.cpp  (LLM_ARCH_GEMMA4 的 tensor mapping / LLM_TENSOR_NAMES)
  src/models/gemma4.cpp (load_tensors 里 create_tensor 的名字与形状)
  conversion/gemma.py + gguf-py/gguf/tensor_mapping.py (HF -> GGUF 的映射)

  专家张量 (每层一个三维张量, 所有 128 个专家在同一张量内):
    blk.N.ffn_gate_up_exps.weight  {n_embd, 2*n_ff_exp, n_expert}   <- HF experts.gate_up_proj (融合)
    blk.N.ffn_down_exps.weight     {n_ff_exp, n_embd, n_expert}     <- HF experts.down_proj
    (非融合变体 ffn_gate_exps / ffn_up_exps 只在转换器拆分时才存在; gemma4 的 HF 权重本身
     就是融合的 gate_up_proj, 所以 GGUF 实际是 ffn_gate_up_exps)
  非专家角色:
    token_embd, output_norm, output(可选, tie_word_embeddings=true 时通常缺失)
    blk.N.attn_norm / attn_q / attn_k / attn_v(仅 25 个滑窗层, full-attn 层共享 k)
    blk.N.attn_output / attn_q_norm / attn_k_norm / post_attention_norm / layer_output_scale
    blk.N.ffn_norm / ffn_gate / ffn_up / ffn_down / post_ffw_norm
    blk.N.pre_ffw_norm_2 / post_ffw_norm_1 / post_ffw_norm_2
    blk.N.ffn_gate_inp(+.scale) / ffn_down_exps.scale
    rope_freqs(全局, full-attn 层)

tensor-type-file 语义 (src/llama-quant.cpp 的 parse + llama_tensor_get_type):
  - 文件按空白切分, 每个 token 必须是 pattern=type; 不支持 # 注释 (会被当成非法 token 直接报错)
  - pattern 转小写后和完整张量名做 std::regex_search (ECMAScript 正则, 不是 glob!), 首个命中生效
  - 只在命令行基础 ftype 是量化类型时才生效
  - *_norm.weight / ffn_gate_inp.weight / 一维张量 被硬性排除, 写 F32 也只是声明意图

关键限制 (脚本会打印):
  所有专家共用一个三维张量, 因此 --tensor-type-file 无法"逐专家"分档;
  C 退化为"全体专家 Q4_K", R 为"全体专家 Q3_K"; 逐专家的 top-25% 清单写进 meta,
  供 tcmoe 补丁表 (q4_k_rows) 使用。
  另外 ffn_down_exps 的 ne[0]=n_ff_exp=704, 不是 256 的倍数, llama-quantize 会把
  Q3_K -> Q4_0、Q4_K -> Q5_0 自动降级, 体积估算已按降级后的实际类型计算。

输出: results/e_g1_types_c.txt, results/e_g1_types_r.txt, results/e_g1_typefiles_meta.json
用法: python e_g1_typefiles.py --usage results/e_g1_route_usage.json
      python e_g1_typefiles.py --usage <fake.json> --dry-run
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

DEFAULT_USAGE = os.path.join(RES, "e_g1_route_usage.json")
DEFAULT_OUT_C = os.path.join(RES, "e_g1_types_c.txt")
DEFAULT_OUT_CP = os.path.join(RES, "e_g1_types_cprime.txt")
DEFAULT_OUT_R = os.path.join(RES, "e_g1_types_r.txt")
DEFAULT_META = os.path.join(RES, "e_g1_typefiles_meta.json")
TARGET_GIB = 13.5

# GGML_QUANT_SIZES: (block, type_size) -> bpw = type_size*8/block
QUANT = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "Q5_1": (32, 24),
    "Q5_0": (32, 22),
    "Q4_1": (32, 20),
    "Q4_0": (32, 18),
    "Q6_K": (256, 210),
    "Q5_K": (256, 176),
    "Q4_K": (256, 144),
    "Q3_K": (256, 110),
    "Q2_K": (256, 84),
}
# llama-quant.cpp tensor_type_fallback: 256-block 类型遇到 ne[0]%256!=0 时的降级
FALLBACK = {"Q2_K": "Q4_0", "Q3_K": "Q4_0", "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0"}

VERIFIED_EXPERTS = [
    ("blk.N.ffn_gate_up_exps.weight", "{n_embd, 2*n_ff_exp, n_expert}", "融合 gate+up"),
    ("blk.N.ffn_down_exps.weight", "{n_ff_exp, n_embd, n_expert}", ""),
]
VERIFIED_NON_EXPERT = [
    "token_embd.weight", "output_norm.weight", "output.weight(可选)",
    "blk.N.attn_norm.weight", "blk.N.attn_q.weight", "blk.N.attn_k.weight",
    "blk.N.attn_v.weight(仅25个滑窗层)", "blk.N.attn_output.weight",
    "blk.N.attn_q_norm.weight", "blk.N.attn_k_norm.weight",
    "blk.N.post_attention_norm.weight", "blk.N.layer_output_scale.weight",
    "blk.N.ffn_norm.weight", "blk.N.ffn_gate.weight", "blk.N.ffn_up.weight",
    "blk.N.ffn_down.weight", "blk.N.post_ffw_norm.weight",
    "blk.N.pre_ffw_norm_2.weight", "blk.N.post_ffw_norm_1.weight",
    "blk.N.post_ffw_norm_2.weight", "blk.N.ffn_gate_inp.weight",
    "blk.N.ffn_gate_inp.scale", "blk.N.ffn_down_exps.scale", "rope_freqs.weight",
]


def bpw(type_name):
    block, size = QUANT[type_name]
    return size * 8.0 / block


def effective_type(type_name, ncols):
    """复刻 tensor_type_fallback: 256 块类型遇到 ne[0] 不整除时降级。"""
    block, _ = QUANT[type_name]
    if ncols % block == 0:
        return type_name
    return FALLBACK.get(type_name, "F16")


def load_cfg(usage, cfg_path):
    if cfg_path:
        with open(cfg_path, encoding="utf-8") as f:
            raw = json.load(f)
        src = raw.get("text_config", raw)
        keys = ["num_hidden_layers", "hidden_size", "intermediate_size",
                "moe_intermediate_size", "num_experts", "top_k_experts", "vocab_size",
                "num_attention_heads", "head_dim", "global_head_dim",
                "num_key_value_heads", "num_global_key_value_heads",
                "layer_types", "tie_word_embeddings"]
        return {k: src[k] for k in keys if k in src}
    return usage["config"]


def tensor_plan(cfg):
    """返回 [(name, n_elem, ncols, kind)], kind 决定目标类型。"""
    n_embd = cfg["hidden_size"]
    n_layer = cfg["num_hidden_layers"]
    n_exp = cfg["num_experts"]
    n_ff_exp = cfg["moe_intermediate_size"]
    n_ff = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    n_head = cfg["num_attention_heads"]
    hd = cfg["head_dim"]
    ghd = cfg["global_head_dim"]
    n_kv = cfg["num_key_value_heads"]
    n_gkv = cfg["num_global_key_value_heads"]
    layer_types = cfg["layer_types"]

    rows = [("token_embd.weight", vocab * n_embd, n_embd, "embd"),
            ("output_norm.weight", n_embd, n_embd, "norm"),
            ("rope_freqs.weight", ghd // 2, ghd // 2, "scale")]
    for i, lt in enumerate(layer_types):
        full = lt == "full_attention"
        hd_i = ghd if full else hd
        kv_i = n_gkv if full else n_kv
        q_dim = n_head * hd_i
        kv_dim = kv_i * hd_i
        rows += [
            (f"blk.{i}.attn_norm.weight", n_embd, n_embd, "norm"),
            (f"blk.{i}.attn_q.weight", n_embd * q_dim, n_embd, "attn"),
            (f"blk.{i}.attn_k.weight", n_embd * kv_dim, n_embd, "attn"),
            (f"blk.{i}.attn_output.weight", q_dim * n_embd, q_dim, "attn"),
            (f"blk.{i}.attn_q_norm.weight", hd_i, hd_i, "norm"),
            (f"blk.{i}.attn_k_norm.weight", hd_i, hd_i, "norm"),
            (f"blk.{i}.post_attention_norm.weight", n_embd, n_embd, "norm"),
            (f"blk.{i}.layer_output_scale.weight", 1, 1, "scale"),
            (f"blk.{i}.ffn_norm.weight", n_embd, n_embd, "norm"),
            (f"blk.{i}.ffn_gate.weight", n_embd * n_ff, n_embd, "ffn"),
            (f"blk.{i}.ffn_up.weight", n_embd * n_ff, n_embd, "ffn"),
            (f"blk.{i}.ffn_down.weight", n_ff * n_embd, n_ff, "ffn"),
            (f"blk.{i}.post_ffw_norm.weight", n_embd, n_embd, "norm"),
            (f"blk.{i}.pre_ffw_norm_2.weight", n_embd, n_embd, "norm"),
            (f"blk.{i}.post_ffw_norm_1.weight", n_embd, n_embd, "norm"),
            (f"blk.{i}.post_ffw_norm_2.weight", n_embd, n_embd, "norm"),
            (f"blk.{i}.ffn_gate_inp.weight", n_embd * n_exp, n_embd, "router"),
            (f"blk.{i}.ffn_gate_inp.scale", n_embd, n_embd, "scale"),
            (f"blk.{i}.ffn_down_exps.scale", n_exp, n_exp, "scale"),
            (f"blk.{i}.ffn_gate_up_exps.weight", n_embd * 2 * n_ff_exp * n_exp, n_embd, "expert"),
            (f"blk.{i}.ffn_down_exps.weight", n_ff_exp * n_embd * n_exp, n_ff_exp, "expert"),
        ]
        if not full:  # full-attn 层没有 v_proj (attention_k_eq_v)
            rows.append((f"blk.{i}.attn_v.weight", n_embd * kv_dim, n_embd, "attn"))
    return rows


def pick_type(kind, variant, non_expert="Q5_K"):
    if kind in ("norm", "router", "scale"):
        return "F32"
    if kind in ("embd", "output"):
        return "Q4_K"
    if kind == "expert":
        return variant
    return non_expert  # attn / ffn


def rules_for(variant, non_expert="Q5_K"):
    """(regex pattern, type)。已按"先具体后宽泛"排序 (首个命中生效)。"""
    expert = variant
    return [
        (r"_norm", "F32"),  # 覆盖 *_norm.weight / pre_ffw_norm_2 / post_ffw_norm_1 等
        (r"ffn_gate_inp\.weight", "F32"),
        (r"\.scale", "F32"),
        (r"rope_freqs\.weight", "F32"),
        (r"layer_output_scale\.weight", "F32"),
        (r"token_embd\.weight", "Q4_K"),
        (r"^output\.weight", "Q4_K"),  # 锚定开头, 避免误命中 attn_output.weight
        (r"blk\.\d+\.ffn_gate_up_exps\.weight", expert),
        (r"blk\.\d+\.ffn_down_exps\.weight", expert),
        (r"blk\.\d+\.ffn_gate_exps\.weight", expert),
        (r"blk\.\d+\.ffn_up_exps\.weight", expert),
        (r"blk\.\d+\.attn_q\.weight", non_expert),
        (r"blk\.\d+\.attn_k\.weight", non_expert),
        (r"blk\.\d+\.attn_v\.weight", non_expert),
        (r"blk\.\d+\.attn_output\.weight", non_expert),
        (r"blk\.\d+\.ffn_gate\.weight", non_expert),
        (r"blk\.\d+\.ffn_up\.weight", non_expert),
        (r"blk\.\d+\.ffn_down\.weight", non_expert),
    ]


def rank_experts(counts, top_frac):
    n_exp = len(counts[0])
    top_n = max(1, int(math.ceil(top_frac * n_exp)))
    tiers = []
    for layer, row in enumerate(counts):
        order = sorted(range(n_exp), key=lambda e: (-row[e], e))
        tiers.append(order[:top_n])
    return tiers, top_n


def estimate(plan, variant, non_expert="Q5_K"):
    total = 0
    per_kind = {}
    for name, n_elem, ncols, kind in plan:
        want = pick_type(kind, variant, non_expert)
        got = effective_type(want, ncols)
        nbytes = n_elem / QUANT[got][0] * QUANT[got][1]
        total += nbytes
        per_kind.setdefault(kind, {"bytes": 0, "types": {}})
        per_kind[kind]["bytes"] += nbytes
        key = f"{want}->{got}" if want != got else want
        per_kind[kind]["types"][key] = per_kind[kind]["types"].get(key, 0) + 1
    return int(total), per_kind


def gib(nbytes):
    return nbytes / float(1 << 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", default=DEFAULT_USAGE)
    ap.add_argument("--config", default="", help="可选: 直接读 config.json (默认取 usage 的 config)")
    ap.add_argument("--domain", default="combined",
                    help="用哪个域的使用率排序专家: combined/math/code/general")
    ap.add_argument("--top-frac", type=float, default=0.25)
    ap.add_argument("--out-c", default=DEFAULT_OUT_C)
    ap.add_argument("--out-cprime", default=DEFAULT_OUT_CP)
    ap.add_argument("--out-r", default=DEFAULT_OUT_R)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--dry-run", action="store_true", help="只打印, 不写文件")
    args = ap.parse_args()

    with open(args.usage, encoding="utf-8") as f:
        usage = json.load(f)
    cfg = load_cfg(usage, args.config)
    src = usage.get("combined") if args.domain == "combined" else usage["domains"].get(args.domain)
    if src is None:
        raise SystemExit(f"usage 里没有域 '{args.domain}'")
    counts = src["counts"]
    tiers, top_n = rank_experts(counts, args.top_frac)

    plan = tensor_plan(cfg)
    n_total = sum(r[1] for r in plan)
    est_c, kinds_c = estimate(plan, "Q4_K")
    est_cp, kinds_cp = estimate(plan, "Q4_K", "Q6_K")
    est_r, kinds_r = estimate(plan, "Q3_K")

    print("[tensor names] 专家张量 (来自 llama.cpp 源码, 非猜测):")
    for name, shape, note in VERIFIED_EXPERTS:
        print(f"  {name:34s} {shape:32s} {note}")
    print("[tensor names] 非专家角色:")
    for name in VERIFIED_NON_EXPERT:
        print(f"  {name}")
    print(f"[plan] 层={cfg['num_hidden_layers']} 专家={cfg['num_experts']} "
          f"top_k={cfg['top_k_experts']} n_ff_exp={cfg['moe_intermediate_size']} "
          f"张量元素合计={n_total:,}")
    print(f"[tier] domain={args.domain} top_frac={args.top_frac} -> 每层 {top_n} 个专家")
    print(f"[tier] layer0 前 8: {tiers[0][:8]}")

    if cfg["moe_intermediate_size"] % 256 != 0:
        print(f"[warn] ffn_down_exps 的 ne[0]={cfg['moe_intermediate_size']} 不是 256 的倍数; "
              f"Q3_K->Q4_0 / Q4_K->Q5_0 自动降级, 估算已按降级后类型计算")
    print("[blocker] GGUF 专家张量是单个三维张量 (ffn_gate_up_exps / ffn_down_exps), "
          "--tensor-type-file 只能按张量名指定, 无法逐专家分档;")
    print("          C 因此退化为 '全体专家 Q4_K', R 为 '全体专家 Q3_K'; "
          "top-25% 清单已写入 meta, 供 tcmoe 补丁表 (q4_k_rows) 使用")

    print(f"[est] C: {gib(est_c):.2f} GiB  (目标 {TARGET_GIB})")
    print(f"[est] C': {gib(est_cp):.2f} GiB  (非专家 Q6_K, 专家 Q4_K)")
    print(f"[est] R: {gib(est_r):.2f} GiB")
    dev = abs(gib(est_c) - TARGET_GIB) / TARGET_GIB * 100
    if dev > 5:
        print(f"[WARN] C 估算与目标 {TARGET_GIB} GiB 偏离 {dev:.1f}% (>5%)")

    rules_c = rules_for("Q4_K")
    rules_cp = rules_for("Q4_K", "Q6_K")
    rules_r = rules_for("Q3_K")
    text_c = "\n".join(f"{p}={t}" for p, t in rules_c) + "\n"
    text_cp = "\n".join(f"{p}={t}" for p, t in rules_cp) + "\n"
    text_r = "\n".join(f"{p}={t}" for p, t in rules_r) + "\n"

    meta = {
        "usage": os.path.abspath(args.usage),
        "domain": args.domain,
        "top_frac": args.top_frac,
        "top_n": top_n,
        "per_expert_supported": False,
        "per_expert_reason": "GGUF stores all experts in one 3D tensor per layer "
                             "(blk.N.ffn_gate_up_exps.weight / blk.N.ffn_down_exps.weight); "
                             "tensor-type-file matches tensor names only",
        "expert_tiers": {str(i): tiers[i] for i in range(len(tiers))},
        "verified_expert_tensors": [{"name": n, "shape": s} for n, s, _ in VERIFIED_EXPERTS],
        "verified_non_expert_tensors": VERIFIED_NON_EXPERT,
        "type_file_semantics": {
            "parse": "whitespace-separated pattern=type; no comments",
            "match": "std::regex_search on lowercased full tensor name (ECMAScript regex, not glob)",
            "precedence": "first matching rule wins",
            "requires": "base ftype must be quantized",
            "excluded": ["*_norm.weight", "ffn_gate_inp.weight", "1-D tensors"],
        },
        "variants": {
            "c": {"file": os.path.abspath(args.out_c), "expert_type": "Q4_K",
                  "note": "per-expert tiering not expressible; whole-tensor Q4_K; "
                          "down_exps falls back Q5_0", "est_gib": round(gib(est_c), 3),
                  "est_bytes": est_c},
            "d": {"file": os.path.abspath(args.out_c),
                  "note": "same type file as C; add --imatrix on the command line"},
            "r": {"file": os.path.abspath(args.out_r), "expert_type": "Q3_K",
                  "note": "down_exps falls back Q4_0",
                  "est_gib": round(gib(est_r), 3), "est_bytes": est_r},
            "cprime": {"file": os.path.abspath(args.out_cprime), "expert_type": "Q4_K",
                       "non_expert_type": "Q6_K",
                       "note": "non-expert bit gradient vs C; experts identical to C",
                       "est_gib": round(gib(est_cp), 3), "est_bytes": est_cp},
        },
        "target_gib": TARGET_GIB,
        "est_deviation_pct": round(dev, 2),
        "per_kind_bytes_c": kinds_c,
        "per_kind_bytes_cprime": kinds_cp,
        "rules_c": rules_c,
        "rules_cprime": rules_cp,
        "rules_r": rules_r,
    }

    if args.dry_run:
        print("[dry-run] 将写出的 C 规则:")
        print(text_c)
        print("[dry-run] 将写出的 C' 规则:")
        print(text_cp)
        print("[dry-run] 将写出的 R 规则:")
        print(text_r)
        print(f"[dry-run] meta -> {args.meta}")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out_c)), exist_ok=True)
    with open(args.out_c, "w", encoding="utf-8") as f:
        f.write(text_c)
    with open(args.out_cprime, "w", encoding="utf-8") as f:
        f.write(text_cp)
    with open(args.out_r, "w", encoding="utf-8") as f:
        f.write(text_r)
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[write] {args.out_c}")
    print(f"[write] {args.out_cprime}")
    print(f"[write] {args.out_r}")
    print(f"[write] {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
