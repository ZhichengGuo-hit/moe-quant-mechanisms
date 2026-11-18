"""E-G1: gemma-4-26B-A4B 的 MoE 路由使用率统计 (只做 prefill, 不 generate)。

按域 (math/code/general) 取校准 prompt, 每条截断到 --max-tokens 个 token, 跑一次
forward, 用输出里的 router 概率做 top-k, 累加每个 (层, 专家) 的选中次数。

加载方式 (已核对 transformers 5.5 的 gemma4 源码):
  - AutoModelForCausalLM 对 model_type=gemma4 实际得到 Gemma4ForConditionalGeneration,
    其 text 主干在 model.model.language_model (Gemma4TextModel)。
  - Gemma4TextModel._can_record_outputs 注册了
    OutputRecorder(Gemma4TextRouter, index=0) -> 键名 router_logits;
    直接调用语言模型并传 output_router_logits=True 即可拿到 30 层概率张量 [T, 128]。
    (Gemma4ForConditionalGeneration.forward 会把 router_logits 丢掉, 所以不能走顶层。)
  - 若该路径拿不到, 回退到 Gemma4TextRouter 的 forward hook, 取返回元组第 3 项
    (top_k_index, 已按 router_probabilities 排序, 与 topk(probs) 等价)。

输出: results/e_g1_route_usage.json
  {"config": {...}, "domains": {"math": {"counts": [[...128]*30], "tokens": N}, ...},
   "combined": {...}}
支持断点续跑: 每个域算完立即落盘, 已有域默认跳过 (--force 可重算); 单域失败不影响其它域。

用法:
  python e_g1_route_stats.py --domains math,code,general --per-domain 32 --max-tokens 512
  python e_g1_route_stats.py --limit 2 --domains math          # 冒烟
  python e_g1_route_stats.py --dry-run
"""
import argparse
import json
import os
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")

DEFAULT_MODEL = r"E:\CUDA11.8\model\gemma4\bf16_safetensors"
DEFAULT_OUT = os.path.join(RES, "e_g1_route_usage.json")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def read_prompts(path, limit):
    """读 jsonl, 取 text 字段; limit<=0 表示不限。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line)["text"])
            if limit and len(items) >= limit:
                break
    return items


def get_text_model(model):
    """从 Gemma4ForConditionalGeneration / Gemma4ForCausalLM 里取出语言模型。"""
    inner = getattr(model, "model", None)
    language_model = getattr(inner, "language_model", None)
    if language_model is not None:
        return language_model
    if inner is not None and hasattr(inner, "layers"):
        return inner
    raise RuntimeError(f"无法从 {type(model).__name__} 定位 text 主干")


def text_config_dict(model_dir):
    """读 config.json 的 text_config, 汇总成本脚本需要的元数据。"""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    fields = [
        "model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
        "moe_intermediate_size", "num_experts", "top_k_experts", "vocab_size",
        "num_attention_heads", "head_dim", "global_head_dim", "num_key_value_heads",
        "num_global_key_value_heads", "layer_types", "tie_word_embeddings",
        "hidden_size_per_layer_input", "sliding_window",
    ]
    out = {k: getattr(tcfg, k) for k in fields if hasattr(tcfg, k)}
    out["architectures"] = list(getattr(cfg, "architectures", []) or [])
    out["model_dir"] = os.path.abspath(model_dir)
    out["transformers_version"] = transformers.__version__
    out["torch_version"] = torch.__version__
    return out


def capture_topk_from_output(text_model, input_ids, n_layers, top_k):
    """优先返回 out.router_logits (每层 [T, n_exp] 概率) 的 top-k。"""
    try:
        out = text_model(input_ids=input_ids, output_router_logits=True,
                         use_cache=False, return_dict=True)
    except TypeError:  # 该版本不接受 output_router_logits 时回退到 hook
        return None
    logits = getattr(out, "router_logits", None)
    if logits is None or len(logits) != n_layers:
        return None
    return [p.topk(top_k, dim=-1).indices for p in logits]


def attach_router_hooks(text_model, store):
    """回退方案: 在 Gemma4TextRouter 上挂 forward hook, 取 top_k_index (元组第 3 项)。"""
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            store[layer_idx] = output[2]
        return hook

    routers = [m for m in text_model.modules() if type(m).__name__ == "Gemma4TextRouter"]
    for i, mod in enumerate(routers):
        handles.append(mod.register_forward_hook(make_hook(i)))
    return handles, len(routers)


def run_domain(text_model, prompts, tok, n_layers, n_exp, top_k, max_tokens, router_state):
    counts = np.zeros((n_layers, n_exp), dtype=np.int64)
    total_tokens = 0
    used_primary = True
    with torch.no_grad():
        for text in prompts:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
            input_ids = enc["input_ids"]
            total_tokens += int(input_ids.shape[1])

            router_state["store"].clear()
            topk_list = capture_topk_from_output(text_model, input_ids, n_layers, top_k)
            if topk_list is None:  # 回退: hook 路径
                used_primary = False
                text_model(input_ids=input_ids, use_cache=False, return_dict=True)
                if len(router_state["store"]) != n_layers:
                    raise RuntimeError(
                        f"router hook 只收到 {len(router_state['store'])} 层, 期望 {n_layers}")
                topk_list = [router_state["store"][i] for i in range(n_layers)]

            for layer_idx, idx in enumerate(topk_list):
                flat = idx.reshape(-1).to(torch.int64)
                counts[layer_idx] += torch.bincount(flat, minlength=n_exp).numpy()[:n_exp]
    return counts, total_tokens, used_primary


def load_usage(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def dump_usage(path, cfg, domains):
    combined = np.zeros((cfg["num_hidden_layers"], cfg["num_experts"]), dtype=np.int64)
    tokens = 0
    for d in domains.values():
        combined += np.asarray(d["counts"], dtype=np.int64)
        tokens += int(d["tokens"])
    payload = {
        "config": cfg,
        "domains": domains,
        "combined": {"counts": combined.tolist(), "tokens": tokens},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--domains", default="math,code,general")
    ap.add_argument("--per-domain", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="每个域最多用多少条 prompt (冒烟用)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--force", action="store_true", help="忽略已有结果, 重算所有域")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划, 不加载模型/不写文件")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    plan = {}
    for d in domains:
        path = os.path.join(DATA, f"{d}_calib.jsonl")
        n_avail = 0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                n_avail = sum(1 for line in f if line.strip())
        n_use = min(args.per_domain, n_avail)
        if args.limit:
            n_use = min(n_use, args.limit)
        plan[d] = (path, n_use, n_avail)

    print(f"[plan] model     : {args.model_dir}")
    print(f"[plan] out       : {args.out}")
    print(f"[plan] dtype     : {args.dtype}  max_tokens={args.max_tokens}")
    for d, (path, n_use, n_avail) in plan.items():
        print(f"[plan] {d:8s}: {n_use} prompts (file has {n_avail})  <- {path}")

    if args.dry_run:
        print("[dry-run] 不加载模型, 不写结果")
        return 0

    existing = load_usage(args.out)
    if existing and not args.force:
        cfg = existing["config"]
        cfg["model_dir"] = os.path.abspath(args.model_dir)
        cfg["transformers_version"] = transformers.__version__
        cfg["torch_version"] = torch.__version__
        have = existing.get("domains", {})
        print(f"[resume] 已有域: {sorted(have)}")
    else:
        cfg = text_config_dict(args.model_dir)
        have = {}
    for d in domains:
        if d not in have:
            have[d] = {"counts": np.zeros(
                (cfg["num_hidden_layers"], cfg["num_experts"]), dtype=np.int64).tolist(),
                "tokens": 0, "prompts": 0}

    text_model = None
    for d, (path, n_use, _n_avail) in plan.items():
        if have[d].get("tokens", 0) > 0 and not args.force:
            print(f"[skip] {d}: 已有 {have[d]['tokens']} tokens")
            continue
        if n_use == 0:
            print(f"[skip] {d}: prompt 文件不存在或为空 ({path})")
            continue
        try:
            if text_model is None:
                print("[load] tokenizer ...")
                tok = AutoTokenizer.from_pretrained(args.model_dir)
                print("[load] model (bf16, cpu) ...")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_dir,
                    torch_dtype=getattr(torch, args.dtype),
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
                model.eval()
                text_model = get_text_model(model)
                n_layers = int(cfg["num_hidden_layers"])
                n_exp = int(cfg["num_experts"])
                top_k = int(cfg["top_k_experts"])
                routers = [m for m in text_model.modules()
                           if type(m).__name__ == "Gemma4TextRouter"]
                print(f"[load] text={type(text_model).__name__} layers={n_layers} "
                      f"experts={n_exp} top_k={top_k} routers={len(routers)}")
                if len(routers) != n_layers:
                    raise RuntimeError(f"router 数 {len(routers)} != 层数 {n_layers}")
                router_state = {"store": {}}
                attach_router_hooks(text_model, router_state["store"])

            prompts = read_prompts(path, n_use)
            t0 = time.time()
            counts, ntokens, used_primary = run_domain(
                text_model, prompts, tok, n_layers, n_exp, top_k,
                args.max_tokens, router_state)
            src = "output_router_logits" if used_primary else "router_hook"
            cfg["router_source"] = src
            have[d] = {"counts": counts.tolist(), "tokens": int(ntokens),
                       "prompts": len(prompts), "elapsed_sec": round(time.time() - t0, 1)}
            dump_usage(args.out, cfg, have)
            print(f"[done] {d}: {len(prompts)} prompts, {ntokens} tokens, "
                  f"{time.time() - t0:.1f}s, source={src}")
        except Exception:
            print(f"[fail] {d}:")
            traceback.print_exc()
            continue

    dump_usage(args.out, cfg, have)
    print(f"[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
