"""E-G2: MiniCPM-V-4.7-35B-A3B 文本主干的路由使用率统计 (只 prefill)。

与 e_g1_route_stats.py 同口径 (每域 8 条 / 256 token / top-8), 差异只在加载方式:
  - model_type=minicpmv4_7 在 transformers 5.5 里没有实现, 但文本主干是 qwen3_5_moe_text;
    于是直接构造 Qwen3_5MoeTextModel(text_config), 从 safetensors 里按
    "model.language_model." 前缀取 692 个张量喂进去 (键名已逐一对齐, 无缺失/多余)。
  - Qwen3_5MoeTextModel 的 _can_record_outputs 注册了
    OutputRecorder(Qwen3_5MoeTopKRouter, index=0) -> router_logits,
    传 output_router_logits=True 即得 40 层 [T,256] 概率, topk 取 8。
  - 视觉塔完全不加载 (纯文本路径)。

输出 results/e_g2_route_usage.json (结构同 G1: combined.counts = 每层每专家计数),
按域断点续跑。

用法:
  python e_g2_route_stats.py --per-domain 8 --max-tokens 256
  python e_g2_route_stats.py --limit 1 --domains math    # 冒烟
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")

MODEL_DIR = r"E:\CUDA11.8\model\minicpm\safetensors"
OUT = os.path.join(RES, "e_g2_route_usage.json")
PREFIX = "model.language_model."

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import transformers  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig  # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel  # noqa: E402

sys.path.insert(0, HERE)
from e_g_build import heavy_lock  # noqa: E402


def say(m):
    print("%s %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def read_prompts(path, limit):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line)["text"])
            if limit and len(items) >= limit:
                break
    return items


def load_cfg(model_dir):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    tc = dict(raw["text_config"])
    tc.pop("model_type", None)
    cfg = Qwen3_5MoeTextConfig(**tc)
    meta = {k: tc[k] for k in tc}
    meta["architectures"] = raw.get("architectures", [])
    meta["transformers_version"] = transformers.__version__
    meta["torch_version"] = torch.__version__
    meta["model_dir"] = os.path.abspath(model_dir)
    return cfg, meta


def load_text_model(model_dir, cfg):
    with torch.device("meta"):
        model = Qwen3_5MoeTextModel(cfg)
    model.to_empty(device="cpu")
    model = model.to(torch.bfloat16)
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({v for k, v in idx.items() if k.startswith(PREFIX)})
    say("[load] %d shards ..." % len(shards))
    t0 = time.time()
    for i, sh in enumerate(shards, 1):
        batch = {}
        with safe_open(os.path.join(model_dir, sh), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(PREFIX):
                    batch[k[len(PREFIX):]] = f.get_tensor(k)
        model.load_state_dict(batch, strict=False)
        del batch
        if i % 4 == 0:
            say("  shard %d/%d  %.1f min" % (i, len(shards), (time.time() - t0) / 60))
    model.eval()
    say("[load] done %.1f min" % ((time.time() - t0) / 60))
    return model


def load_tok(model_dir):
    try:
        return AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:  # noqa: BLE001
        say("[tok] AutoTokenizer 失败(%r), 用 tokenizer.json" % (e,))
        return PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))


def load_usage(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def dump_usage(path, cfg, domains):
    combined = np.zeros((cfg["num_hidden_layers"], cfg["num_experts"], ), dtype=np.int64)
    tokens = 0
    for d in domains.values():
        combined += np.asarray(d["counts"], dtype=np.int64)
        tokens += int(d["tokens"])
    payload = {"config": cfg, "domains": domains,
               "combined": {"counts": combined.tolist(), "tokens": tokens}}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)
    return payload


def run_domain(model, prompts, tok, n_layers, n_exp, top_k, max_tokens):
    counts = np.zeros((n_layers, n_exp), dtype=np.int64)
    total = 0
    with torch.no_grad():
        for text in prompts:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
            ids = enc["input_ids"]
            total += int(ids.shape[1])
            out = model(input_ids=ids, output_router_logits=True,
                        use_cache=False, return_dict=True)
            logits = out.router_logits
            if logits is None or len(logits) != n_layers:
                raise RuntimeError("router_logits 形状异常: %r"
                                   % (None if logits is None else len(logits)))
            for li, p in enumerate(logits):
                idx = p.topk(top_k, dim=-1).indices.reshape(-1).to(torch.int64)
                counts[li] += torch.bincount(idx, minlength=n_exp).numpy()[:n_exp]
    return counts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--domains", default="math,code,general")
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
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
    for d, (p, n, a) in plan.items():
        say("[plan] %-8s %d prompts (file %d) <- %s" % (d, n, a, p))
    if args.dry_run:
        return 0

    cfg_obj, meta = load_cfg(args.model_dir)
    n_layers = int(cfg_obj.num_hidden_layers)
    n_exp = int(cfg_obj.num_experts)
    top_k = int(cfg_obj.num_experts_per_tok)
    existing = load_usage(args.out)
    have = (existing or {}).get("domains", {}) if not args.force else {}
    meta["router_source"] = "output_router_logits(Qwen3_5MoeTopKRouter)"
    meta["note"] = "文本主干单独加载, 视觉塔未加载; 与 G1 同口径 (8 prompt/域, 256 token)"

    model = None
    tok = None
    lk = heavy_lock("route_stats")
    lk.__enter__()
    try:
        for d, (path, n_use, _a) in plan.items():
            if d in have and have[d].get("tokens", 0) > 0 and not args.force:
                say("[skip] %s 已有 %d tokens" % (d, have[d]["tokens"]))
                continue
            if n_use == 0:
                say("[skip] %s 无 prompt (%s)" % (d, path))
                continue
            try:
                if model is None:
                    tok = load_tok(args.model_dir)
                    model = load_text_model(args.model_dir, cfg_obj)
                    say("[load] layers=%d experts=%d top_k=%d" % (n_layers, n_exp, top_k))
                prompts = read_prompts(path, n_use)
                t0 = time.time()
                counts, nt = run_domain(model, prompts, tok, n_layers, n_exp, top_k,
                                        args.max_tokens)
                have[d] = {"counts": counts.tolist(), "tokens": int(nt), "prompts": len(prompts),
                           "elapsed_sec": round(time.time() - t0, 1)}
                dump_usage(args.out, meta, have)
                say("[done] %s %d prompts / %d tokens / %.1f s"
                    % (d, len(prompts), nt, time.time() - t0))
            except Exception:
                say("[fail] %s" % d)
                traceback.print_exc()
                continue
    finally:
        lk.__exit__(None, None, None)
    dump_usage(args.out, meta, have)
    say("[write] %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
