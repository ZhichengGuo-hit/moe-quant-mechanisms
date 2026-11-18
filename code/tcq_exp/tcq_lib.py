"""任务条件化量化实验公共库:
- 构建 INT4 专家引擎 (Qwen3.6-35B-A3B, 16GB GPU + 128GB RAM, CPU offload)
- NLL 评测 (prefill CPU 批量直接INT4路径)
- 校准集激活/路由捕获
- 逐专家精度替换(INT2/INT3/INT4)
"""
import os
import sys
import json
import time
import math

HERE = os.path.dirname(os.path.abspath(__file__))
COSERVING = os.path.join(os.path.dirname(HERE), "coserving")
sys.path.insert(0, COSERVING)
os.environ["MOE_PREFILL_CPU_BATCH"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

from moe_lib import (MOE_MODEL_PATH, STORE_DIR, Int4Store, ExpertCache,
                     build_moe_model, quantize_int4, dequant_int4)
import int4_ext

DATA_DIR = os.path.join(HERE, "data")
RES_DIR = os.path.join(HERE, "results")
STORE_INT4 = STORE_DIR
STORE_INT3 = os.path.join(HERE, "stores", "int3")
STORE_INT2 = os.path.join(HERE, "stores", "int2")
DOMAINS = ("math", "code", "general")
os.makedirs(RES_DIR, exist_ok=True)

LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")


# ---------------------------------------------------------------- 数据
def load_domain(domain, split):
    p = os.path.join(DATA_DIR, f"{domain}_{split}.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- 引擎
_ENGINE = {}


def build_engine(slots=512, store_dir=STORE_INT4, cpu_threads=16):
    key = (slots, store_dir)
    if key in _ENGINE:
        return _ENGINE[key]
    torch.set_num_threads(cpu_threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MOE_MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ext = int4_ext.load_ext(threads=cpu_threads)
    store = Int4Store(store_dir, mmap_mode="c")  # COW: 支持写回覆盖单专家
    cache = ExpertCache(store, device="cuda", dtype=torch.float16, slots=slots,
                        pin_pool=8, trace=False, prefetch=False)
    t0 = time.time()
    model = build_moe_model("gpu", MOE_MODEL_PATH, device="cuda",
                            dtype=torch.float16, store=store, cache=cache,
                            cpu_cache=None, direct_ext=ext)
    print(f"[engine] built {time.time()-t0:.0f}s store={store_dir}", flush=True)
    eng = {"tok": tok, "store": store, "cache": cache, "model": model,
           "ext": ext, "store_dir": store_dir}
    _ENGINE[key] = eng
    return eng


# ---------------------------------------------------------------- 评测
def _batch_encode(tok, texts, max_len, padding="max_length"):
    enc = tok([t for t in texts], return_tensors="pt", padding=padding,
              truncation=True, max_length=max_len, add_special_tokens=False)
    return enc["input_ids"].cuda(), enc["attention_mask"].cuda()


@torch.no_grad()
def eval_nll(model, tok, texts, max_len=64, batch_size=8, verbose=False,
             return_per_seq=False):
    """返回 (nll_sum, n_tok, nll_per_tok); return_per_seq=True 时追加 [(nll_sum_i, n_tok_i)]"""
    tot, ntok = 0.0, 0
    per_seq = [] if return_per_seq else None
    for b0 in range(0, len(texts), batch_size):
        chunk = texts[b0:b0 + batch_size]
        ids, mask = _batch_encode(tok, chunk, max_len)
        t0 = time.time()
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        lg = logits[:, :-1, :].float()
        lab = ids[:, 1:]
        m = mask[:, 1:] > 0
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1),
                               reduction="none")
        loss = loss.reshape(lab.shape)
        nll = (loss * m).sum().item()
        n = int(m.sum().item())
        tot += nll
        ntok += n
        if return_per_seq:
            row_nll = (loss * m).sum(dim=1)
            row_n = m.sum(dim=1)
            per_seq.extend((float(a), int(b))
                           for a, b in zip(row_nll.tolist(), row_n.tolist()))
        if verbose:
            print(f"  [nll] batch {b0} n={n} nll={nll:.1f} {time.time()-t0:.1f}s",
                  flush=True)
    if return_per_seq:
        return tot, ntok, tot / max(ntok, 1), per_seq
    return tot, ntok, tot / max(ntok, 1)


def eval_domain_nll(eng, domain, split="eval", **kw):
    rows = load_domain(domain, split)
    texts = [r["text"] for r in rows]
    return eval_nll(eng["model"], eng["tok"], texts, **kw)


# ---------------------------------------------------------------- 捕获
TOPK = 8


def capture_layer_stats(eng, texts, max_len=96, batch_size=8):
    """前向捕获每层 MoE 输入、路由topk(prob/idx), 仅保留有效token。
    返回 {layer: {"x": fp16 [T,H], "topi": int16 [T,K], "topv": fp16 [T,K]}}
    """
    model, tok = eng["model"], eng["tok"]
    n_layers = len(model.model.language_model.layers)
    buf = {li: {"x": [], "topi": [], "topv": []} for li in range(n_layers)}
    hooks = []

    def make_hook(li, mod):
        def hook(m, inp, out):
            x = inp[0].detach()  # [B,S,H]
            B, S, H = x.shape
            flat = x.reshape(-1, H)
            logits = F.linear(flat, m.gate.weight).float()
            probs = torch.softmax(logits, dim=-1)
            topv, topi = torch.topk(probs, TOPK, dim=-1)
            buf[li]["x"].append(flat.half().cpu())
            buf[li]["topi"].append(topi.to(torch.int16).cpu())
            buf[li]["topv"].append(topv.half().cpu())
        return hook

    for li in range(n_layers):
        hooks.append(model.model.language_model.layers[li].mlp
                     .register_forward_hook(make_hook(li, model.model.language_model.layers[li].mlp)))
    try:
        with torch.no_grad():
            for b0 in range(0, len(texts), batch_size):
                ids, mask = _batch_encode(tok, texts[b0:b0 + batch_size], max_len)
                model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for li in range(n_layers):
        x = torch.cat(buf[li]["x"], dim=0)
        topi = torch.cat(buf[li]["topi"], dim=0)
        topv = torch.cat(buf[li]["topv"], dim=0)
        out[li] = {"x": x, "topi": topi, "topv": topv}
    return out


def capture_route_hist(eng, texts, max_len=8, batch_size=8, padding="max_length"):
    """路由直方图特征 [L,E]: 每层 top-8 路由权重按专家聚合, 逐层归一化。"""
    model, tok = eng["model"], eng["tok"]
    n_layers = len(model.model.language_model.layers)
    E = eng["store"].num_experts
    hist = np.zeros((n_layers, E), dtype=np.float64)
    hooks = []

    def make_hook(li, mod):
        def hook(m, inp, out):
            x = inp[0].detach()
            B, S, H = x.shape
            flat = x.reshape(-1, H)
            logits = F.linear(flat, m.gate.weight).float()
            probs = torch.softmax(logits, dim=-1)
            topv, topi = torch.topk(probs, TOPK, dim=-1)
            idx = topi.reshape(-1).cpu().numpy()
            w = topv.reshape(-1).double().cpu().numpy()
            np.add.at(hist[li], idx, w)
        return hook

    for li in range(n_layers):
        hooks.append(model.model.language_model.layers[li].mlp
                     .register_forward_hook(make_hook(li, model.model.language_model.layers[li].mlp)))
    try:
        with torch.no_grad():
            for b0 in range(0, len(texts), batch_size):
                ids, mask = _batch_encode(tok, texts[b0:b0 + batch_size], max_len, padding)
                model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1e-12)
    return hist


# ---------------------------------------------------------------- 量化/替换
def quantize_bits(w: torch.Tensor, bits: int, group=128):
    """w [ROWS,IN] -> packed uint8, scale fp16 (nibble格式, 与 int4_store 兼容)"""
    maxq = 2 ** (bits - 1) - 1
    if bits == 4:
        return quantize_int4(w, group)
    ROWS, IN = w.shape
    wg = w.float().reshape(ROWS, IN // group, group)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / maxq).clamp(min=1e-8)
    q = torch.clamp(torch.round(wg / scale), -maxq, maxq).to(torch.int8) + 8
    q = q.reshape(ROWS, IN).to(torch.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed, scale.squeeze(-1).to(torch.float16)


class ExpertPatcher:
    """单专家精度替换: 写入 COW mmap 视图; restore 时从只读影子映射拷回原字节"""

    def __init__(self, store, int4_dir=STORE_INT4):
        self.store = store
        self.ro = np.memmap(os.path.join(int4_dir, "experts.bin"), dtype=np.uint8,
                            mode="r", shape=(store.meta["total_bytes"],))
        self.parts = store.meta
        off = 0
        self.part_off = {}
        for p in LAYER_PARTS:
            self.part_off[p] = off
            off += int(np.prod(self.parts["parts"][p]) * (2 if p.endswith("_s") else 1))
        self.stride = store.meta["layer_stride"]
        self.backups = {}

    def _seg(self, layer, part, expert, tensor_bytes):
        base = layer * self.stride + self.part_off[part]
        off = base + expert * tensor_bytes
        return off, off + tensor_bytes

    def _part_info(self, layer, part):
        shape = self.parts["parts"][part]
        n = int(np.prod(shape[1:]))
        nbytes = n * (2 if part.endswith("_s") else 1)
        return n, nbytes

    def set_expert(self, layer, expert, parts):
        """parts: {"gu_q":uint8[2I,H/2],"gu_s":fp16[2I,H/G],"dn_q":...,"dn_s":...}"""
        for p in LAYER_PARTS:
            n, nbytes = self._part_info(layer, p)
            off, end = self._seg(layer, p, expert, nbytes)
            raw = parts[p]
            if p.endswith("_s"):
                raw = raw.numpy().view(np.uint8).reshape(-1)
            else:
                raw = raw.numpy().reshape(-1)
            assert raw.shape[0] == nbytes, (p, raw.shape[0], nbytes)
            self.store._mm[off:end] = raw

    def restore_expert(self, layer, expert):
        """用原INT4字节覆盖回去"""
        for p in LAYER_PARTS:
            n, nbytes = self._part_info(layer, p)
            off, end = self._seg(layer, p, expert, nbytes)
            self.store._mm[off:end] = self.ro[off:end]

    def set_expert_from_bits(self, layer, expert, bits, w_gu_bf16, w_dn_bf16):
        pq, sq = quantize_bits(w_gu_bf16, bits)
        pd, sd = quantize_bits(w_dn_bf16, bits)
        self.set_expert(layer, expert, {"gu_q": pq, "gu_s": sq,
                                        "dn_q": pd, "dn_s": sd})


def read_expert_bf16(wmap, model_path, layer, expert):
    """从 safetensors 读单专家 BF16 (gate_up [2I,H], down [H,I])"""
    import safetensors
    gu_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
    dn_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    with safetensors.safe_open(os.path.join(model_path, wmap[gu_key]),
                               framework="pt") as f:
        w_gu = f.get_slice(gu_key)[expert].float()
    with safetensors.safe_open(os.path.join(model_path, wmap[dn_key]),
                               framework="pt") as f:
        w_dn = f.get_slice(dn_key)[expert].float()
    return w_gu, w_dn


def load_weight_map(model_path=MOE_MODEL_PATH):
    with open(os.path.join(model_path, "model.safetensors.index.json"),
              encoding="utf-8") as f:
        return json.load(f)["weight_map"]
