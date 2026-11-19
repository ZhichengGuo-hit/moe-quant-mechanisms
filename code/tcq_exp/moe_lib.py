"""
MoE 推训一体 INT4 核心库 (Qwen3.6-35B-A3B)
=============================================
架构: 全部专家权重INT4常驻CPU内存(mmap共享) + 路由/attention常驻GPU + 专家按需拷贝
      + CPU侧LoRA训练(共享同一份INT4专家权重, 自定义autograd按需反量化)

组件:
  1. quantize_int4 / dequant_int4   - INT4对称分组量化 (group=128, scale=fp16)
  2. Int4Store                      - mmap共享专家权重库 (训练/推理两进程零拷贝共享同一物理内存)
  3. ExpertCache                    - GPU LRU专家缓存 (pinned staging + 异步H2D + GPU端反量化)
  4. GPUMoE                         - 替换HF SparseMoeBlock (推理侧: 专家按需拷贝)
  5. CPUMoE / ExpertLinearFn        - 训练侧: INT4存储+按需反量化, 反向不驻留权重
  6. build_moe_model                - meta设备初始化 + 非专家权重定向加载 + rotary buffer修复

权重布局(checkpoint): layers.N.mlp.experts.gate_up_proj [E,2I,H] / down_proj [E,H,I]
"""
import os
import json
import math
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MOE_MODEL_PATH = r"E:\CUDA11.8\model\Qwen3.6-35B-A3B"
STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "int4_store")
GROUP = 128  # INT4 量化分组大小

# prefill专家分块预取: 每块 PREFILL_CHUNK 个专家取回->计算->释放引用.
# 默认0=整层一次取回(生产16GB显存无压力); 显存受限节点(如双进程仿真8G)设16
# 可将长prompt prefill的专家权重峰值从 ~唯一专家数x6.2MB 压到 (slots+chunk)x6.2MB.
PREFILL_CHUNK = int(os.environ.get("PREFILL_CHUNK", "0"))

# ===== Routing Trace (for expert cache analysis) =====
# When enabled, each GPUMoE layer records its routing decisions.
# Used offline to analyze expert access patterns and cache hit rates.
TRACE_ENABLED = False
_TRACE_BUFFER = []  # list of {layer, token_idx, experts, weights}
_TRACE_TOKEN_COUNTER = 0


def trace_reset():
    """Clear trace buffer and reset token counter."""
    global _TRACE_BUFFER, _TRACE_TOKEN_COUNTER
    _TRACE_BUFFER = []
    _TRACE_TOKEN_COUNTER = 0


def trace_get():
    """Return current trace buffer contents."""
    return list(_TRACE_BUFFER)


def trace_next_token():
    """Increment token counter (call once per decode step)."""
    global _TRACE_TOKEN_COUNTER
    _TRACE_TOKEN_COUNTER += 1


# =============================================================================
# 1. INT4 对称分组量化
# =============================================================================
def quantize_int4(w: torch.Tensor, group: int = GROUP):
    """w [OUT,IN] -> packed uint8 [OUT,IN/2], scale fp16 [OUT,IN/group]
    对称量化: q = clamp(round(w/scale), -7, 7), nibble = q+8 in [1,15]"""
    OUT, IN = w.shape
    assert IN % group == 0
    wg = w.float().reshape(OUT, IN // group, group)
    amax = wg.abs().amax(dim=-1, keepdim=True)          # [OUT, IN/group, 1]
    scale = (amax / 7.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(wg / scale), -7, 7).to(torch.int8) + 8  # [1,15]
    q = q.reshape(OUT, IN)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale.squeeze(-1).to(torch.float16)


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """packed uint8 [OUT,IN/2] -> int8 [OUT,IN], 值域[-8,7]"""
    lo = (packed & 15).to(torch.int8) - 8
    hi = (packed >> 4).to(torch.int8) - 8
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.float32, group: int = GROUP):
    """CPU/GPU通用反量化: packed [OUT,IN/2] + scale [OUT,IN/group] -> [OUT,IN]
    reshape广播实现(无repeat_interleave物化), 比朴素实现快~3x"""
    OUT = packed.shape[0]
    lo = packed & 15                                   # uint8 [OUT,IN/2]
    hi = packed >> 4
    w = torch.empty(OUT, lo.shape[1], 2, dtype=packed.dtype, device=packed.device)
    w[..., 0] = lo
    w[..., 1] = hi
    q = (w.reshape(OUT, -1).to(dtype) - 8.0).reshape(OUT, -1, group)
    return (q * scale.to(dtype).unsqueeze(-1)).reshape(OUT, -1)


# =============================================================================
# 2. INT4 专家权重库 (mmap共享)
# =============================================================================
# 每层布局: gu_q [E,2I,H/2]u8 | gu_s [E,2I,H/G]f16 | dn_q [E,H,I/2]u8 | dn_s [E,H,I/G]f16
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")


class Int4Store:
    """mmap方式打开专家INT4权重库.
    mode='c'(copy-on-write): 读共享OS页缓存(多进程共享同一份物理内存), 写私有(不回写文件)
    mode='r'(只读): 文件后备只读映射, 不占进程commit额度 —— 纯推理多机实验必需
    (mode='c'在Windows上对整个视图一次性计收commit, 15.5GB store + safetensors分片
    整读会在32GB Job Object上限下越界, 原生分配失败表现为访问违例)"""

    def __init__(self, store_dir: str = STORE_DIR, mmap_mode: str = "c"):
        with open(os.path.join(store_dir, "index.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.num_layers = self.meta["num_layers"]
        self.num_experts = self.meta["num_experts"]
        self.hidden = self.meta["hidden"]
        self.moe_inter = self.meta["moe_inter"]
        self.group = self.meta["group"]
        self.parts = self.meta["parts"]  # {"gu_q": [E,2I,H/2], ...}
        self.layer_stride = self.meta["layer_stride"]
        self.fpath = os.path.join(store_dir, "experts.bin")
        self._pinned_buf = None
        if mmap_mode == "r":
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._mm = np.memmap(self.fpath, dtype=np.uint8, mode="r",
                                     shape=(self.meta["total_bytes"],))
        else:
            self._mm = np.memmap(self.fpath, dtype=np.uint8, mode="c",
                                 shape=(self.meta["total_bytes"],))
        self._part_off = {}
        off = 0
        for p in LAYER_PARTS:
            self._part_off[p] = off
            off += int(np.prod(self.parts[p])) * (2 if p.endswith("_s") else 1)
        assert off == self.layer_stride, f"layout mismatch {off} != {self.layer_stride}"

    def pin_all(self, verbose=True):
        """全量store拷贝到pinned CPU内存 (推理进程私有副本).
        之后expert_slices直接从pinned发起H2D async, 消除每次miss的mmap->pinned memcpy.
        代价: 15.5GB物理内存 + 与训练进程的mmap共享失效(训练侧仍走mmap共享原文件)"""
        t0 = time.time()
        n = self.meta["total_bytes"]
        buf = torch.empty(n, dtype=torch.uint8).pin_memory()
        buf.numpy()[:] = self._mm[:]
        self._pinned_buf = buf  # _view基于pinned tensor视图(保留pinned标记)
        if verbose:
            print(f"[Store] pin_all: {n/2**30:.2f} GiB pinned in {time.time()-t0:.1f}s", flush=True)

    def _view(self, layer: int, part: str):
        """零拷贝视图 [E, ...] (pinned模式下为pinned tensor视图, 支持async H2D)"""
        shape = self.parts[part]
        is_q = part.endswith("_q")
        dt_t = torch.uint8 if is_q else torch.float16
        dt_n = np.uint8 if is_q else np.float16
        off = layer * self.layer_stride + self._part_off[part]
        n = int(np.prod(shape))
        nbytes = n * (1 if is_q else 2)
        if self._pinned_buf is not None:
            return self._pinned_buf[off:off + nbytes].view(dt_t).reshape(shape)
        arr = self._mm[off:off + nbytes].view(dt_n).reshape(shape)
        return torch.from_numpy(arr)

    def layer_views(self, layer: int):
        return {p: self._view(layer, p) for p in LAYER_PARTS}

    def expert_slices(self, layer: int, expert: int):
        """单专家4个分片的零拷贝视图 (供H2D拷贝)"""
        return {p: self._view(layer, p)[expert] for p in LAYER_PARTS}

    def dequant_expert(self, layer: int, expert: int, dtype=torch.float32):
        """CPU端反量化单专家 -> (w_gu [2I,H], w_dn [H,I])"""
        v = self.layer_views(layer)
        w_gu = dequant_int4(v["gu_q"][expert], v["gu_s"][expert], dtype, self.group)
        w_dn = dequant_int4(v["dn_q"][expert], v["dn_s"][expert], dtype, self.group)
        return w_gu, w_dn

    def dequant_part(self, layer: int, expert: int, part: str, dtype=torch.float32):
        """反量化单专家的单个矩阵 ('gu'或'dn') —— CPU训练路径"""
        v = self.layer_views(layer)
        if part == "gu":
            return dequant_int4(v["gu_q"][expert], v["gu_s"][expert], dtype, self.group)
        return dequant_int4(v["dn_q"][expert], v["dn_s"][expert], dtype, self.group)

    def expert_bytes(self):
        """单专家INT4字节数(H2D传输量)"""
        return sum(int(np.prod(self.parts[p][1:])) * (2 if p.endswith("_s") else 1)
                   for p in LAYER_PARTS)

    def total_gb(self):
        return self.meta["total_bytes"] / 2**30


# =============================================================================
# 3b. CPU大容量专家缓存 (llama.cpp式: 专家FFN在CPU计算, 权重fp16常驻RAM)
#     后台线程从INT4源解包填充(异步), 推理路径零等待
# =============================================================================
import threading
from collections import deque


class CPUExpertCache:
    """CPU RAM fp16专家LRU缓存. 容量5000槽(30GB)时命中率~92%(LRU曲线实测).
    miss不阻塞: 提交后台线程解包, 本次走GPU路径, 下次访问即CPU命中"""

    def __init__(self, store: Int4Store, slots: int = 5000, dtype=torch.float16,
                 fill_threads: int = 2):
        self.store, self.slots, self.dtype = store, slots, dtype
        self.entries = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "filled": 0}
        self._queue = deque()
        self._lock = threading.Lock()
        self._stop = False
        self._threads = [threading.Thread(target=self._fill_worker, daemon=True)
                         for _ in range(fill_threads)]
        for t in self._threads:
            t.start()

    def _fill_worker(self):
        while not self._stop:
            try:
                layer, e = self._queue.popleft()
            except IndexError:
                time.sleep(0.01)
                continue
            key = (layer, e)
            with self._lock:
                if key in self.entries:
                    continue
            try:
                w_gu, w_dn = self.store.dequant_expert(layer, e, self.dtype)
            except Exception:
                continue
            with self._lock:
                self._put(key, (w_gu, w_dn))
                self.stats["filled"] += 1

    def _put(self, key, weights):
        if key in self.entries:
            return
        while len(self.entries) >= self.slots:
            self.entries.popitem(last=False)
        self.entries[key] = weights

    def submit(self, layer, experts):
        """请求后台填充 (去重, 不阻塞)"""
        with self._lock:
            known = set()
            for e in experts:
                key = (layer, e)
                if key not in self.entries and key not in known:
                    known.add(key)
                    self._queue.append(key)

    def get_all(self, layer, experts):
        """全命中返回{expert:(w_gu,w_dn)}, 任一miss返回None(并触发后台填充)"""
        with self._lock:
            got = {}
            for e in experts:
                w = self.entries.get((layer, e))
                if w is None:
                    self.stats["misses"] += 1
                    return None
                self.entries.move_to_end((layer, e))
                got[e] = w
            self.stats["hits"] += len(got)
            return got

    def get_or_dequant(self, layer, experts):
        """命中直接返回; miss现场同步解包(~1.5ms/专家, 仍远快于GPU整层路径切换)"""
        with self._lock:
            got, miss = {}, []
            for e in experts:
                w = self.entries.get((layer, e))
                if w is None:
                    miss.append(e)
                else:
                    self.entries.move_to_end((layer, e))
                    got[e] = w
            self.stats["hits"] += len(got)
            self.stats["misses"] += len(miss)
        for e in miss:
            w_gu, w_dn = self.store.dequant_expert(layer, e, self.dtype)
            with self._lock:
                self._put((layer, e), (w_gu, w_dn))
                self.stats["filled"] += 1
            got[e] = (w_gu, w_dn)
        return got

    def hit_rate(self):
        h, m = self.stats["hits"], self.stats["misses"]
        return h / max(h + m, 1)

    def ram_gb(self):
        if not self.entries:
            return 0.0
        per = (2 * self.store.moe_inter * self.store.hidden
               + self.store.hidden * self.store.moe_inter) * 2  # fp16
        return len(self.entries) * per / 2**30


# =============================================================================
# 2b. INT4 存储构建 (量化脚本与单元测试共用)
# =============================================================================
def write_store(layer_experts, out_dir: str, num_layers: int, num_experts: int,
                hidden: int, moe_inter: int, group: int = GROUP, extra_meta: dict = None):
    """layer_experts: 每层迭代 (gu [E,2I,H], dn [E,H,I]) -> 写出 experts.bin + index.json"""
    E, H, I = num_experts, hidden, moe_inter
    parts = {
        "gu_q": [E, 2 * I, H // 2], "gu_s": [E, 2 * I, H // group],
        "dn_q": [E, H, I // 2], "dn_s": [E, H, I // group],
    }
    part_off, off = {}, 0
    for p in LAYER_PARTS:
        part_off[p] = off
        off += int(np.prod(parts[p])) * (2 if p.endswith("_s") else 1)
    layer_stride = off
    total = layer_stride * num_layers
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, "experts.bin")
    with open(fpath, "wb") as f:
        f.truncate(total)
    mm = np.memmap(fpath, dtype=np.uint8, mode="r+", shape=(total,))
    err_stats = []
    t0 = time.time()
    for li, (gu, dn) in enumerate(layer_experts):
        base = li * layer_stride
        for part, w, out_rows in (("gu", gu, 2 * I), ("dn", dn, H)):
            packed, scale = quantize_int4(w.reshape(-1, w.shape[-1]), group)
            packed = packed.reshape(num_experts, out_rows, -1)
            scale = scale.reshape(num_experts, out_rows, -1)
            qk, sk = f"{part}_q", f"{part}_s"
            pq = np.ascontiguousarray(packed.numpy())
            ps = np.ascontiguousarray(scale.numpy()).view(np.uint8)  # fp16字节视图
            o = base + part_off[qk]
            mm[o:o + pq.nbytes] = pq.reshape(-1)
            o = base + part_off[sk]
            mm[o:o + ps.nbytes] = ps.reshape(-1)
            if li == 0:  # 首层误差统计(前4个专家)
                for e in range(min(4, num_experts)):
                    w_orig = w[e].float()
                    w_rec = dequant_int4(packed[e], scale[e], torch.float32, group)
                    rel = (w_orig - w_rec).norm() / w_orig.norm().clamp(min=1e-6)
                    err_stats.append(round(float(rel), 4))
        if (li + 1) % 8 == 0 or li == num_layers - 1:
            print(f"[Quant] layer {li+1}/{num_layers} | {time.time()-t0:.0f}s", flush=True)
    mm.flush()
    meta = {"version": 1, "group": group, "num_layers": num_layers, "num_experts": E,
            "hidden": H, "moe_inter": I, "parts": parts, "layer_stride": layer_stride,
            "total_bytes": total, "scheme": "sym-int4-group128-scale-fp16",
            "layer_rel_err_sample": err_stats}
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"[Quant] DONE {total/2**30:.2f} GiB -> {out_dir} | "
          f"rel_err sample max={max(err_stats):.4f} mean={sum(err_stats)/len(err_stats):.4f}", flush=True)



class _Pending:
    __slots__ = ("layer", "expert", "pin", "gpu", "event", "buf_idx")

    def __init__(self, layer, expert, pin, gpu, event, buf_idx):
        self.layer, self.expert = layer, expert
        self.pin, self.gpu = pin, gpu
        self.event, self.buf_idx = event, buf_idx


class ExpertCache:
    """全局LRU (key=(layer,expert)) -> (w_gu fp16 [2I,H], w_dn fp16 [H,I]) 常驻GPU
    + 跨token投机预取: 相邻token路由高度相关, 上一token的专家在下一token计算期间异步预取"""

    def __init__(self, store: Int4Store, device="cuda", dtype=torch.float16,
                 slots: int = 1024, pin_pool: int = 8, use_pinned: bool = True,
                 trace: bool = False, prefetch: bool = True):
        self.store, self.device, self.dtype = store, device, dtype
        self.slots = slots
        self.use_pinned = use_pinned
        self.prefetch_enabled = prefetch
        self.trace = [] if trace else None
        self.last_route = []       # 上一token各层 (layer, ids) — 预取依据
        self._inflight = {}        # (layer,expert) -> _Pending (预取飞行中)
        self.entries = OrderedDict()
        self.copy_stream = torch.cuda.Stream(device=device)
        # GPU uint8/fp16 staging池 (反量化源)
        self._gpu_pool = []
        self._pin_pool_free = list(range(pin_pool))
        self._pin_pool = []
        for _ in range(pin_pool):
            pin = {p: torch.empty(tuple(store.parts[p][1:]),
                                  dtype=torch.uint8 if p.endswith("_q") else torch.float16).pin_memory()
                   for p in LAYER_PARTS}
            self._pin_pool.append(pin)
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "h2d_bytes": 0,
                      "h2d_s": 0.0, "dequant_s": 0.0, "pinned": use_pinned}

    def _gpu_stage(self):
        if self._gpu_pool:
            return self._gpu_pool.pop()
        g = {p: torch.empty(tuple(self.store.parts[p][1:]),
                            dtype=torch.uint8 if p.endswith("_q") else torch.float16,
                            device=self.device)
             for p in LAYER_PARTS}
        return g

    def hits(self, layer, experts):
        return [e for e in experts if (layer, e) in self.entries]

    def prefetch_from_last_route(self):
        """token间投机预取: 上一token的(layer, ids)中不在缓存的专家发起异步H2D,
        与当前token的attention计算(copy_stream vs 默认流)重叠. 丢弃上一轮未消费的飞行中拷贝"""
        if not self.prefetch_enabled:
            self.last_route = []
            return
        if self._inflight:  # 丢弃未被消费的投机拷贝(路由漂移部分)
            self.discard_inflight()
        seen = set()
        keys = []
        for layer, ids in self.last_route:
            for e in ids:
                k = (layer, e)
                if k not in seen and k not in self.entries:
                    seen.add(k)
                    keys.append(k)
        self.last_route = []
        if not keys:
            return
        for pd in self.begin_fetch_multi(keys):
            self._inflight[(pd.layer, pd.expert)] = pd

    def begin_fetch_multi(self, keys):
        """批量发起 [(layer,expert),...] 的异步拷贝 (pinned store直连; mmap模式受pin池限制)"""
        pendings = []
        pinned_store = self.store._pinned_buf is not None
        for layer, e in keys:
            if pinned_store:
                src = self.store.expert_slices(layer, e)
                gpu = self._gpu_stage()
                t0 = time.perf_counter()
                with torch.cuda.stream(self.copy_stream):
                    for p in LAYER_PARTS:
                        gpu[p].copy_(src[p], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(self.copy_stream)
                self.stats["h2d_s"] += time.perf_counter() - t0
                self.stats["h2d_bytes"] += self.store.expert_bytes()
                pendings.append(_Pending(layer, e, None, gpu, ev, -1))
            else:
                if len(self._pin_pool_free) == 0:
                    break
                bi = self._pin_pool_free.pop(0)
                pin, gpu = self._pin_pool[bi], self._gpu_stage()
                src = self.store.expert_slices(layer, e)
                t0 = time.perf_counter()
                with torch.cuda.stream(self.copy_stream):
                    for p in LAYER_PARTS:
                        pin[p].copy_(src[p])
                    for p in LAYER_PARTS:
                        gpu[p].copy_(pin[p], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(self.copy_stream)
                self.stats["h2d_s"] += time.perf_counter() - t0
                self.stats["h2d_bytes"] += self.store.expert_bytes()
                pendings.append(_Pending(layer, e, pin, gpu, ev, bi))
        return pendings

    def discard_inflight(self):
        """丢弃飞行中的投机拷贝: 等event(通常已完成)后直接还staging, 不反量化不占缓存槽"""
        for pd in self._inflight.values():
            pd.event.synchronize()
            if pd.pin is not None:
                self._pin_pool_free.append(pd.buf_idx)
            self._gpu_pool.append(pd.gpu)
        self._inflight.clear()

    def begin_fetch(self, layer, experts):
        """发起未命中专家的异步拷贝, 返回pending列表.
        pinned store模式: 源即pinned, 直接async H2D (无CPU memcpy, 无pin池);
        mmap模式: mmap->pinned staging->GPU异步 (pin池耗尽时其余专家下次再取)"""
        pendings = []
        pinned_store = self.store._pinned_buf is not None
        for e in experts:
            src = self.store.expert_slices(layer, e)
            if pinned_store:
                gpu = self._gpu_stage()
                t0 = time.perf_counter()
                with torch.cuda.stream(self.copy_stream):
                    for p in LAYER_PARTS:
                        gpu[p].copy_(src[p], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(self.copy_stream)
                self.stats["h2d_s"] += time.perf_counter() - t0
                self.stats["h2d_bytes"] += self.store.expert_bytes()
                pendings.append(_Pending(layer, e, None, gpu, ev, -1))
                continue
            if len(self._pin_pool_free) == 0:
                break  # pin池耗尽, 其余专家走同步路径
            bi = self._pin_pool_free.pop(0)
            pin, gpu = self._pin_pool[bi], self._gpu_stage()
            t0 = time.perf_counter()
            with torch.cuda.stream(self.copy_stream):
                if self.use_pinned:
                    for p in LAYER_PARTS:
                        pin[p].copy_(src[p])          # CPU->pinned (memcpy)
                    for p in LAYER_PARTS:
                        gpu[p].copy_(pin[p], non_blocking=True)  # pinned->GPU 异步
                else:
                    for p in LAYER_PARTS:
                        gpu[p].copy_(src[p], non_blocking=True)  # pageable->GPU
                ev = torch.cuda.Event()
                ev.record(self.copy_stream)
            self.stats["h2d_s"] += time.perf_counter() - t0
            self.stats["h2d_bytes"] += self.store.expert_bytes()
            pendings.append(_Pending(layer, e, pin, gpu, ev, bi))
        return pendings

    def finish_fetch(self, pendings):
        """等待拷贝完成 -> GPU端反量化 -> 写入LRU. 返回 {expert:(w_gu,w_dn)}
        优化: 同一copy stream顺序执行, 只同步最后一个event; 反量化批量launch后单次sync"""
        out = {}
        if not pendings:
            return out
        pendings[-1].event.synchronize()  # H2D完成 (pinned缓冲此后可复用)
        for pd in pendings:
            t0 = time.perf_counter()
            w_gu = dequant_int4(pd.gpu["gu_q"], pd.gpu["gu_s"], self.dtype, self.store.group)
            w_dn = dequant_int4(pd.gpu["dn_q"], pd.gpu["dn_s"], self.dtype, self.store.group)
            self.stats["dequant_s"] += time.perf_counter() - t0
            self._put(pd.layer, pd.expert, (w_gu, w_dn))
            out[pd.expert] = (w_gu, w_dn)
            if pd.pin is not None:
                self._pin_pool_free.append(pd.buf_idx)
        ev2 = torch.cuda.Event()
        ev2.record()  # 默认流上全部反量化kernel完成
        ev2.synchronize()  # staging缓冲复用前确保反量化已执行
        for pd in pendings:
            self._gpu_pool.append(pd.gpu)
        return out

    def _put(self, layer, expert, weights):
        key = (layer, expert)
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        while len(self.entries) >= self.slots:
            self.entries.popitem(last=False)
            self.stats["evictions"] += 1
        self.entries[key] = weights

    def get_weights(self, layer, experts):
        """同步获取 (命中直接返回; 未命中分批拷贝+反量化)"""
        if self.trace is not None:
            self.trace.extend((layer, e) for e in experts)
        got = {}
        miss = []
        for e in experts:
            w = self.entries.get((layer, e))
            if w is not None:
                self.entries.move_to_end((layer, e))
                got[e] = w
            else:
                miss.append(e)
        self.stats["hits"] += len(got)
        self.stats["misses"] += len(miss)
        if not miss:
            return got
        # 优先消费投机预取的飞行中拷贝
        pend, hard = [], []
        for e in miss:
            pd = self._inflight.pop((layer, e), None)
            if pd is not None:
                pend.append(pd)
            else:
                hard.append(e)
        if pend:
            got.update(self.finish_fetch(pend))
        if hard:
            if self.store._pinned_buf is not None:
                got.update(self.finish_fetch(self.begin_fetch(layer, hard)))
            else:
                i = 0
                while i < len(hard):  # pin池耗尽时分批处理, 保证不丢专家
                    batch = hard[i:i + len(self._pin_pool)]
                    i += len(batch)
                    got.update(self.finish_fetch(self.begin_fetch(layer, batch)))
        return got

    def reset(self):
        if self._inflight:
            self.discard_inflight()
        self.last_route = []
        self.entries.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "h2d_bytes": 0,
                      "h2d_s": 0.0, "dequant_s": 0.0, "pinned": self.use_pinned}

    def hit_rate(self):
        h, m = self.stats["hits"], self.stats["misses"]
        return h / max(h + m, 1)

    def vram_gb(self):
        """缓存中专家权重的fp16显存占用"""
        if not self.entries:
            return 0.0
        p = self.store.parts
        per = (p["gu_q"][1] * p["gu_q"][2] * 2 + p["dn_q"][1] * p["dn_q"][2] * 2) \
            * self.dtype.itemsize
        return len(self.entries) * per / 2**30


# =============================================================================
# 4. GPU推理侧 MoE块 (替换 HF Qwen3_5MoeSparseMoeBlock)
#    权重命名与checkpoint一致: gate.weight / shared_expert.* / shared_expert_gate.weight
# =============================================================================
DIRECT_STATS = {"calls": 0, "d2h_ms": 0.0, "kern_ms": 0.0, "h2d_ms": 0.0}


class GPUMoE(nn.Module):
    def __init__(self, layer_idx: int, hidden: int, moe_inter: int, num_experts: int,
                 top_k: int, cache: ExpertCache, cpu_cache: "CPUExpertCache" = None,
                 direct_ext=None, store: "Int4Store" = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden, num_experts, bias=False)  # 路由(常驻GPU)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)
        self.shared_expert = nn.ModuleDict({
            "gate_proj": nn.Linear(hidden, moe_inter, bias=False),
            "up_proj": nn.Linear(hidden, moe_inter, bias=False),
            "down_proj": nn.Linear(moe_inter, hidden, bias=False),
        })
        self.cache = cache  # 共享的全局ExpertCache
        self.cpu_cache = cpu_cache  # CPU大容量缓存(llama.cpp式专家CPU计算), 可为None
        self.direct_ext = direct_ext  # 直接INT4 GEMV扩展(llama.cpp Q4_K原理), 可为None
        self.dv = store.layer_views(layer_idx) if (direct_ext is not None and store) else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)
        # ---- 路由 (与HF Qwen3_5MoeTopKRouter逐算子一致) ----
        logits = F.linear(x, self.gate.weight).float()
        probs = torch.softmax(logits, dim=-1)
        top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_v = top_v / top_v.sum(dim=-1, keepdim=True)
        w = top_v.to(x.dtype)
        ti = top_i.to(torch.long)
        # ---- 路由追踪 (expert cache 分析用) ----
        if TRACE_ENABLED and x.shape[0] == 1:  # 仅 decode 阶段记录
            _TRACE_BUFFER.append({
                "layer": self.layer_idx,
                "token": _TRACE_TOKEN_COUNTER,
                "experts": ti[0].tolist(),
                "weights": [round(float(v), 6) for v in w[0]],
            })
        # ---- 专家计算 ----
        if x.shape[0] == 1:
            out = self._decode_experts(x, ti, w)
        else:
            out = self._prefill_experts(x, ti, w)
        # ---- shared expert (sigmoid门控, 与HF一致) ----
        se = self.shared_expert["down_proj"](
            F.silu(self.shared_expert["gate_proj"](x)) * self.shared_expert["up_proj"](x))
        se = torch.sigmoid(self.shared_expert_gate(x)) * se
        return (out + se).reshape(B, S, D)

    def _decode_experts(self, x, ti, w):
        """单token解码: direct_ext -> 直接INT4 GEMV(llama.cpp式, 无缓存无miss);
        否则CPU缓存全命中->CPU计算; 否则GPU路径+后台填充CPU缓存"""
        if self.direct_ext is not None:
            return self._direct_experts(x, ti, w)
        ids = ti[0].tolist()
        cc = self.cpu_cache
        if cc is not None:
            got = cc.get_all(self.layer_idx, ids)
            if got is not None:
                return self._cpu_expert_compute(x, ids, w, got)
            cc.submit(self.layer_idx, ids)  # 后台解包填充, 下次CPU命中
        if self.cache.prefetch_enabled:
            self.cache.last_route.append((self.layer_idx, ids))
        weights = self.cache.get_weights(self.layer_idx, ids)
        w_gu = torch.stack([weights[i][0] for i in ids])  # [k,2I,H]
        w_dn = torch.stack([weights[i][1] for i in ids])  # [k,H,I]
        k = w_gu.shape[0]
        xs = x.expand(k, x.shape[1]).unsqueeze(1)                      # [k,1,H]
        gu = torch.bmm(xs, w_gu.transpose(1, 2)).squeeze(1)            # [k,2I]
        g, u = gu.chunk(2, dim=-1)
        hh = F.silu(g) * u                                             # [k,I]
        o = torch.bmm(hh.unsqueeze(1), w_dn.transpose(1, 2)).squeeze(1)  # [k,H]
        return (o * w[0].unsqueeze(-1)).sum(dim=0, keepdim=True)       # [1,H]

    def _cpu_expert_compute(self, x, ids, w, got):
        """专家FFN在CPU计算(fp16 GEMV, benchmark实测43.8ms/token全模型):
        x D2H(4KB) -> 8专家循环GEMV -> 结果H2D. attention仍在GPU"""
        dev = x.device
        x_cpu = x[0].to("cpu", torch.float16)          # [H]
        wts = w[0].to("cpu", torch.float32).tolist()    # 路由权重
        acc = torch.zeros(x.shape[1], dtype=torch.float32)
        for e, wj in zip(ids, wts):
            w_gu, w_dn = got[e]
            gu = w_gu @ x_cpu                           # [2I] fp16
            g, u = gu.chunk(2, dim=-1)
            hh = F.silu(g.float()) * u.float()          # [I] fp32
            o = (w_dn @ hh.half()).float()              # [H]
            acc += o * wj
        return acc.to(dev, torch.float16).unsqueeze(0)  # [1,H]

    def _direct_experts(self, x, ti, w):
        """llama.cpp Q4_K原理: 专家权重常驻INT4(mmap, 与训练进程共享物理页),
        C++扩展寄存器内反量化GEMV, 无fp16缓存无miss. 每层2次同步:
        payload(x+ids+wts合并D2H) -> kernel -> 结果H2D"""
        t0 = time.perf_counter()
        payload = torch.cat((x[0], ti[0].to(x.dtype), w[0]))   # GPU fp16 [H+2K]
        p = payload.cpu()                                       # D2H (同步1)
        t1 = time.perf_counter()
        y = self.direct_ext.moe_ffn_int4(
            self.dv["gu_q"], self.dv["gu_s"], self.dv["dn_q"], self.dv["dn_s"], p)
        t2 = time.perf_counter()
        out = y.to(device=x.device, dtype=x.dtype).unsqueeze(0)  # H2D (同步2)
        t3 = time.perf_counter()
        DIRECT_STATS["calls"] += 1
        DIRECT_STATS["d2h_ms"] += (t1 - t0) * 1000
        DIRECT_STATS["kern_ms"] += (t2 - t1) * 1000
        DIRECT_STATS["h2d_ms"] += (t3 - t2) * 1000
        return out

    def _direct_experts_batch(self, x, ti, w):
        """预填充批量直接INT4 (llama.cpp Q4_K原理): B个token一次D2H -> CPU线程池批量GEMV
        -> 结果一次H2D. 每token独立路由的K个专家, 无GPU反量化/无专家权重H2D."""
        B, K = ti.shape
        payload = torch.cat((x, ti.to(x.dtype), w), dim=1)   # GPU fp16 [B, H+2K]
        t0 = time.perf_counter()
        p = payload.cpu()                                    # D2H (同步1)
        t1 = time.perf_counter()
        y = self.direct_ext.moe_ffn_int4_batch(
            self.dv["gu_q"], self.dv["gu_s"], self.dv["dn_q"], self.dv["dn_s"], p)
        t2 = time.perf_counter()
        out = y.to(device=x.device, dtype=x.dtype)           # H2D (同步2)
        t3 = time.perf_counter()
        DIRECT_STATS["calls"] += B
        DIRECT_STATS["d2h_ms"] += (t1 - t0) * 1000
        DIRECT_STATS["kern_ms"] += (t2 - t1) * 1000
        DIRECT_STATS["h2d_ms"] += (t3 - t2) * 1000
        return out

    def _prefill_experts(self, x, ti, w):
        """多token预填充: 默认走按专家分组GPU缓存GEMM (与HF参考实现同构).
        可选 CPU批量直接INT4 (llama.cpp式): 仅在显存/带宽受限实验(如MOE_PREFILL_CPU_BATCH=1)
        时启用 —— 实测CPU路径带宽受限~0.7ms/token/层, 长prompt(B较大)是回退, B≤128才占优."""
        n_tok, k = ti.shape
        if (int(os.environ.get("MOE_PREFILL_CPU_BATCH", "0")) == 1
                and self.direct_ext is not None
                and hasattr(self.direct_ext, "moe_ffn_int4_batch")
                and n_tok > 0):
            return self._direct_experts_batch(x, ti, w)
        flat = ti.reshape(-1)
        order = flat.argsort()
        sorted_e = flat[order]
        # 段边界
        uniq_mask = torch.ones_like(sorted_e, dtype=torch.bool)
        uniq_mask[1:] = sorted_e[1:] != sorted_e[:-1]
        seg_starts = uniq_mask.nonzero(as_tuple=True)[0]
        experts_list = sorted_e[seg_starts].tolist()
        boundaries = seg_starts.tolist() + [n_tok * k]
        if self.cpu_cache is not None:
            self.cpu_cache.submit(self.layer_idx, experts_list)  # 后台填充供decode用
        tok_of = order // k
        pos_of = order % k
        out = torch.zeros_like(x)
        chunk = PREFILL_CHUNK or len(experts_list)
        for c0 in range(0, len(experts_list), chunk):
            seg = experts_list[c0:c0 + chunk]
            weights = self.cache.get_weights(self.layer_idx, seg)
            for j, e in enumerate(seg):
                b0, b1 = boundaries[c0 + j], boundaries[c0 + j + 1]
                toks = tok_of[b0:b1]
                w_gu, w_dn = weights[e]
                gu = F.linear(x[toks], w_gu)
                g, u = gu.chunk(2, dim=-1)
                hh = F.silu(g) * u
                o = F.linear(hh, w_dn)
                o = o * w[toks, pos_of[b0:b1], None]
                out.index_add_(0, toks, o)
            weights = None  # 释放本块专家权重引用(下一块取回时LRU可回收)
        return out


# =============================================================================
# 5. CPU训练侧: INT4专家 + 按需反量化autograd
# =============================================================================
class ExpertLinearFn(torch.autograd.Function):
    """y = x @ W.T; W为冻结的INT4专家权重, 前向/反向各自按需反量化(不驻留显存/内存)
    -> CPU大内存直接访问专家权重, 反向无需保留32B反量化权重"""

    @staticmethod
    def forward(ctx, x, store: Int4Store, layer: int, expert: int, part: str):
        ctx.store, ctx.layer, ctx.expert, ctx.part = store, layer, expert, part
        x = x.float()
        ctx.save_for_backward(x)
        with torch.no_grad():
            W = store.dequant_part(layer, expert, part, torch.float32)
            y = F.linear(x, W)
        return y

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        with torch.no_grad():
            W = ctx.store.dequant_part(ctx.layer, ctx.expert, ctx.part, torch.float32)
            gx = gy @ W
        return gx, None, None, None, None


class CPUMoE(nn.Module):
    """训练侧MoE块: 路由/shared expert为fp32常驻内存参数, 专家权重走INT4Store"""

    def __init__(self, layer_idx: int, hidden: int, moe_inter: int, num_experts: int,
                 top_k: int, store: Int4Store):
        super().__init__()
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)
        self.shared_expert = nn.ModuleDict({
            "gate_proj": nn.Linear(hidden, moe_inter, bias=False),
            "up_proj": nn.Linear(hidden, moe_inter, bias=False),
            "down_proj": nn.Linear(moe_inter, hidden, bias=False),
        })
        self.store = store

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)
        logits = F.linear(x, self.gate.weight).float()
        probs = torch.softmax(logits, dim=-1)
        top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_v = top_v / top_v.sum(dim=-1, keepdim=True)
        w = top_v.to(x.dtype)
        ti = top_i.to(torch.long)

        n_tok, k = ti.shape
        flat = ti.reshape(-1)
        order = flat.argsort()
        sorted_e = flat[order]
        uniq_mask = torch.ones_like(sorted_e, dtype=torch.bool)
        uniq_mask[1:] = sorted_e[1:] != sorted_e[:-1]
        seg_starts = uniq_mask.nonzero(as_tuple=True)[0]
        experts_list = sorted_e[seg_starts].tolist()
        boundaries = seg_starts.tolist() + [n_tok * k]
        tok_of = order // k
        pos_of = order % k
        out = torch.zeros_like(x)
        for e, b0, b1 in zip(experts_list, boundaries[:-1], boundaries[1:]):
            toks = tok_of[b0:b1]
            xe = x[toks]
            gu = ExpertLinearFn.apply(xe, self.store, self.layer_idx, e, "gu")
            g, u = gu.chunk(2, dim=-1)
            hh = F.silu(g) * u
            o = ExpertLinearFn.apply(hh, self.store, self.layer_idx, e, "dn")
            o = o * w[toks, pos_of[b0:b1], None]
            out = out.index_add(0, toks, o)
        se = self.shared_expert["down_proj"](
            F.silu(self.shared_expert["gate_proj"](x)) * self.shared_expert["up_proj"](x))
        se = torch.sigmoid(self.shared_expert_gate(x)) * se
        return (out + se).reshape(B, S, D)


# =============================================================================
# 6. 模型组装: meta初始化 + 非专家权重定向加载 + rotary修复 + MoE块替换
# =============================================================================
SKIP_PREFIXES = ("model.visual", "mtp")  # 纯文本场景不加载


def load_checkpoint_params(model, model_path: str, device, dtype,
                           skip_missing_on_model=True, verbose=True):
    """从safetensors分片定向加载非专家权重 (专家/visual/mtp已在meta上跳过)
    要求参数已被to_empty materialize到目标设备, 这里用copy_覆盖"""
    import safetensors
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(idx_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    shards = sorted(set(weight_map.values()))
    loaded, skipped = 0, 0
    with torch.no_grad():
        for shard in shards:
            with safetensors.safe_open(os.path.join(model_path, shard), framework="pt") as f:
                for key in f.keys():
                    if key.startswith(SKIP_PREFIXES) or ".mlp.experts." in key:
                        skipped += 1
                        continue
                    tgt = params.get(key)
                    if tgt is None:
                        tgt = buffers.get(key)
                    if tgt is None:
                        skipped += 1
                        continue
                    t = f.get_tensor(key)
                    if t.shape != tgt.shape:
                        skipped += 1
                        continue
                    tgt.copy_(t.to(device=tgt.device, dtype=tgt.dtype))
                    loaded += 1
    if verbose:
        n = sum(p.numel() for p in params.values() if p.device.type != "meta")
        print(f"[Load] {loaded} tensors loaded, {skipped} skipped | "
              f"materialized params {n/1e9:.2f}B on {device}", flush=True)
    return loaded


def fix_rotary_buffers(model, device):
    """meta初始化后, 在真实设备上重建文本rotary inv_freq buffer.
    注意: inv_freq非持久buffer不在checkpoint中, to_empty物化的是未初始化内存,
    必须无条件重建 (原`device.type=='meta'`检查在to_empty之后恒为False, 从不触发)"""
    fixed = 0
    for m in model.modules():
        if type(m).__name__ == "Qwen3_5MoeTextRotaryEmbedding":
            with torch.no_grad():
                inv_freq, _ = m.compute_default_rope_parameters(m.config, device=device)
            m.inv_freq = inv_freq.to(device).float()
            m.original_inv_freq = inv_freq.clone().to(device).float()
            fixed += 1
    return fixed


def build_moe_model(mode: str, model_path: str = MOE_MODEL_PATH, device="cuda",
                    dtype=torch.float16, store: Int4Store = None, cache: ExpertCache = None,
                    cpu_cache: "CPUExpertCache" = None, num_layers=None, direct_ext=None):
    """mode='gpu': 推理实例 (GPUMoE+ExpertCache) | mode='cpu': 训练实例 (CPUMoE+Int4Store)
    专家权重从不物化为完整张量 —— 只存在于INT4Store
    direct_ext: int4_ext扩展 -> decode走直接INT4 GEMV(llama.cpp式), prefill仍走GPU缓存"""
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration)

    cfg = AutoConfig.from_pretrained(model_path)
    tc = cfg.text_config
    with torch.device("meta"):
        model = Qwen3_5MoeForConditionalGeneration(cfg)
    # 纯文本场景: 删除visual tower (0.45B参数, --mm模式由vision_lib单独加载),
    # 避免to_empty物化死重; checkpoint中mtp.*键在HF模型中无对应模块, 天然不物化
    if hasattr(model.model, "visual"):
        del model.model.visual
    layers = model.model.language_model.layers
    for i in range(len(layers)):
        if mode == "gpu":
            layers[i].mlp = GPUMoE(i, tc.hidden_size, tc.moe_intermediate_size,
                                   tc.num_experts, tc.num_experts_per_tok, cache,
                                   cpu_cache=cpu_cache, direct_ext=direct_ext,
                                   store=store)
        else:
            layers[i].mlp = CPUMoE(i, tc.hidden_size, tc.moe_intermediate_size,
                                   tc.num_experts, tc.num_experts_per_tok, store)
    # meta上先转fp16 (零内存), to_empty直接以fp16物化 —— 消除fp32瞬时峰值
    # (2.45B参数: fp32峰值9.8GB -> fp16稳定4.9GB, 8GB显存上限下必需)
    model = model.to(dtype)
    model.to_empty(device=device)
    load_checkpoint_params(model, model_path, device, dtype)
    n_rot = fix_rotary_buffers(model, device)
    if n_rot == 0:
        print("[Build] WARN: no rotary buffers fixed", flush=True)
    model.eval()
    return model


def count_vram_gb():
    return round(torch.cuda.memory_allocated() / 2**30, 2)


def ram_gb():
    import psutil
    return round(psutil.Process(os.getpid()).memory_info().rss / 2**30, 2)
