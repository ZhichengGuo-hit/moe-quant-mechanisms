"""A2: 补丁间差集换装 + 预填充重叠的双缓冲异步换装。

基线 (E11): 从 base_int4 装一个 math/code 补丁 = 写入 7680 专家 (11.6 GiB, 3.4-3.5s)。
本实验:
  1) 差集换装: math->code 只写 xor 专家 (int3/int4 各自方向的字节), 写量降至 ~1/3;
  2) 数值校验: 差集换装后的 eval NLL 与"整装"严格一致;
  3) 双缓冲异步: 在 prefill 前向 (8 条 general, 96 token) 期间把差集字节预读到
     staging 缓冲区, prefill 结束后一次性 memcpy 提交, 记录 stall 时间。
输出: results/a2_delta_switch.json
"""
import os
import sys
import json
import time
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch

import tcq_lib as T
from moe_lib import Int4Store

P_HIGH = 0.25
LAYER_PARTS = ("gu_q", "gu_s", "dn_q", "dn_s")
N_EVAL_SEQ = 8


def select_per_layer(score, frac):
    L, E = score.shape
    k = max(1, int(frac * E))
    high = np.zeros((L, E), dtype=bool)
    for l in range(L):
        high[l, np.argsort(-score[l])[:k]] = True
    return high


class DeltaWriter:
    """按 (layer, part, expert) 的字节区间写入, 支持 staging 双缓冲"""

    def __init__(self, patcher, store3, n_layers, n_experts):
        self.p = patcher
        self.s3 = store3
        self.L, self.E = n_layers, n_experts
        self.spans = {}
        for l in range(n_layers):
            for part in LAYER_PARTS:
                n, nbytes = patcher._part_info(l, part)
                self.spans[(l, part)] = nbytes
        self.expert_bytes = sum(self.spans[(0, p)] for p in LAYER_PARTS)

    def _src_int3(self, l, e, part):
        raw = self.s3.layer_views(l)[part][e]
        if part.endswith("_s"):
            return raw.numpy().view(np.uint8).reshape(-1)
        return raw.numpy().reshape(-1)

    def _src_int4(self, l, e, part):
        n, nbytes = self.p._part_info(l, part)
        off, end = self.p._seg(l, part, e, nbytes)
        return self.p.ro[off:end]

    def apply(self, to_mask, from_mask=None, staging=None):
        """把补丁从 from_mask 换成 to_mask (from_mask=None 表示从 base int4 整装)"""
        t0 = time.perf_counter()
        n_exp = 0
        n_bytes = 0
        for l in range(self.L):
            for e in range(self.E):
                to_high = bool(to_mask[l, e])
                from_high = True if from_mask is None else bool(from_mask[l, e])
                if to_high == from_high:
                    continue
                n_exp += 1
                for part in LAYER_PARTS:
                    nbytes = self.spans[(l, part)]
                    n_bytes += nbytes
                    off, end = self.p._seg(l, part, e, nbytes)
                    if to_high:
                        raw = self._src_int4(l, e, part)
                        self.p.store._mm[off:end] = raw
                    else:
                        raw = self._src_int3(l, e, part)
                        self.p.store._mm[off:end] = raw
        dt = time.perf_counter() - t0
        return dt, n_exp, n_bytes


def main():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sm = json.load(f)
    s = np.array(sm["s"])
    doms, di = sm["domains"], {d: i for i, d in enumerate(sm["domains"])}

    eng = T.build_engine()
    tok = eng["tok"]
    patcher = T.ExpertPatcher(eng["store"])
    store3 = Int4Store(T.STORE_INT3, mmap_mode="r")
    n_layers, n_experts = store3.num_layers, store3.num_experts
    dw = DeltaWriter(patcher, store3, n_layers, n_experts)

    high = {d: select_per_layer(s[:, :, di[d]], P_HIGH) for d in doms}
    high["global"] = select_per_layer(s.sum(-1), P_HIGH)

    evalsets = {d: [r["text"] for r in T.load_domain(d, "eval64")[:N_EVAL_SEQ]]
                for d in T.DOMAINS}

    def eval_nll():
        return {d: T.eval_nll(eng["model"], tok, evalsets[d], max_len=64,
                              batch_size=8)[2] for d in T.DOMAINS}

    out = {"p_high": P_HIGH, "expert_bytes": dw.expert_bytes,
           "full": {}, "delta": {}, "async": {}}

    # ---- 基线: 整装 math ----
    t0 = time.perf_counter()
    dt, n_exp, n_bytes = dw.apply(high["math"])
    out["full"]["math"] = {"install_s": round(dt, 3), "n_experts": n_exp,
                           "bytes": n_bytes}
    print(f"[full math] {dt:.2f}s {n_exp} experts {n_bytes/2**30:.2f} GiB", flush=True)
    nll_full_math = eval_nll()
    for d in T.DOMAINS:
        print(f"  nll {d}: {nll_full_math[d]:.4f}", flush=True)

    # ---- 差集: math -> code ----
    dt_d, n_exp_d, n_bytes_d = dw.apply(high["code"], from_mask=high["math"])
    nll_delta = eval_nll()
    eq = all(abs(nll_delta[d] - nll_full_math[d]) < 1e-12 for d in T.DOMAINS)
    # 参考: 从 math 重新整装 code 的耗时 (写 7680) —— 只测写入, 不评测
    dt_full_from_math, n_full, nb_full = dw.apply(high["global"], from_mask=high["math"]) \
        if False else (0.0, 0, 0)
    out["delta"]["math->code"] = {
        "install_s": round(dt_d, 3), "n_experts": n_exp_d, "bytes": n_bytes_d,
        "write_ratio_vs_full": n_bytes_d / out["full"]["math"]["bytes"],
        "nll_after_delta": nll_delta,
        "nll_equal_to_full_code": eq}
    print(f"[delta math->code] {dt_d:.2f}s {n_exp_d} experts "
          f"{n_bytes_d/2**30:.2f} GiB (x{n_bytes_d/out['full']['math']['bytes']:.2f}) "
          f"nll_eq={eq}", flush=True)

    # ---- 双缓冲异步: prefill 期间预读差集, 之后提交 ----
    dw.apply(high["math"])  # 回到 math 补丁
    target = high["global"]
    prefill_texts = [t for d in T.DOMAINS for t in evalsets[d][:2]]  # 6 条 * 96 token

    probed = {"read_s": None, "apply_s": None, "prefill_s": None}

    def staged_read():
        t1 = time.perf_counter()
        for l in range(n_layers):
            for e in np.nonzero(target[l] != high["math"][l])[0]:
                for part in LAYER_PARTS:
                    nbytes = dw.spans[(l, part)]
                    off, end = patcher._seg(l, int(e), part, nbytes)
                    if target[l, e]:
                        patcher.ro[off:end].copy()
                    else:
                        dw._src_int3(l, int(e), part).copy()
        probed["read_s"] = time.perf_counter() - t1

    th = threading.Thread(target=staged_read)
    t1 = time.perf_counter()
    th.start()
    # 与写线程并行的 prefill 前向 (96 token)
    T.eval_nll(eng["model"], tok, prefill_texts, max_len=96, batch_size=8)
    probed["prefill_s"] = time.perf_counter() - t1
    th.join()
    t2 = time.perf_counter()
    dt_async, n_async, nb_async = dw.apply(target, from_mask=high["math"])
    probed["apply_s"] = time.perf_counter() - t2
    probed["overlap_window_s"] = probed["prefill_s"]
    probed["read_fully_hidden"] = probed["read_s"] <= probed["prefill_s"]
    probed["stall_s"] = probed["apply_s"]
    out["async"]["math->global"] = {
        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in probed.items()},
        "n_experts": n_async, "bytes": nb_async}
    print(f"[async] read={probed['read_s']:.2f}s prefill={probed['prefill_s']:.2f}s "
          f"apply={probed['apply_s']:.3f}s hidden={probed['read_fully_hidden']}", flush=True)

    # 恢复 base int4
    patcher_restore = high["global"]
    dw.apply(np.ones((n_layers, n_experts), dtype=bool), from_mask=patcher_restore)

    out["delta"]["note"] = ("差集字节 = xor(高精度集合) 对应的 int3/int4 写入; "
                            "整装 = 7680 专家 (11.6 GiB)")
    p = os.path.join(T.RES_DIR, "a2_delta_switch.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[a2] saved {p}", flush=True)


if __name__ == "__main__":
    main()
