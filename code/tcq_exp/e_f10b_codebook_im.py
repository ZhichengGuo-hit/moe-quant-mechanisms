"""E-F10b: imatrix 加权的 4 级共享码本（每 (layer, proj) 一张）。

与 E21 的差别：
  - E21 的 `quant_v1` 是"每矩阵一张码本"（只用于离线对比，容器装不下）；
  - 容器 `tcmoe.cb` 只有 [layer, gate/up/down, 4]，所以这里把同一 (layer, proj)
    下全部专家的归一化值 z = w / d（d 为容器已有的 fp16 组尺度）"合并成一组"再拟合；
  - 权重沿用 E21 的目标：a = imatrix[e][col] * d^2（权重空间误差的高斯近似）。

内存控制：
  - 逐 (layer, proj) 累积加权直方图 (W, S, S2)，bin 只占 ~200 KB；
  - 每层只保留一个专家的 gate_up (8 MB) / down (4 MB)；
  - 拟合时用直方图 bin 的精确加权均值调用 `e21_codebook.lloyd_grouped`，不复制实现。

另外做一次 recode 判定（默认 layer 0 + 前 2 个专家）：
  预打包库 `packed2_sym4` 的 2-bit code 是"均匀码本 {-1.5,-0.5,0.5,1.5}"下的索引，
  换码本后有两种用法：
    direct  直接用新码本解码旧 code        （错误：索引语义变了）
    recode  按新码本边界从 f32 权重重分配索引
  对比两者的 imatrix 加权 MSE，把结论写进 JSON 的 recode_needed。

用法：
  python e_f10b_codebook_im.py --dry-run                       # 只打印规模估算
  python e_f10b_codebook_im.py --layers 0 --experts 0-1 \
      --out _tmp_e_f10b_smoke.npz --json _tmp_e_f10b_smoke.json  # 小规模自检
  python e_f10b_codebook_im.py                                 # 全量 40x256x3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(ROOT, "tcmoe"))

from e21_codebook import (read_imatrix_diag, amax_scale, expand_d,  # noqa: E402
                          lloyd_grouped, parse_list)

GROUP = 128
N_LEVEL = 4
PROJS = ("gate", "up", "down")
HIDDEN = 2048
MOE_INTER = 512
IMATRIX = r"E:\CUDA11.8\model\e20\im_global.gguf"
MODEL_DIR = r"E:\CUDA11.8\model\Qwen3.6-35B-A3B"
E21_CB = os.path.join(HERE, "stores", "packed2_v1", "e21_cb.npz")
P2_DIR = os.path.join(HERE, "stores", "packed2_sym4")
RESULTS = os.path.join(HERE, "results")
UNIFORM = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float16)
NBINS = 8192
LO, HI = -2.0, 2.0


def boundaries(cb):
    """[4] 级 -> [3] 判定边界。"""
    return (cb[1:] + cb[:-1]) * 0.5


def round_cb_int8(cb):
    """按 pack_tcmoe.py 的容器规则把 [..,4] 码本量化为 int8 x scale，再乘回。"""
    arr = np.asarray(cb, dtype=np.float32).reshape(-1, 4)
    amax = np.abs(arr).max(axis=1)
    scale = np.where(amax > 0.0, amax / 127.0, 1.0)
    c8 = np.clip(np.round(arr / scale[:, None]), -127, 127)
    return (c8 * scale[:, None]).astype(np.float32).reshape(np.shape(cb))


class WeightedHist:
    """同一 (layer, proj) 下所有专家的激活加权 z 直方图（流式累积）。"""

    def __init__(self, bins=NBINS, lo=LO, hi=HI):
        self.step = (hi - lo) / bins
        self.W = np.zeros(bins, dtype=np.float64)
        self.S = np.zeros(bins, dtype=np.float64)
        self.S2 = np.zeros(bins, dtype=np.float64)
        # 分母: 纯 imatrix 权重之和（报告权重空间 MSE 时用，与 E21 口径一致）
        self.I = 0.0

    def add(self, z, a, imp_total):
        """z 归一化权重 [R,C]，a=imp*d^2 逐元素权重 [R,C]，imp_total=sum(imp)。"""
        idx = np.floor((z - LO) / self.step).astype(np.int32)
        np.clip(idx, 0, self.W.shape[0] - 1, out=idx)
        idx = idx.reshape(-1)
        zz = z.reshape(-1).astype(np.float64)
        ww = a.reshape(-1).astype(np.float64)
        self.W += np.bincount(idx, weights=ww, minlength=self.W.shape[0])
        self.S += np.bincount(idx, weights=ww * zz, minlength=self.W.shape[0])
        self.S2 += np.bincount(idx, weights=ww * zz * zz, minlength=self.W.shape[0])
        self.I += float(imp_total)

    def _zbar(self):
        nz = self.W > 0.0
        zbar = np.zeros_like(self.W)
        zbar[nz] = self.S[nz] / self.W[nz]
        return zbar

    def fit(self, iters, tol):
        """用 bin 的精确加权均值喂给 lloyd_grouped，得到共享 4 级码本。"""
        zbar = self._zbar()
        keep = self.W > (self.W.max() * 1e-9)
        vals = np.clip(zbar[keep], LO, HI)[None, :]
        wts = self.W[keep][None, :]
        lv = lloyd_grouped(vals, wts, k=N_LEVEL, iters=iters, tol=tol)[0]
        return np.sort(lv).astype(np.float16)

    def wmse(self, cb):
        """给定码本的权重空间 imatrix 加权 MSE（与 E21 quant_v1 的 msez 同口径）。

        W = imp*d^2，所以 W*(z-cb)^2 已经是 imp*(w-deq)^2；分母用纯 imp 之和。
        """
        cb = np.asarray(cb, dtype=np.float32)
        zbar = self._zbar()
        code = (zbar[:, None] > boundaries(cb)[None, :]).sum(axis=-1)
        within = np.maximum(self.S2 - self.S * self.S / np.maximum(self.W, 1e-300), 0.0)
        err = self.W * (zbar - cb[code]) ** 2 + within
        return float(err.sum() / max(self.I, 1e-30))


def parity_text(cbs):
    return " ".join(
        f"{p}={np.round(cbs[p].astype(np.float32), 3).tolist()}" for p in PROJS)


def collect_layer(reader, im, layer, experts, iters, tol):
    """一层：流式读全部专家的 f32 权重，累积 3 个直方图后拟合 3 张码本。"""
    hists = {p: WeightedHist() for p in PROJS}
    keys = {p: f"blk.{layer}.ffn_{p}_exps.weight" for p in PROJS}
    for e in experts:
        gu = reader.expert(layer, "gate_up_proj", e)
        dn = reader.expert(layer, "down_proj", e)
        for proj, w in (("gate", gu[:MOE_INTER]), ("up", gu[MOE_INTER:]), ("down", dn)):
            w = np.ascontiguousarray(w, dtype=np.float32)
            imp = im[keys[proj]][e].astype(np.float32)
            d = amax_scale(w)                       # fp16 组尺度，与容器存储一致
            d32 = np.maximum(expand_d(d), 1e-12)
            z = w / d32                             # [R,C] 归一化值
            a = np.tile(imp, (w.shape[0], 1)) * d32 ** 2
            hists[proj].add(z, a, float(w.shape[0] * imp.sum()))
            del z, a, d32, d
    cbs = {p: hists[p].fit(iters, tol) for p in PROJS}
    return cbs, hists


def recode_check(layer, experts, im, reader, p2_dir, cb_new):
    """验证换码本后是否必须重分配 code（对比 direct decode 与 recode）。"""
    from pack_tcmoe import Packed2Base
    from bitpack import unpack2

    base = Packed2Base(p2_dir, 40, 256, HIDDEN, MOE_INTER)
    cb_n = round_cb_int8(cb_new)
    keys = {p: f"blk.{layer}.ffn_{p}_exps.weight" for p in PROJS}
    acc = {k: [0.0, 0.0] for k in ("v0", "direct", "recode")}
    raw = {k: [0.0, 0.0] for k in ("v0", "direct", "recode")}
    mismatch = [0, 0]
    for e in experts:
        gu = np.ascontiguousarray(reader.expert(layer, "gate_up_proj", e), dtype=np.float32)
        dn = np.ascontiguousarray(reader.expert(layer, "down_proj", e), dtype=np.float32)
        gu_q = base._part(layer, "gu_q", e)
        gu_s = base._part(layer, "gu_s", e)
        dn_q = base._part(layer, "dn_q", e)
        dn_s = base._part(layer, "dn_s", e)
        cases = (("gate", gu[:MOE_INTER], gu_q[:MOE_INTER], gu_s[:MOE_INTER], 0),
                 ("up", gu[MOE_INTER:], gu_q[MOE_INTER:], gu_s[MOE_INTER:], 1),
                 ("down", dn, dn_q, dn_s, 2))
        for proj, w, qp, sp, pi in cases:
            q_old = unpack2(np.asarray(qp))
            d32 = np.maximum(np.repeat(sp.astype(np.float32), GROUP, axis=1), 1e-12)
            imp = im[keys[proj]][e].astype(np.float32)
            a = np.tile(imp, (w.shape[0], 1)) * d32 ** 2
            deq = {
                "v0": (q_old.astype(np.float32) - 1.5) * d32,
                "direct": cb_n[pi][q_old] * d32,
            }
            q_new = ((w / d32)[..., None] > boundaries(cb_n[pi])[None, None, :]).sum(-1)
            q_new = q_new.astype(np.uint8)
            deq["recode"] = cb_n[pi][q_new] * d32
            for k in acc:
                err = (w - deq[k]) ** 2
                acc[k][0] += float((a * err).sum())
                acc[k][1] += float(a.sum())
                raw[k][0] += float(err.sum())
                raw[k][1] += float(err.size)
            mismatch[0] += int((q_old != q_new).sum())
            mismatch[1] += int(q_old.size)
    out = {
        "layer": layer, "experts": list(experts), "store": p2_dir,
        "weighted_wmse": {k: acc[k][0] / acc[k][1] for k in acc},
        "unweighted_mse": {k: raw[k][0] / raw[k][1] for k in raw},
        "code_mismatch_frac": mismatch[0] / max(mismatch[1], 1),
        "cb_storage": "int8 x scale (与 pack_tcmoe 写入的 tcmoe.cb 一致)",
    }
    d_w = out["weighted_wmse"]["recode"]
    out["gain_recode_vs_direct_pct"] = float(
        100 * (1 - d_w / max(out["weighted_wmse"]["direct"], 1e-30)))
    out["gain_recode_vs_v0_pct"] = float(
        100 * (1 - d_w / max(out["weighted_wmse"]["v0"], 1e-30)))
    out["recode_needed"] = bool(d_w < out["weighted_wmse"]["direct"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix", default=IMATRIX)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--experts", default="0-255")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--e21-cb", default=E21_CB, help="现有 V1 码本（用于 MSE 对照）")
    ap.add_argument("--p2-dir", default=P2_DIR, help="预打包 2bit 库（recode 验证用）")
    ap.add_argument("--out", default=os.path.join(RESULTS, "e21_cb_im.npz"))
    ap.add_argument("--json", default=os.path.join(RESULTS, "e_f10b_cb.json"))
    ap.add_argument("--recode-layer", type=int, default=0)
    ap.add_argument("--recode-experts", default="0-1")
    ap.add_argument("--no-recode", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印规模/内存估算")
    args = ap.parse_args()

    layers = parse_list(args.layers, [])
    experts = parse_list(args.experts, [])
    n_matrix = len(layers) * len(experts) * len(PROJS)
    f32_bytes = len(layers) * len(experts) * 12 * 1024 * 1024
    print(f"[plan] layers={len(layers)} experts={len(experts)} "
          f"matrices={n_matrix} 需读 f32 权重约 {f32_bytes / 2**30:.1f} GiB", flush=True)
    print(f"[plan] hist: {NBINS} bins x 3 数组 x 8 B x 3 proj = "
          f"{3 * 3 * NBINS * 8 / 1024:.0f} KiB/layer", flush=True)
    if args.dry_run:
        print("[dry-run] 不读权重、不拟合", flush=True)
        return

    t0 = time.time()
    im = read_imatrix_diag(args.imatrix)
    print(f"[imatrix] {len(im)} tensors ({time.time() - t0:.0f}s)", flush=True)

    from pack_tcmoe import ExpertReader
    reader = ExpertReader(args.model_dir)

    e21 = None
    if os.path.exists(args.e21_cb):
        e21 = np.load(args.e21_cb)["cb"].astype(np.float16)
        print(f"[e21] {args.e21_cb} {e21.shape}", flush=True)

    cb_all = np.zeros((len(layers), len(PROJS), N_LEVEL), dtype=np.float16)
    rows = {}
    for li, layer in enumerate(layers):
        ts = time.time()
        cbs, hists = collect_layer(reader, im, layer, experts, args.iters, args.tol)
        for pi, proj in enumerate(PROJS):
            cb_all[li, pi] = cbs[proj]
            m_v0 = hists[proj].wmse(UNIFORM)
            m_im = hists[proj].wmse(cbs[proj])
            m_e21 = hists[proj].wmse(e21[layer, pi]) if e21 is not None else None
            row = {"v0": m_v0, "im": m_im, "e21": m_e21,
                   "gain_vs_v0_pct": 100 * (1 - m_im / max(m_v0, 1e-30)),
                   "gain_vs_e21_pct": (100 * (1 - m_im / max(m_e21, 1e-30))
                                       if m_e21 else None),
                   "n_weight": float(hists[proj].I)}
            rows.setdefault(str(layer), {})[proj] = row
        print(f"[fit] l{layer} {parity_text(cbs)} "
              f"wmse(v0/e21/im)=" + " ".join(
                  f"{rows[str(layer)][p]['v0']:.2e}/"
                  f"{(rows[str(layer)][p]['e21'] or float('nan')):.2e}/"
                  f"{rows[str(layer)][p]['im']:.2e}" for p in PROJS)
              + f" ({time.time() - ts:.0f}s, total {time.time() - t0:.0f}s)", flush=True)

    def agg(key):
        vals = [rows[l][p][key] for l in rows for p in PROJS
                if rows[l][p][key] is not None]
        return float(np.mean(vals)) if vals else None

    wmse = {"v0": agg("v0"), "im": agg("im"), "e21": agg("e21")}
    gains = {
        "vs_v0_pct": float(100 * (1 - wmse["im"] / wmse["v0"])) if wmse["v0"] else None,
        "vs_e21_pct": (float(100 * (1 - wmse["im"] / wmse["e21"]))
                       if wmse["e21"] else None),
    }

    recode = None
    if not args.no_recode:
        rl = args.recode_layer
        cb_for = None
        for li, layer in enumerate(layers):
            if layer == rl:
                cb_for = cb_all[li]
        if cb_for is None and os.path.exists(args.out):
            saved = np.load(args.out)["cb"]
            if saved.shape[0] > rl:
                cb_for = saved[rl]
        if cb_for is None:
            print(f"[recode] layer {rl} 未参与本次拟合且无已存 npz，跳过", flush=True)
        else:
            rexp = parse_list(args.recode_experts, [])
            recode = recode_check(rl, rexp, im, reader, args.p2_dir, cb_for)
            print(f"[recode] l{rl} experts={rexp} "
                  f"direct={recode['weighted_wmse']['direct']:.3e} "
                  f"recode={recode['weighted_wmse']['recode']:.3e} "
                  f"v0={recode['weighted_wmse']['v0']:.3e} "
                  f"-> recode_needed={recode['recode_needed']}", flush=True)

    cb_flat = cb_all.reshape(-1, N_LEVEL)
    out = {
        "experiment": "E-F10b",
        "imatrix": args.imatrix,
        "model_dir": args.model_dir,
        "layers": layers,
        "experts": experts,
        "n_layer": len(layers),
        "n_expert": len(experts),
        "n_matrix": n_matrix,
        "group": GROUP,
        "n_level": N_LEVEL,
        "fit": {"objective": "per (layer,proj) pooled z=w/d, weight=imatrix[e][col]*d^2",
                "hist_bins": NBINS, "hist_range": [LO, HI],
                "iters": args.iters, "tol": args.tol},
        "wmse": wmse,
        "gain_pct": gains,
        "per_layer_proj": rows,
        "codebook_mean": np.mean(cb_flat.astype(np.float32), axis=0).tolist(),
        "codebook_std": np.std(cb_flat.astype(np.float32), axis=0).tolist(),
        "codebook_int8_rounded_mean": np.mean(
            round_cb_int8(cb_all).reshape(-1, N_LEVEL), axis=0).tolist(),
        "recode": recode,
        "recode_needed": bool(recode["recode_needed"]) if recode else None,
        "elapsed_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {"layers": layers, "experts": experts, "hist_bins": NBINS,
            "objective": out["fit"]["objective"]}
    np.savez(args.out, cb=cb_all.astype(np.float16),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[e_f10b] saved {args.out} and {args.json} "
          f"({time.time() - t0:.0f}s) gain_vs_v0={gains['vs_v0_pct']:.1f}% "
          f"gain_vs_e21={gains['vs_e21_pct']}", flush=True)


if __name__ == "__main__":
    main()
