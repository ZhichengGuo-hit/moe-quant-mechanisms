#!/usr/bin/env python
"""E-H1: 十配置量化构建（5 臂 × imatrix 开/关）

im-on  : --imatrix E:\\CUDA11.8\\model\\e20\\im_global2.gguf  (E20 三域交错；blk.40/nextn 由 e_h1_im_extend.py 补齐)
im-off : --imatrix E:\\CUDA11.8\\model\\e20\\im_ones2.gguf    (全 1；b8746 对 IQ 族强制要求 imatrix，
         全 1 在数值上等价于不提供)
产物校验: rc==0 且张量数/元素数与源模型一致，否则删除并记 fail（避免截断文件被当作成功）

用法:
  python e_h1_build.py                 # 全部 10 个配置，串行
  python e_h1_build.py --arms g0,g1 --im on
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin"
SRC = r"E:\CUDA11.8\model\e20\qwen36_f16.gguf"
OUT_DIR = r"E:\CUDA11.8\model\e20\h1"
TYPES = os.path.join(HERE, "h1_types")
RES = os.path.join(HERE, "results")
LOGS = os.path.join(RES, "h1_logs")
STATE = os.path.join(RES, "e_h1_build_state.json")
META = os.path.join(RES, "e_h1_types_meta.json")
IM = {"on": r"E:\CUDA11.8\model\e20\im_global2.gguf",
      "off": r"E:\CUDA11.8\model\e20\im_ones2.gguf"}
TARGET, TOL = 11.45, 0.30

GGUF_PY = r"E:\CUDA11.8\train\llama.cpp\gguf-py"
sys.path.insert(0, GGUF_PY)
from gguf import GGUFReader


def src_sig():
    r = GGUFReader(SRC)
    return len(r.tensors), sum(int(t.n_elements) for t in r.tensors)


def valid_gguf(path, sig):
    try:
        r = GGUFReader(path)
    except Exception:
        return False
    n = len(r.tensors)
    e = sum(int(t.n_elements) for t in r.tensors)
    return n == sig[0] and e == sig[1]


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=1, ensure_ascii=False)


def run_one(arm, immode, meta, st, sig):
    tag = "%s_im%s" % (arm, immode)
    out = os.path.join(OUT_DIR, "e_h1_%s.gguf" % tag)
    if os.path.exists(out) and os.path.getsize(out) > 10 * 2**30:
        if valid_gguf(out, sig):
            print("[skip] %s exists %.2f GiB valid" % (tag, os.path.getsize(out) / 2**30), flush=True)
            return
        print("[rebuild] %s 不完整（张量数与源不符），删除重跑" % tag, flush=True)
        os.remove(out)
    cmd = [os.path.join(BIN, "llama-quantize.exe"),
           "--tensor-type-file", os.path.join(TYPES, "e_h1_%s.txt" % arm),
           "--imatrix", IM[immode],
           SRC, out, meta["arms"][arm]["base_type"], "16"]
    log = os.path.join(LOGS, "quant_%s.log" % tag)
    t0 = time.time()
    print("[build] %s ... -> %s" % (tag, log), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dur = (time.time() - t0) / 60
    ok = p.returncode == 0 and os.path.exists(out) and valid_gguf(out, sig)
    if not ok and os.path.exists(out):
        os.remove(out)
    g = os.path.getsize(out) / 2**30 if ok else 0
    st[tag] = {"status": "ok" if ok else "fail", "rc": p.returncode,
               "dur_min": round(dur, 1), "size_gib": round(g, 3), "path": out,
               "within_tol": bool(ok and abs(g - TARGET) <= TOL)}
    save_state(st)
    print("[done] %s rc=%d %.1f min %.3f GiB (%s)" % (
        tag, p.returncode, dur, g, "OK" if st[tag]["within_tol"] else "FAIL/OUT"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="g0,g1,g2,g3,g4")
    ap.add_argument("--im", default="on,off")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    meta = json.load(open(META))
    st = load_state()
    sig = src_sig()
    print("src sig: tensors=%d elements=%d" % sig, flush=True)
    for arm in args.arms.split(","):
        for immode in args.im.split(","):
            run_one(arm.strip(), immode.strip(), meta, st, sig)
    print("ALL DONE", json.dumps({k: v.get("status") for k, v in st.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
