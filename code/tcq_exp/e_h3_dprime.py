#!/usr/bin/env python
"""E-H3b: Xing D′ —— C 类型 + 通用语料 imatrix（与 D 的三域/外部 calib 对照）。

步骤:
  1. 通用语料 = data_e20/general.txt + data/general_calib.jsonl 的 text
  2. llama-imatrix (Xing F16) -> xing4_0-29b-imatrix-general.gguf
  3. C 类型文件 (e_g3_types_c.txt, base Q4_K) + 新 imatrix -> xing e_h3_dprime.gguf
  4. GPQA 100 题评测 (tag xing_Dprime, prefix e_g3_xing_)
  5. C vs D′ 配对 McNemar -> results/e_h3_xing_C_vs_Dprime.json
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin"
XING = r"E:\CUDA11.8\model\xing"
F16 = os.path.join(XING, "Xing4.0-29B-A4B-f16.gguf")
IM_NEW = os.path.join(XING, "xing4_0-29b-imatrix-general.gguf")
OUT = os.path.join(XING, "e_h3_dprime.gguf")
TYPES = os.path.join(HERE, "results", "e_g3_types_c.txt")
WORK = os.path.join(HERE, "h3")
RES = os.path.join(HERE, "results")
PY = sys.executable
TAG = "xing_Dprime"
PREFIX = "e_g3_xing_"


def sh(cmd, env=None):
    print("$ " + " ".join(cmd), flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, env=e).returncode


def corpus():
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, "general_corpus.txt")
    if os.path.exists(p):
        return p
    parts = []
    g = os.path.join(HERE, "data_e20", "general.txt")
    if os.path.exists(g):
        parts.append(open(g, encoding="utf-8").read())
    calib = os.path.join(HERE, "data", "general_calib.jsonl")
    if os.path.exists(calib):
        for line in open(calib, encoding="utf-8"):
            if line.strip():
                parts.append(json.loads(line)["text"])
    open(p, "w", encoding="utf-8").write("\n\n".join(parts) + "\n")
    print("corpus %s: %d bytes" % (p, os.path.getsize(p)), flush=True)
    return p


def main():
    c = corpus()
    if not os.path.exists(IM_NEW):
        # 58 GiB F16 在 16 GiB 卡上不能显式 -ngl 99（会尝试整模上卡）；与 G3 一致用默认 auto
        rc = sh([os.path.join(BIN, "llama-imatrix.exe"), "-m", F16, "-f", c,
                 "-o", IM_NEW, "-c", "512", "-t", "16", "--chunks", "-1"])
        if rc != 0:
            print("imatrix failed rc=%d" % rc)
            return rc
    if not os.path.exists(OUT):
        rc = sh([os.path.join(BIN, "llama-quantize.exe"),
                 "--tensor-type-file", TYPES, "--imatrix", IM_NEW,
                 F16, OUT, "Q4_K", "16"])
        if rc != 0:
            print("quantize failed rc=%d" % rc)
            return rc
    env = {"E_G_PREFIX": PREFIX, "E_G_BASELINE": "C_roleq4"}
    rc = sh([PY, os.path.join(HERE, "e_g1_eval.py"), "run", "--tag", TAG,
             "--model", OUT, "--task", "gpqa", "--port", "18301",
             "--bin-dir", BIN, "--block", "h3", "--ctx", "4096",
             "--n-predict", "1024"], env=env)
    if rc != 0:
        print("gpqa eval failed rc=%d" % rc)
        return rc
    sh([PY, os.path.join(HERE, "e_g1_eval.py"), "score", "--baseline", "C_roleq4"], env=env)
    a = os.path.join(RES, "%sC_roleq4_gpqa.jsonl" % PREFIX)
    b = os.path.join(RES, "%s%s_gpqa.jsonl" % (PREFIX, TAG))
    if os.path.exists(a) and os.path.exists(b):
        sh([PY, os.path.join(HERE, "e_h3_mcnemar.py"), "--a", a, "--b", b,
            "--label-a", "C", "--label-b", "Dprime",
            "--out", os.path.join(RES, "e_h3_xing_C_vs_Dprime.json")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
