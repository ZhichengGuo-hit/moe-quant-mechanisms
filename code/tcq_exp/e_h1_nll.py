#!/usr/bin/env python
"""E-H1: NLL eval64 三域评测 + 两两 bootstrap 判定。

主指标: 每配置 × 每域的 chunk 级 mean NLL（llama-perplexity, ctx=512, --ppl-output-type 1）。
以 per-chunk NLL 做配对 bootstrap（同一语料的 chunk 划分对所有模型完全相同）。
判定: 同域两两 |ΔNLL| < 0.005 -> “受控同预算消融未发现可分辨贡献”。

用法:
  python e_h1_nll.py run            # 跑所有已构建配置（跳过已完成）
  python e_h1_nll.py report         # 汇总 + bootstrap -> results/e_h1_nll.json
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-perplexity.exe"
MODEL_DIR = r"E:\CUDA11.8\model\e20\h1"
DATA = os.path.join(HERE, "data")
WORK = os.path.join(HERE, "h1_nll")
RES = os.path.join(HERE, "results")
LOGS = os.path.join(RES, "h1_nll_logs")
OUT = os.path.join(RES, "e_h1_nll.json")
DOMS = ["math", "code", "general"]
ARMS = ["g0", "g1", "g2", "g3", "g4"]
MODES = ["on", "off"]
CHUNK = 512
SEED = 20261108
NBOOT = 10000
JUDGE = 0.005
LINE_RE = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*$")


def corpus_path(dom):
    return os.path.join(WORK, "eval64_%s.txt" % dom)


def make_corpora():
    os.makedirs(WORK, exist_ok=True)
    for dom in DOMS:
        src = os.path.join(DATA, "%s_eval64.jsonl" % dom)
        texts = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    texts.append(json.loads(line)["text"])
        with open(corpus_path(dom), "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts) + "\n")
        print("corpus %s: %d texts, %d bytes" % (dom, len(texts), os.path.getsize(corpus_path(dom))), flush=True)


def model_path(tag):
    return os.path.join(MODEL_DIR, "e_h1_%s.gguf" % tag)


def run_one(tag, dom):
    out = os.path.join(WORK, "%s_%s.json" % (tag, dom))
    if os.path.exists(out):
        return
    m = model_path(tag)
    if not os.path.exists(m):
        return
    cmd = [BIN, "-m", m, "-f", corpus_path(dom), "-c", str(CHUNK), "-b", "512",
           "-ngl", "99", "-t", "16", "--ppl-output-type", "1", "--chunks", "-1"]
    log = os.path.join(LOGS, "ppl_%s_%s.log" % (tag, dom))
    t0 = time.time()
    print("[ppl] %s %s ..." % (tag, dom), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    with open(log, "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n[stderr]\n" + p.stderr)
    chunks = []
    for line in (p.stdout + "\n" + p.stderr).splitlines():
        mm = LINE_RE.match(line)
        if mm:
            chunks.append({"off": int(mm.group(1)), "ppl": float(mm.group(2)),
                           "av_nll": float(mm.group(3))})
    if not chunks:
        print("[warn] no chunks parsed for %s %s (rc=%d)" % (tag, dom, p.returncode), flush=True)
        return
    toks = []
    for i, c in enumerate(chunks):
        toks.append(CHUNK - 1 if c["off"] == 0 else CHUNK // 2)
    tot = sum(toks)
    mean_nll = sum(c["av_nll"] * t for c, t in zip(chunks, toks)) / tot
    json.dump({"tag": tag, "domain": dom, "mean_nll": mean_nll, "chunks": chunks,
               "tokens_per_chunk": toks, "dur_s": round(time.time() - t0, 1)},
              open(out, "w"), indent=1)
    print("[ppl] %s %s  mean_nll=%.5f  chunks=%d  %.0fs" % (tag, dom, mean_nll, len(chunks), time.time() - t0), flush=True)


def dv(tag, dom):
    p = os.path.join(WORK, "%s_%s.json" % (tag, dom))
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def report():
    rows = {}
    for dom in DOMS:
        rows[dom] = {}
        for mode in MODES:
            for arm in ARMS:
                d = dv("%s_im%s" % (arm, mode), dom)
                if d:
                    rows[dom]["%s/im%s" % (arm, mode)] = d
    out = {"judge_threshold": JUDGE, "nboot": NBOOT, "seed": SEED, "domains": {},
           "pairs": {}}
    rng = np.random.default_rng(SEED)
    for dom in DOMS:
        if not rows[dom]:
            continue
        # 等长校验: 所有配置 chunk 数相同
        tags = list(rows[dom])
        n = min(len(rows[dom][t]["chunks"]) for t in tags)
        toks = np.array(rows[dom][tags[0]]["tokens_per_chunk"][:n], dtype=float)
        vals = {t: np.array([c["av_nll"] for c in rows[dom][t]["chunks"][:n]], dtype=float)
                for t in tags}
        out["domains"][dom] = {t: {"mean_nll": float(np.sum(vals[t] * toks) / toks.sum()),
                                    "mean_ppl": float(np.exp(np.sum(vals[t] * toks) / toks.sum())),
                                    "n_chunks": int(n)} for t in tags}
        # 配对 bootstrap
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = tags[i], tags[j]
                key = "%s|%s|%s" % (dom, a, b)
                d = vals[a] - vals[b]
                point = float(np.sum(d * toks) / toks.sum())
                idx = rng.integers(0, n, size=(NBOOT, n))
                bs = np.array([np.sum(d[k] * toks[k]) / toks[k].sum() for k in idx])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                out["pairs"][key] = {"diff_nll": point,
                                     "ci95": [float(lo), float(hi)],
                                     "undetectable": bool(abs(point) < JUDGE)}
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT)
    for dom in DOMS:
        print("--- %s ---" % dom)
        for t, v in out["domains"].get(dom, {}).items():
            print("  %-12s nll=%.5f ppl=%.4f" % (t, v["mean_nll"], v["mean_ppl"]))
    for k, v in out["pairs"].items():
        print("  %-42s d=%+.5f ci=[%+.5f,%+.5f] %s" % (
            k, v["diff_nll"], v["ci95"][0], v["ci95"][1],
            "<0.005" if v["undetectable"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["corpus", "run", "report", "all"])
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if a.cmd in ("corpus", "all"):
        make_corpora()
    if a.cmd in ("run", "all"):
        make_corpora()
        for mode in MODES:
            for arm in ARMS:
                for dom in DOMS:
                    run_one("%s_im%s" % (arm, mode), dom)
    if a.cmd in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
