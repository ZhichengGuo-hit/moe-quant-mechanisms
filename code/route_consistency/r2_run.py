"""E-R2 采集：对三个配置导出 eval64 三域的逐 token 路由概率 + NLL。

用法:
    python r2_run.py --config bf16|tcq|iq2xxs [--force]

产物 (results/):
    probs_<tag>.bin   llama_context 导出 (per ubatch: n_tokens, token ids, 每层 [256, n])
    nll_<tag>.bin     route-dump 导出 (per text: tokens, nll)
    run_<tag>.log     运行日志
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WORK = os.path.join(HERE, "work")
DATA = os.path.abspath(os.path.join(HERE, "..", "tcq_exp", "data"))
MODEL = r"E:\CUDA11.8\model"
SEP = "%%__TEXT_BOUNDARY__%%"
DOMAINS = ["math", "code", "general"]

CONFIGS = {
    "bf16": {
        "bin": r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-route-dump.exe",
        "model": os.path.join(MODEL, "e20", "qwen36_f16.gguf"),
        "ngl": 0,
    },
    "tcq": {
        "bin": r"E:\CUDA11.8\train\llama.cpp\build-tcmoe\bin\Release\llama-route-dump.exe",
        "model": r"E:\CUDA11.8\train\tcmoe\out\model-tcmoe-q4k-nl.gguf",
        "ngl": 0,
    },
    "iq2xxs": {
        "bin": r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-route-dump.exe",
        "model": os.path.join(MODEL, "baselines", "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"),
        "ngl": 0,
    },
}


def say(m):
    print("%s %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def build_texts():
    """全部 192 条文本按域顺序写入一个文件，并记录每条的域与字符数。"""
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, "eval64_all.txt")
    meta = []
    parts = []
    for d in DOMAINS:
        src = os.path.join(DATA, "%s_eval64.jsonl" % d)
        with open(src, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for i, r in enumerate(rows):
            parts.append(r["text"])
            meta.append({"domain": d, "idx": i, "chars": len(r["text"])})
    with open(path, "w", encoding="utf-8") as f:
        f.write(SEP.join(parts))
    with open(os.path.join(WORK, "texts_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    say("[texts] %d texts -> %s (%.1f MB)" % (len(parts), path, os.path.getsize(path) / 1e6))
    return path, meta


def run_config(tag, texts_path, force=False):
    cfg = CONFIGS[tag]
    os.makedirs(RES, exist_ok=True)
    probs = os.path.join(RES, "probs_%s.bin" % tag)
    nll = os.path.join(RES, "nll_%s.bin" % tag)
    log = os.path.join(RES, "run_%s.log" % tag)
    if not force and os.path.exists(probs) and os.path.exists(nll) and os.path.getsize(probs) > 0:
        say("[%s] 已存在，跳过 (--force 重跑)" % tag)
        return probs, nll
    for p in (probs, nll):
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ)
    env["LLAMA_MOE_DUMP"] = probs
    cmd = [cfg["bin"], "-m", cfg["model"], "-c", "4096", "-b", "1024", "-ub", "1024",
           "-ngl", str(cfg["ngl"]), "-t", "16", "--threads-batch", "16",
           "--texts", texts_path, "--out", nll, "--tag", tag]
    say("[%s] %s" % (tag, " ".join(cmd)))
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        lf.write(" ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError("config %s failed rc=%d, see %s" % (tag, proc.returncode, log))
    if not os.path.exists(probs) or os.path.getsize(probs) == 0:
        raise RuntimeError("probs file missing for %s" % tag)
    say("[%s] done %.1f min: probs %.1f MB, nll %.1f MB"
        % (tag, dt / 60, os.path.getsize(probs) / 1e6, os.path.getsize(nll) / 1e6))
    return probs, nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all", choices=["all"] + list(CONFIGS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    texts, _ = build_texts()
    tags = list(CONFIGS) if args.config == "all" else [args.config]
    for t in tags:
        run_config(t, texts, args.force)
    say("all done")


if __name__ == "__main__":
    main()
