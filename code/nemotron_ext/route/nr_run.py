"""E-R2 Nemotron 臂采集：i1-Q3_K_M vs i1-Q6_K 的逐 token 路由。

用法: python nr_run.py [--force]
"""
import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WORK = os.path.join(HERE, "work")
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "tcq_exp", "data"))
MODEL = r"E:\CUDA11.8\model\nemotron"
BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-route-dump.exe"
SEP = "%%__TEXT_BOUNDARY__%%"
PER_DOMAIN = 8
CONFIGS = {
    "nemotron_q3km": "Nemotron-Cascade-2-30B-A3B.i1-Q3_K_M.gguf",
    "nemotron_q6k": "Nemotron-Cascade-2-30B-A3B.i1-Q6_K.gguf",
}


def say(m):
    print("%s %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def build_texts():
    os.makedirs(WORK, exist_ok=True)
    parts = []
    for d in ["math", "code", "general"]:
        with open(os.path.join(DATA, "%s_eval64.jsonl" % d), encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()][:PER_DOMAIN]
        parts += [r["text"] for r in rows]
    path = os.path.join(WORK, "eval8x3.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SEP.join(parts))
    say("[texts] %d" % len(parts))
    return path


def run(tag, file, texts, force=False):
    probs = os.path.join(RES, "probs_%s.bin" % tag)
    nll = os.path.join(RES, "nll_%s.bin" % tag)
    log = os.path.join(RES, "run_%s.log" % tag)
    if not force and os.path.exists(probs) and os.path.getsize(probs) > 0:
        say("[%s] exists, skip" % tag)
        return
    env = dict(os.environ)
    env["LLAMA_MOE_DUMP"] = probs
    cmd = [BIN, "-m", os.path.join(MODEL, file), "-c", "4096", "-b", "1024", "-ub", "1024",
           "-ngl", "20", "-t", "16", "--threads-batch", "16",
           "--texts", texts, "--out", nll, "--tag", tag]
    say("[%s] " % tag + " ".join(cmd))
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        lf.write(" ".join(cmd) + "\n")
        lf.flush()
        p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError("%s failed rc=%d" % (tag, p.returncode))
    say("[%s] done %.1f min" % (tag, (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    texts = build_texts()
    for tag, f in CONFIGS.items():
        run(tag, f, texts, args.force)


if __name__ == "__main__":
    main()
