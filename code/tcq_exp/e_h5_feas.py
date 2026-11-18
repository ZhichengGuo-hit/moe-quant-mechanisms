#!/usr/bin/env python
"""E-H5 可行性压测（30-60 min）：确定哪些臂能进 16 GB 显存、各长度 prefill 速度。

方法: llama-bench -p <len 列表> -n 0 -r 1，逐作业记录 tok/s；作业期间 2 s 采样 nvidia-smi 显存峰值。
臂:
  gpu  iq2xxs      baselines/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf     -ngl 99
  gpu  ud_q2kxl    ud/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf            -ngl 99
  gpu  ud_q4km     Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99（预计 OOM）→ -ngl 20 重试
  cpu  tcmoe_q4k_nl cmoe/out/model-tcmoe-q4k-nl.gguf             -ngl 0（tcmoe fork 仅 CPU 构建）
        CPU 只测 4k/8k（64k 外推），GPU 测 4k/8k/16k/64k

输出: results/e_h5_feas.json（含每次运行原始 stdout 摘要与显存峰值）
"""
import json
import os
import re
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e_h5_feas.json")
LOG = os.path.join(RES, "h5_feas.log")
GPU_BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-bench.exe"
CPU_BIN = r"E:\CUDA11.8\train\llama.cpp\build-tcmoe\bin\Release\llama-bench.exe"
M = {
    "iq2xxs": r"E:\CUDA11.8\model\baselines\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
    "ud_q2kxl": r"E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",
    "ud_q4km": r"E:\CUDA11.8\model\Qwen3.6-35B-A3B\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "tcmoe_q4k_nl": r"E:\CUDA11.8\train\tcmoe\out\model-tcmoe-q4k-nl.gguf",
}
GPU_LENS = "4096,8192,16384,65536"
CPU_LENS = "4096,8192"

JOBS = [
    ("gpu", "iq2xxs", GPU_LENS, 99),
    ("gpu", "ud_q2kxl", GPU_LENS, 99),
    ("gpu", "ud_q4km", "4096,8192,16384", 20),
    ("cpu", "tcmoe_q4k_nl", CPU_LENS, 0),
]

# 人工判定（不重跑）: ud_q4km -ngl 99 在 Windows WDDM 下超卖到共享内存，
# nvidia-smi 显示 15969/16384 MiB 且 util 100%，4k prefill 逾 10 min 未完成 → 不可用。
MANUAL = [{"backend": "gpu", "tag": "ud_q4km", "ngl": 99, "verdict": "infeasible",
           "note": "WDDM 超卖: VRAM 15969/16384 MiB, 4k prefill >10min 未完, 已终止"}]


def say(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class VramWatch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10)
                v = int(out.stdout.strip().splitlines()[0])
                self.peak = max(self.peak, v)
            except Exception:
                pass
            time.sleep(2)


def parse(stdout):
    """从 llama-bench 表格解析 pp 速度: 行形如 | qwen... | 4096 | 0 | ... | 123.45 ± ... |"""
    rows = []
    for line in stdout.splitlines():
        if not line.startswith("|") or "model" in line.lower() or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6:
            continue
        try:
            n_prompt = int(cells[1].replace(",", ""))
            n_gen = int(cells[2].replace(",", ""))
        except ValueError:
            continue
        m = re.findall(r"[0-9]+\.[0-9]+", cells[-1])
        if not m:
            continue
        rows.append({"n_prompt": n_prompt, "n_gen": n_gen, "tok_s": float(m[0])})
    return rows


def main():
    os.makedirs(RES, exist_ok=True)
    done = set()
    res = []
    if os.path.exists(OUT):
        prev = json.load(open(OUT))
        for r in prev.get("jobs", []):
            res.append(r)
            if r.get("rc") == 0:
                done.add((r["backend"], r["tag"], r["ngl"]))
    for backend, tag, lens, ngl in JOBS:
        if (backend, tag, ngl) in done:
            say("skip %s %s -ngl %d (已完成)" % (backend, tag, ngl))
            continue
        binp = GPU_BIN if backend == "gpu" else CPU_BIN
        cmd = [binp, "-m", M[tag], "-p", lens, "-n", "0", "-r", "1",
               "-ngl", str(ngl), "-t", "16", "-o", "json"]
        say("run %s %s -ngl %d -p %s" % (backend, tag, ngl, lens))
        t0 = time.time()
        w = VramWatch()
        w.start()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=3000)
            out = p.stdout
            rst = p.stderr
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            rst = "TIMEOUT 50min"
            p = type("P", (), {"returncode": -9})()
        w.stop = True
        time.sleep(1)
        rows = parse(out) or parse(rst)
        j = None
        if out.strip().startswith("["):
            try:
                j = json.loads(out)
            except ValueError:
                j = None
        if j:
            rows = [{"n_prompt": x.get("n_prompt"), "n_gen": x.get("n_gen"),
                     "tok_s": x.get("avg_ts"), "std": x.get("std_ts"),
                     "backend": x.get("backend")} for x in j]
        rec = {"backend": backend, "tag": tag, "ngl": ngl, "lens": lens,
               "rc": p.returncode, "dur_s": round(time.time() - t0, 1),
               "vram_peak_mb": w.peak, "rows": rows,
               "err": rst.strip()[-600:] if p.returncode != 0 else ""}
        res.append(rec)
        say("  rc=%d %.0fs vram=%d MB rows=%s" % (
            rec["rc"], rec["dur_s"], rec["vram_peak_mb"],
            [(r["n_prompt"], r["tok_s"]) for r in rows]))
        json.dump({"jobs": res, "manual": MANUAL}, open(OUT, "w"), indent=1, ensure_ascii=False)
    say("DONE -> %s" % OUT)


if __name__ == "__main__":
    main()
