#!/usr/bin/env python
"""E-H4 补测：工作集四列表 + 冷/热切换（无额外精度评测，纯资源量测）

四列表（每臂）:
  1) 文件体积         os.stat 字节
  2) 实际加载量       GGUF 张量 nbytes 之和（真正进内存/显存的权重字节）
  3) 驻留工作集       server 起来后、未发请求时的进程 WorkingSet
  4) 单请求工作集     一次 4k prompt 请求期间的峰值 WorkingSet 增量

冷/热切换（仅多域 tcmoe 文件）:
  依次请求 global → math → base → global → math，记录每次请求的耗时；
  换域后第一次 = 冷（要重建 graph），同域重复 = 热。

用法: python e_h4_workset.py [--port 18401] [--skip-cpu]
输出: results/e_h4_workset.json
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

sys.path.insert(0, r"E:\CUDA11.8\train\llama.cpp\gguf-py")
from gguf import GGUFReader

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LOG = os.path.join(RES, "h4_logs")
BIN_CUDA = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-server.exe"
BIN_CPU = r"E:\CUDA11.8\train\llama.cpp\build-tcmoe\bin\Release\llama-server.exe"
TCMOE = r"E:\CUDA11.8\train\tcmoe\out\model-tcmoe-q4k-nl.gguf"

ARMS = [
    {"tag": "iq2xxs", "path": r"E:\CUDA11.8\model\baselines\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
     "ngl": 99, "bin": BIN_CUDA, "domains": None},
    {"tag": "ud_q2kxl", "path": r"E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",
     "ngl": 99, "bin": BIN_CUDA, "domains": None},
    {"tag": "ud_q4km", "path": r"E:\CUDA11.8\model\Qwen3.6-35B-A3B\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
     "ngl": 20, "bin": BIN_CUDA, "domains": None},
    {"tag": "tcmoe_q4k_nl", "path": TCMOE, "ngl": 0, "bin": BIN_CPU,
     "domains": ["global", "math", "base", "global", "math"]},
]
CTX = 8192
PROMPT_TOK_TARGET = 4096


def log(msg):
    print(msg, flush=True)
    os.makedirs(LOG, exist_ok=True)
    with open(os.path.join(LOG, "h4_workset.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def gguf_bytes(path):
    r = GGUFReader(path)
    n = sum(int(t.n_bytes) for t in r.tensors)
    return len(r.tensors), n


def corpus_prompt(chars):
    parts = []
    for p in (os.path.join(HERE, "data", "prose_test.jsonl"),
              os.path.join(HERE, "data", "general_eval.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts.append(json.loads(line)["text"].strip())
    out, n = [], 0
    i = 0
    while n < chars:
        s = parts[i % len(parts)]
        out.append(s)
        n += len(s) + 1
        i += 1
    return "\n".join(out)


class Watch:
    """后台采样进程 WorkingSet 峰值"""

    def __init__(self, proc):
        self.proc = proc
        self.peak = 0
        self.run_flag = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        p = psutil.Process(self.proc.pid)
        while self.run_flag:
            try:
                ws = p.memory_info().rss
                self.peak = max(self.peak, ws)
            except Exception:
                return
            time.sleep(0.4)

    def stop(self):
        self.run_flag = False
        self.t.join(timeout=2)


def post(port, path, payload, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(port, text, domain=None, max_tokens=32):
    body = {"messages": [{"role": "user", "content": text}],
            "temperature": 0, "top_k": 1, "max_tokens": max_tokens,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if domain:
        body["domain"] = domain
    t0 = time.time()
    d = post(port, "/v1/chat/completions", body)
    return time.time() - t0, int(d.get("usage", {}).get("prompt_tokens") or 0)


def run_arm(arm, port):
    os.makedirs(LOG, exist_ok=True)
    rec = {"tag": arm["tag"], "ngl": arm["ngl"], "file_bytes": os.path.getsize(arm["path"])}
    n_tensor, wbytes = gguf_bytes(arm["path"])
    rec.update({"n_tensor": n_tensor, "weight_bytes": wbytes})

    lf = open(os.path.join(LOG, "server_%s.log" % arm["tag"]), "w", encoding="utf-8")
    cmd = [arm["bin"], "-m", arm["path"], "-c", str(CTX), "-ngl", str(arm["ngl"]),
           "-t", "16", "--host", "127.0.0.1", "--port", str(port), "-np", "1"]
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    w = Watch(proc)
    t0 = time.time()
    ok = False
    while time.time() - t0 < 900:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as r:
                if json.loads(r.read().decode()).get("status") == "ok":
                    ok = True
                    break
        except Exception:
            time.sleep(3)
    rec["load_s"] = round(time.time() - t0, 1)
    rec["peak_ws_load"] = w.peak
    try:
        p = psutil.Process(proc.pid)
        rec["ws_after_load"] = p.memory_info().rss
    except Exception:
        rec["ws_after_load"] = None
    if not ok:
        rec["error"] = "server not up"
        w.stop()
        proc.terminate()
        return rec

    prompt = corpus_prompt(PROMPT_TOK_TARGET * 4)

    # 单请求工作集：一次 4k 冷却请求
    w.peak = rec["ws_after_load"] or 0
    try:
        dt, ptok = chat(port, prompt)
        rec["req4k_s"] = round(dt, 1)
        rec["req4k_prompt_tokens"] = ptok
        rec["peak_ws_req4k"] = w.peak
        rec["ws_after_req4k"] = p.memory_info().rss
        rec["per_request_ws"] = w.peak - (rec["ws_after_load"] or 0)
    except Exception as e:
        rec["req4k_error"] = str(e)[:200]

    # 冷/热切换
    if arm["domains"]:
        sw = []
        for i, d in enumerate(arm["domains"]):
            w.peak = p.memory_info().rss
            try:
                dt, ptok = chat(port, prompt, domain=d)
                sw.append({"domain": d, "seq": i, "wall_s": round(dt, 2),
                           "prompt_tokens": ptok, "peak_ws": w.peak,
                           "rss_after": p.memory_info().rss})
                log("[%s] switch %d -> %s  %.1fs" % (arm["tag"], i, d, dt))
            except Exception as e:
                sw.append({"domain": d, "seq": i, "error": str(e)[:160]})
        rec["switches"] = sw

    w.stop()
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    lf.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18401)
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    arms = ARMS
    if a.skip_cpu:
        arms = [x for x in arms if not x["domains"]]
    if a.only:
        want = set(a.only.split(","))
        arms = [x for x in arms if x["tag"] in want]

    out = {"protocol": {"ctx": CTX, "prompt_tok_target": PROMPT_TOK_TARGET,
                        "ws_metric": "psutil rss = Windows WorkingSetSize",
                        "sample_interval_s": 0.4, "arms": [x["tag"] for x in arms]},
           "arms": []}
    for i, arm in enumerate(arms):
        log("=== %s ===" % arm["tag"])
        rec = run_arm(arm, a.port + i)
        out["arms"].append(rec)
        log(json.dumps({k: v for k, v in rec.items() if k not in ("switches",)},
                       ensure_ascii=False))
        if rec.get("switches"):
            for s in rec["switches"]:
                log("   " + json.dumps(s, ensure_ascii=False))
        with open(os.path.join(RES, "e_h4_workset.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)

    log("wrote results/e_h4_workset.json")
    for r in out["arms"]:
        g = lambda k: r.get(k)
        log("%-12s file=%.2fGiB weights=%.2fGiB load=%.0fs ws_load=%.2fGiB req+%.2fGiB (%.1fs)" % (
            r["tag"], (g("file_bytes") or 0) / 2 ** 30, (g("weight_bytes") or 0) / 2 ** 30,
            (g("load_s") or 0), (g("ws_after_load") or 0) / 2 ** 30,
            (g("per_request_ws") or 0) / 2 ** 30, (g("req4k_s") or 0)))


if __name__ == "__main__":
    main()
