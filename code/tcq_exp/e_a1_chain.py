#!/usr/bin/env python
"""E-A1 队列编排: 等 A1c 完成 -> A1a (BFCL-v3, EvalScope) -> A1b (LongMemEval).

幂等: 各阶段以 results/.a1c_done / .a1a_done / .a1b_done / .a1_all_done 为完成标记;
已完成的阶段直接跳过. 日志 results/a1_chain.log.

    python e_a1_chain.py              # 全队列
    python e_a1_chain.py --only a1a   # 只补跑某段
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PY_A1 = r"E:\CUDA11.8\train\a1_env311\Scripts\python.exe"
PY = r"C:\Users\administered\miniconda3\python.exe"
LOG = os.path.join(RES, "a1_chain.log")


def log(msg):
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
    print(line, flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def marker(name):
    return os.path.join(RES, "." + name)


def wait_file(path, timeout_h, what):
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        if os.path.exists(path):
            log("WAIT %s ok" % what)
            return True
        time.sleep(120)
    log("WAIT %s 超时 (%.0fh), 继续等待不启动" % (what, timeout_h))
    return False


def wait_no_server(timeout_min=30):
    """等残留 llama-server 退出 (前一段收尾)."""
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"],
                             capture_output=True, text=True, errors="replace",
                             stdin=subprocess.DEVNULL).stdout or ""
        if "llama-server.exe" not in out:
            return
        time.sleep(10)


def run(cmd):
    log("$ " + " ".join(cmd))
    with open(LOG, "a", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
        f.write("[exit %d]\n" % p.returncode)
        f.flush()
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "a1a", "a1b"])
    args = ap.parse_args()

    if args.only in ("all", "a1a") and not os.path.exists(marker("a1a_done")):
        wait_file(marker("a1c_done"), 20, "a1c")
        wait_no_server()
        # 每子集 20 条: tcmoe CPU 臂 ~5 min/条 (32 线程实测), 六臂统一限流 → 全队列约 11-12 h
        rc = run([PY_A1, os.path.join(HERE, "e_a1a_bfcl.py"), "--limit", "20"])
        if rc == 0:
            open(marker("a1a_done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        else:
            log("A1a 失败 rc=%d, 停止队列" % rc)
            return 1

    if args.only in ("all", "a1b") and not os.path.exists(marker("a1b_done")):
        wait_no_server()
        rc = run([PY, os.path.join(HERE, "e_a1b_longmemeval.py"), "--phase", "all"])
        if rc == 0:
            open(marker("a1b_done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        else:
            log("A1b 失败 rc=%d" % rc)
            return 1

    if args.only == "all":
        open(marker("a1_all_done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        log("E-A1 队列全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
