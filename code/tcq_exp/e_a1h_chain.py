#!/usr/bin/env python
"""E-A1h: cwd_hint 干预, 六臂 x 两子集 = 12 个作业顺序执行, 与 A1a 基线对比.

每作业 = 一个独立 llama-server + EvalScope bfcl_v3 (limit 20, seed 20261111), 通过
e_a1g_intervene.py --mode cwd_hint --arm <臂> --subset <子集> 执行.
- 作业开始前确认 18141/18142 端口释放且无 llama-server 残留, 同一时间只起一个 server;
- 单作业异常/rc!=0/无 review 或 90 min 无新增 prediction 视为失败, 最多重试 1 次, 记录后继续;
- 全部结束写 results/a1h_summary.json (行为指标口径同 e_a1f_decomp: 从 reviews 的 messages
  统计; cd 错误 = tool content 匹配 ``cd:.*No such file``).

    E:\\CUDA11.8\\train\\a1_env311\\Scripts\\python.exe e_a1h_chain.py
    E:\\CUDA11.8\\train\\a1_env311\\Scripts\\python.exe e_a1h_chain.py --redo base2:multi_turn_base
"""
import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time

import e_a1a_bfcl as A

HERE = os.path.dirname(os.path.abspath(__file__))
RES = A.RES
PY = r"E:\CUDA11.8\train\a1_env311\Scripts\python.exe"
RUNNER = os.path.join(HERE, "e_a1g_intervene.py")
SUMMARY = os.path.join(RES, "a1h_summary.json")

ARMS = ["ud_q4km", "iq2xxs", "ud_q2kxl", "tcq_nl_global", "tcq_nl_math", "base2"]
SUBSETS = ["multi_turn_base", "multi_turn_long_context"]
LIMIT = 20
STALL_MIN = 90          # 无新增 prediction 的超时阈值 (分钟)
MAX_ATTEMPTS = 2        # 1 次正常 + 1 次重试
HEARTBEAT_MIN = 5

# A1a 基线 acc (results/a1a reviews; iq2xxs review 已被清空, 用记录值)
BASE_ACC = {
    ("iq2xxs", "multi_turn_base"): 0.55,
    ("iq2xxs", "multi_turn_long_context"): 0.50,
    ("ud_q2kxl", "multi_turn_base"): 0.55,
    ("ud_q2kxl", "multi_turn_long_context"): 0.50,
    ("ud_q4km", "multi_turn_base"): 0.20,
    ("ud_q4km", "multi_turn_long_context"): 0.20,
    ("tcq_nl_global", "multi_turn_base"): 0.15,
    ("tcq_nl_global", "multi_turn_long_context"): 0.10,
    ("tcq_nl_math", "multi_turn_base"): 0.05,
    ("tcq_nl_math", "multi_turn_long_context"): 0.15,
    ("base2", "multi_turn_base"): 0.05,
    ("base2", "multi_turn_long_context"): 0.10,
}


def log(msg):
    # 链日志由启动方重定向到 results/a1h_chain.log
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


# ---------------------------------------------------------------- 端口 / 进程
def port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def llama_procs():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, errors="replace",
                         stdin=subprocess.DEVNULL).stdout or ""
    return [l for l in out.splitlines() if "llama-server.exe" in l.lower()]


def kill_llama_servers():
    subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True,
                   text=True, errors="replace", stdin=subprocess.DEVNULL)


def wait_ready(timeout_min=15):
    """等端口释放 + 无 llama-server 残留; 超时后强杀再等, 仍忙则返回 False."""
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        if not port_open(A.PORT) and not port_open(A.PROXY_PORT) and not llama_procs():
            return True
        time.sleep(10)
    log("WARN 端口/进程仍忙, 强杀 llama-server 后重试")
    kill_llama_servers()
    t1 = time.time()
    while time.time() - t1 < 300:
        if not port_open(A.PORT) and not port_open(A.PROXY_PORT) and not llama_procs():
            return True
        time.sleep(5)
    return False


# ---------------------------------------------------------------- 指标 (同 e_a1f_decomp 口径)
def review_path(arm, subset):
    return os.path.join(RES, "a1h", arm, subset, "reviews", arm, "bfcl_v3_%s.jsonl" % subset)


def pred_path(arm, subset):
    return os.path.join(RES, "a1h", arm, subset, "predictions", arm, "bfcl_v3_%s.jsonl" % subset)


def out_json(arm, subset):
    return os.path.join(RES, "a1h_%s_%s.json" % (arm, subset))


def run_log(arm, subset):
    return os.path.join(RES, "a1h_%s_%s.run.log" % (arm, subset))


CD_ERR = re.compile(r"cd:.*No such file")


def analyze(p):
    rows = []
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sc = ((r.get("sample_score") or {}).get("score") or {}).get("value") or {}
                msgs = r.get("messages") or []
                asst = [m for m in msgs if m.get("role") == "assistant"]
                tools = [m for m in msgs if m.get("role") == "tool"]
                calls, multi = [], 0
                for m in asst:
                    t = m.get("tool_calls") or []
                    if len(t) >= 2:
                        multi += 1
                    for tc in t:
                        fn = (tc or {}).get("function") or {}
                        calls.append((fn.get("name"),
                                      json.dumps(fn.get("arguments"), sort_keys=True, ensure_ascii=False)))
                rep = sum(1 for i in range(1, len(calls)) if calls[i] == calls[i - 1])
                err = sum(1 for t in tools if "error" in (t.get("content") or "").lower())
                cd_err = sum(1 for t in tools if CD_ERR.search(t.get("content") or ""))
                rows.append({"acc": float(sc.get("acc") or 0.0), "turns": len(asst),
                             "multi": multi, "repeat_pairs": rep, "tool_errors": err,
                             "cd_errors": cd_err,
                             "terminated": bool(asst) and not (asst[-1].get("tool_calls") or [])})
    n = len(rows)
    if not n:
        return {"n": 0, "source": p}
    return {
        "n": n, "source": p,
        "acc": round(sum(x["acc"] for x in rows) / n, 4),
        "turns_mean": round(sum(x["turns"] for x in rows) / n, 3),
        "multi_call_rate": round(sum(1 for x in rows if x["multi"]) / n, 3),
        "repeat_pairs_mean": round(sum(x["repeat_pairs"] for x in rows) / n, 3),
        "tool_errors_total": sum(x["tool_errors"] for x in rows),
        "cd_errors_total": sum(x["cd_errors"] for x in rows),
        "terminated_rate": round(sum(1 for x in rows if x["terminated"]) / n, 3),
    }


def baseline_acc(arm, subset):
    """优先读 A1a reviews 实算 acc, 缺失/空文件回落到 BASE_ACC 记录值."""
    c = sorted(glob.glob(os.path.join(RES, "a1a", arm, "reviews", arm,
                                      "bfcl_v3_%s.jsonl*" % subset)))
    for p in sorted(c, key=lambda x: ("rerun" not in os.path.basename(x), x)):
        try:
            st = analyze(p)
        except Exception:  # noqa: BLE001
            continue
        if st.get("n"):
            return st["acc"], p
    return BASE_ACC[(arm, subset)], "BASE_ACC"


# ---------------------------------------------------------------- 单作业
def job_record(arm, subset, status, error, attempts, minutes):
    st = analyze(review_path(arm, subset))
    base, src = baseline_acc(arm, subset)
    rec = {"arm": arm, "subset": subset, "status": status, "error": error,
           "attempts": attempts, "minutes": minutes,
           "acc": st.get("acc"), "turns_mean": st.get("turns_mean"),
           "multi_call_rate": st.get("multi_call_rate"),
           "repeat_pairs_mean": st.get("repeat_pairs_mean"),
           "tool_errors_total": st.get("tool_errors_total"),
           "cd_errors_total": st.get("cd_errors_total"),
           "terminated_rate": st.get("terminated_rate"), "n": st.get("n"),
           "baseline_acc": base, "baseline_source": src,
           "delta_acc_vs_a1a": (round(st["acc"] - base, 3) if st.get("acc") is not None else None)}
    return rec


def run_job(arm, subset):
    log("=== JOB %s / %s ===" % (arm, subset))
    t0 = time.time()
    logf = run_log(arm, subset)
    pred = pred_path(arm, subset)
    rc, err, status = None, None, "error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not wait_ready():
            err = "端口未释放/llama-server 残留"
            log("ERROR %s/%s 第 %d 次: %s" % (arm, subset, attempt, err))
            continue
        cmd = [PY, RUNNER, "--mode", "cwd_hint", "--arm", arm,
               "--subset", subset, "--limit", str(LIMIT)]
        env = dict(os.environ)
        env["A1G_MODE"] = "cwd_hint"
        env["PYTHONIOENCODING"] = "utf-8"
        log("$ " + " ".join(cmd) + "  (attempt %d)" % attempt)
        with open(logf, "a", encoding="utf-8") as fh:
            fh.write("\n===== attempt %d @ %s =====\n" % (attempt, time.strftime("%Y-%m-%d %H:%M:%S")))
            fh.flush()
            p = subprocess.Popen(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                                 env=env, stdin=subprocess.DEVNULL)
            last_sz, last_change, last_hb = -1, time.time(), time.time()
            while True:
                rc = p.poll()
                if rc is not None:
                    break
                try:
                    sz = os.path.getsize(pred) if os.path.exists(pred) else 0
                except OSError:
                    sz = 0
                if sz != last_sz:
                    last_sz, last_change = sz, time.time()
                now = time.time()
                if now - last_hb >= HEARTBEAT_MIN * 60:
                    log("  %s/%s pred=%.2f MB elapsed=%.0f min idle=%.0f min" % (
                        arm, subset, sz / 1048576.0, (now - t0) / 60.0, (now - last_change) / 60.0))
                    last_hb = now
                if now - last_change > STALL_MIN * 60:
                    log("TIMEOUT %s/%s: %d min 无新增 prediction, kill 进程树" % (
                        arm, subset, STALL_MIN))
                    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True,
                                   text=True, errors="replace", stdin=subprocess.DEVNULL)
                    try:
                        p.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        pass
                    rc, err, status = None, "stalled %d min" % STALL_MIN, "timeout"
                    break
                time.sleep(20)
        if status != "timeout":
            st = analyze(review_path(arm, subset))
            if rc == 0 and st.get("n"):
                status, err = "ok", None
            elif rc != 0:
                status, err = "error", "exit code %s" % rc
            else:
                status, err = "error", "no review rows"
        if status == "ok":
            break
        log("FAIL %s/%s attempt %d: %s" % (arm, subset, attempt, err))
        if attempt < MAX_ATTEMPTS:
            wait_ready()
    minutes = round((time.time() - t0) / 60.0, 1)
    rec = job_record(arm, subset, status, err, attempt, minutes)
    log("=== DONE %s/%s status=%s acc=%s (A1a %s, delta %s) %.1f min ===" % (
        arm, subset, status, rec["acc"], rec["baseline_acc"], rec["delta_acc_vs_a1a"], minutes))
    return rec


# ---------------------------------------------------------------- 汇总
def write_summary(jobs):
    out = {"experiment": "E-A1h", "mode": "cwd_hint", "limit": LIMIT,
           "seed": 20261111, "stall_min": STALL_MIN,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "jobs": jobs,
           "by_arm": {}}
    for j in jobs:
        out["by_arm"].setdefault(j["arm"], {})[j["subset"]] = {
            k: j.get(k) for k in ("status", "acc", "turns_mean", "multi_call_rate",
                                  "repeat_pairs_mean", "tool_errors_total", "cd_errors_total",
                                  "terminated_rate", "minutes", "baseline_acc",
                                  "delta_acc_vs_a1a")}
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("summary -> %s (%d/%d 作业)" % (SUMMARY, len(jobs), len(ARMS) * len(SUBSETS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", default="", help="逗号分隔的 arm:subset, 强制重跑")
    args = ap.parse_args()
    redo = set(x.strip() for x in args.redo.split(",") if x.strip())

    jobs = []
    t_all = time.time()
    log("A1h chain start: %d 作业, stall=%d min, retry=%d" % (
        len(ARMS) * len(SUBSETS), STALL_MIN, MAX_ATTEMPTS - 1))
    for subset in SUBSETS:
        for arm in ARMS:
            key = "%s:%s" % (arm, subset)
            rec = None
            if key not in redo and os.path.exists(out_json(arm, subset)):
                try:
                    prev = json.load(open(out_json(arm, subset), encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    prev = None
                if prev and prev.get("status") == "ok" and analyze(review_path(arm, subset)).get("n"):
                    rec = job_record(arm, subset, "ok", None, 0, prev.get("minutes"))
                    log("SKIP %s (已完成, acc=%s)" % (key, rec["acc"]))
            if rec is None:
                rec = run_job(arm, subset)
            jobs.append(rec)
            write_summary(jobs)

    log("A1h chain all done in %.1f min" % ((time.time() - t_all) / 60.0))
    return 0 if all(j["status"] == "ok" for j in jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
