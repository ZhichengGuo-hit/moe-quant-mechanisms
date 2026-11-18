#!/usr/bin/env python
"""E-H 系列串行队列（GPU 16 GiB 单卡，全部串行）

stages:
  build   H1 十配置量化（跳过已存在）
  nll     H1 NLL eval64 三域 + bootstrap 判定
  tasks   H1 任务级（G0/G1/G3/G4 × im-on × GSM8K200/HE164）
  h3d     Xing D′（通用语料 imatrix + C 类型 + GPQA100）
  h4      AIME 诊断统计（零机时）

用法: python e_h_queue.py --stages build,nll,tasks,h3d,h4
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin"
MODEL_DIR = r"E:\CUDA11.8\model\e20\h1"
RES = os.path.join(HERE, "results")
STATE = os.path.join(RES, "e_h_queue_state.json")
TASK_ARMS = ["g0", "g1", "g3", "g4"]
TASKS = ["gsm8k", "humaneval"]


def log(msg):
    print(msg, flush=True)


def sh(cmd, env=None, logfile=None):
    log("$ " + " ".join(cmd))
    e = dict(os.environ)
    if env:
        e.update(env)
    if logfile:
        with open(logfile, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=e)
    else:
        p = subprocess.run(cmd, env=e)
    return p.returncode


def stage_build(st):
    rc = sh([PY, os.path.join(HERE, "e_h1_build.py")])
    st["build"] = {"rc": rc}
    return rc == 0


def stage_wait(st):
    """等待独立启动的 build 进程完成（10 个配置全部 ok）。"""
    import time as _t
    t0 = _t.time()
    while _t.time() - t0 < 6 * 3600:
        bs = os.path.join(RES, "e_h1_build_state.json")
        if os.path.exists(bs):
            b = json.load(open(bs))
            done = [k for k, v in b.items() if v.get("status") == "ok"]
            if len(done) >= 10:
                st["wait"] = {"n_ok": len(done), "ok": True}
                return True
        _t.sleep(120)
    st["wait"] = {"ok": False}
    return False


def stage_nll(st):
    rc1 = sh([PY, os.path.join(HERE, "e_h1_nll.py"), "run"])
    rc2 = sh([PY, os.path.join(HERE, "e_h1_nll.py"), "report"])
    st["nll"] = {"run_rc": rc1, "report_rc": rc2}
    return rc1 == 0 and rc2 == 0


def stage_tasks(st):
    env = {"E_G_PREFIX": "e_h1_", "E_G_BASELINE": "g1"}
    ok = True
    port = 18201
    for arm in TASK_ARMS:
        tag = "%s_im" % arm
        model = os.path.join(MODEL_DIR, "e_h1_%s_imon.gguf" % arm)
        if not os.path.exists(model):
            log("[skip] missing %s" % model)
            ok = False
            continue
        for task in TASKS:
            lg = os.path.join(RES, "h1_logs", "task_%s_%s.log" % (tag, task))
            rc = sh([PY, os.path.join(HERE, "e_g1_eval.py"), "run",
                     "--tag", tag, "--model", model, "--task", task,
                     "--port", str(port), "--bin-dir", BIN, "--block", "h1",
                     "--ctx", "4096", "--n-predict", "1024"], env=env, logfile=lg)
            port += 1
            ok = ok and rc == 0
    rc = sh([PY, os.path.join(HERE, "e_g1_eval.py"), "score",
             "--baseline", "g1_im", "--bootstrap", "2000"], env=env)
    st["tasks"] = {"ok": ok, "score_rc": rc}
    return ok and rc == 0


def stage_h3d(st):
    rc = sh([PY, os.path.join(HERE, "e_h3_dprime.py")])
    st["h3d"] = {"rc": rc}
    return rc == 0


def stage_h5(st):
    """H5：长上下文（NIAH 4k/16k/64k + RULER 同构 8k），CUDA 三臂，需 GPU 独占。"""
    rc = sh([PY, os.path.join(HERE, "e_h5_eval.py"), "all",
             "--arms", "iq2xxs,ud_q2kxl,ud_q4km", "--port", "18301"])
    st["h5"] = {"rc": rc}
    return rc == 0


def stage_h4(st):
    rc = sh([PY, os.path.join(HERE, "e_h4_hard_stats.py")])
    st["h4"] = {"rc": rc}
    return rc == 0


STAGES = {"build": stage_build, "wait": stage_wait, "nll": stage_nll,
          "tasks": stage_tasks, "h3d": stage_h3d, "h5": stage_h5, "h4": stage_h4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="build,nll,tasks,h3d,h4")
    a = ap.parse_args()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    for name in a.stages.split(","):
        name = name.strip()
        if not name:
            continue
        log("=" * 60)
        log("[stage] %s" % name)
        ok = STAGES[name](st)
        st[name]["done_at"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
        st[name]["ok"] = ok
        json.dump(st, open(STATE, "w"), indent=1, ensure_ascii=False)
    log("QUEUE DONE: " + json.dumps({k: v.get("ok") for k, v in st.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
