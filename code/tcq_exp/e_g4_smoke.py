"""G4 Clover (dLLM) 冒烟: 等 G3 释放 GPU 后, 用 llama-diffusion-cli 跑 1 道 GSM8K。

判据: 模型能加载 (clover_moe) -> 拿到 mask token -> 扩散解码出可读文本 -> 答案正确。
失败则实验降级为 0(c) 方案, 结论写入 results/e_g4_smoke.json。

    python e_g4_smoke.py --model B --wait-gpu
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e17_task_eval as E17  # noqa: E402  判分口径与自回归评测保持一致 (#### 优先)
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin"
CLI = os.path.join(BIN, "llama-diffusion-cli.exe")
MODELS = {
    "A": r"E:\CUDA11.8\model\clover\Clover-35B-A3B-Preview-Q4_K_M.gguf",
    "B": r"E:\CUDA11.8\model\clover\Clover-35B-A3B-Preview-AD-Q4_K-IQ4_XS.gguf",
}
SYSTEM_MATH = ("You are a helpful assistant. Solve the math problem step by step "
               "and put the final numeric answer after '####'.")
LOG = os.path.join(HERE, "e_g4_smoke.log")


def say(m):
    line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def proc_alive(name):
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process | Where-Object {$_.Name -eq '%s'} | "
                          "Measure-Object | Select -Expand Count" % name],
                         capture_output=True, text=True)
    try:
        return int((out.stdout or "0").strip())
    except ValueError:
        return 0


def gpu_busy():
    return proc_alive("llama-server.exe") > 0 or proc_alive("llama-diffusion-cli.exe") > 0


def wait_gpu():
    """G3 评测结束标记优先; 若 G3 异常退出(无 server 进程), 连续 5 次确认后放行。"""
    mark = os.path.join(RES, ".g3_eval_done")
    calm = 0
    while True:
        if os.path.exists(mark):
            say("[wait] .g3_eval_done 已出现")
            break
        if gpu_busy():
            calm = 0
        else:
            calm += 1
            if calm >= 5:
                say("[wait] GPU 连续 10 分钟空闲, 视为 G3 已结束")
                break
        time.sleep(120)
    while gpu_busy():
        say("[wait] GPU 仍被占用, 继续等待")
        time.sleep(120)
    return True


def model_ready(role):
    p = MODELS[role]
    if not os.path.exists(p):
        return False
    return os.path.exists(p + ".done")


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_out(raw):
    """diffusion-cli 用 \\r 刷新进度条, 最后清屏后打印正文; 先去掉 ANSI 转义再切。"""
    raw = ANSI.sub("", raw).replace("\r", "")
    idx = raw.rfind("diffusion step")
    if idx < 0:
        return raw.strip()
    seg = raw[idx:]
    nl = seg.find("\n")
    text = seg[nl + 1:] if nl >= 0 else ""
    return text.strip()


def extract_number(text):
    return E17.extract_number(text)


def grade(text, gold):
    pred = extract_number(text)
    try:
        return pred is not None and abs(pred - float(gold)) < 1e-4
    except ValueError:
        return False


def run_one(role, row, ngl, steps, block, ub):
    cmd = [CLI, "-m", MODELS[role], "-ngl", str(ngl), "-t", "16",
           "-c", "4096", "-b", str(ub), "-ub", str(ub),
           "-p", row["question"], "-sys", SYSTEM_MATH,
           "--diffusion-block-length", str(block), "--diffusion-steps", str(steps),
           "--diffusion-algorithm", "4", "--temp", "0", "--seed", "1234",
           "--no-perf"]
    say("[run] ngl=%d steps=%d block=%d ub=%d" % (ngl, steps, block, ub))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=2400, cwd=BIN)
    except subprocess.TimeoutExpired as e:
        return {"rc": -9, "err": "timeout", "raw": (e.stderr or "")[-4000:], "elapsed": time.time() - t0}
    raw = (p.stdout or "") + "\n" + (p.stderr or "")
    return {"rc": p.returncode, "raw": raw, "elapsed": time.time() - t0}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="B", choices=["A", "B"])
    ap.add_argument("--wait-gpu", action="store_true")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--ub", type=int, default=512)
    args = ap.parse_args()
    say("[start] model=%s wait_gpu=%s" % (args.model, args.wait_gpu))

    if args.wait_gpu:
        wait_gpu()
    role = args.model
    if not model_ready(role):
        other = "A" if role == "B" else "B"
        say("[skip] %s 档未就绪, 尝试 %s" % (role, other))
        if not model_ready(other):
            res = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "ok": False,
                   "reason": "无已下载完成的权重"}
            json.dump(res, open(os.path.join(RES, "e_g4_smoke.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            return 1
        role = other

    row = json.loads(open(os.path.join(DATA, "gsm8k_200.jsonl"), encoding="utf-8").readline())
    say("[smoke] model=%s q=%s gold=%s" % (role, row["task_id"], row["gold"]))

    result = None
    for ngl in (28, 22, 16):
        r = run_one(role, row, ngl, args.steps, args.block, args.ub)
        raw = r["raw"]
        text = parse_out(raw)
        loaded = r["rc"] == 0 and "mask_token_id" in raw
        res = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": role, "path": MODELS[role], "ngl": ngl,
            "block": args.block, "steps": args.steps, "ub": args.ub,
            "rc": r["rc"], "elapsed_s": round(r["elapsed"], 1),
            "loaded_clover_moe": loaded,
            "mask_token_found": "mask_token_id" in raw,
            "gsm8k_id": row["task_id"], "gold": row["gold"],
            "output": text[:2000],
            "readable": bool(text) and len(text) >= 20,
            "correct": grade(text, row["gold"]),
        }
        if r["rc"] != 0:
            res["error_tail"] = raw[-1200:]
        result = res
        if loaded and res["readable"]:
            break
        say("[retry] ngl=%d 失败 rc=%d" % (ngl, r["rc"]))
    result["ok"] = bool(result["loaded_clover_moe"] and result["readable"] and result["correct"])
    if not result["ok"]:
        result["degrade"] = "0(c): 只做生态观察+打包产物, 不做质量对比"
    json.dump(result, open(os.path.join(RES, "e_g4_smoke.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    say("[result] ok=%s correct=%s" % (result["ok"], result.get("correct")))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
