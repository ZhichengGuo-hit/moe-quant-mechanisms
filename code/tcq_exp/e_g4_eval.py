"""G4 Clover (dLLM) A/B 正式评测: GSM8K 200 + MATH-500 100 + HumanEval 164。

与自回归评测的区别只在推理接口: llama-server 只能 AR(输出乱码), 扩散解码必须走
llama-diffusion-cli.exe, 每题一次进程调用。判分口径尽量复用 e17_task_eval。

    # 等冒烟通过 + GPU 空闲, 自动按冒烟实测耗时挑参数, 跑 A/B 并汇总
    python e_g4_eval.py --auto --wait

    # 手动指定
    python e_g4_eval.py --configs A_off_q4km,B_ad_iq4xs --steps 128 --ub 512
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e17_task_eval as E17  # noqa: E402
import e_g4_smoke as SMOKE  # noqa: E402  复用模型路径 / CLI / 输出解析
from e_g1_eval import check_humaneval  # noqa: E402  复用隔离单测判分
from pin import pin_self  # noqa: E402

RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
PREFIX = "e_g4_clover_"
BASELINE = "A_off_q4km"
CONFIGS = {
    "A_off_q4km": r"E:\CUDA11.8\model\clover\Clover-35B-A3B-Preview-Q4_K_M.gguf",
    "B_ad_iq4xs": r"E:\CUDA11.8\model\clover\Clover-35B-A3B-Preview-AD-Q4_K-IQ4_XS.gguf",
}
TOTALS = {"gsm8k": 200, "math500": 100, "humaneval": 164}
SYSTEM_MATH500 = ("You are a math problem solver. Think step by step, then output the final "
                  "answer on a new line as: #### <answer>")
BLOCK = 32
LOG = os.path.join(HERE, "e_g4_eval.log")


def say(m):
    line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def out_path(tag, task):
    return os.path.join(RES, f"{PREFIX}{tag}_{task}.jsonl")


def done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["task_id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


def load_task(task):
    if task == "gsm8k":
        rows = E17.load_tasks("gsm8k")
        for r in rows:
            r["_user"] = r["question"].strip()
            r["_system"] = E17.SYSTEM_MATH
    elif task == "humaneval":
        rows = E17.load_tasks("humaneval")
        for r in rows:
            r["_user"] = r["prompt"].strip()
            r["_system"] = E17.SYSTEM_CODE
    else:
        path = os.path.join(DATA, "math500_100.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for r in rows:
            r["_user"] = r["problem"].strip()
            r["_system"] = SYSTEM_MATH500
    return rows


NORM_DROP = re.compile(r"\\left|\\right|\\!|\\,|\\;|\\ |\\displaystyle|\\dfrac|\s+")


def norm_math(s):
    s = str(s).strip()
    s = s.replace("$$", "").replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = NORM_DROP.sub("", s)
    s = s.rstrip(".")
    return s.lower()


def as_num(s):
    try:
        return float(s)
    except ValueError:
        return None


def grade(task, row, text):
    if task == "gsm8k":
        pred = E17.extract_number(text)
        return pred is not None and abs(pred - float(row["gold"])) < 1e-4
    if task == "humaneval":
        return check_humaneval(row, text)[0]
    tail = text.split("####")[-1] if "####" in text else text
    cand = tail.strip().splitlines()[-1] if tail.strip() else ""
    g, c = norm_math(row["gold"]), norm_math(cand)
    if g == c:
        return True
    ng, nc = as_num(g), as_num(c)
    return ng is not None and nc is not None and abs(ng - nc) < 1e-4


def call(role, row, ngl, steps, ub, timeout):
    cmd = [SMOKE.CLI, "-m", CONFIGS[role], "-ngl", str(ngl), "-t", "16",
           "-c", "4096", "-b", str(ub), "-ub", str(ub),
           "-p", row["_user"], "-sys", row["_system"],
           "--diffusion-block-length", str(BLOCK), "--diffusion-steps", str(steps),
           "--diffusion-algorithm", "4", "--temp", "0", "--seed", "1234", "--no-perf"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, cwd=SMOKE.BIN)
    except subprocess.TimeoutExpired as e:
        return None, "timeout", time.time() - t0, (e.stderr or "")
    return p.returncode, (p.stdout or "") + "\n" + (p.stderr or ""), time.time() - t0, ""


def alloc_fail(raw):
    low = raw.lower()
    return ("failed to allocate" in low or "out of memory" in low
            or "cuda error" in low or "ggml_backend_cuda" in low and "error" in low)


def run_task(role, task, ngl, steps, ub, timeout, block_cap):
    path = out_path(role, task)
    rows = load_task(task)
    done = done_ids(path)
    todo = [r for r in rows if r["task_id"] not in done]
    if block_cap:
        todo = todo[:block_cap]
    todo.sort(key=lambda r: r["task_id"])
    say("[%s/%s] 共 %d, 已完成 %d, 本轮 %d | ngl=%d steps=%d ub=%d"
        % (role, task, len(rows), len(done), len(todo), ngl, steps, ub))
    if not todo:
        return ngl
    t0, n_ok = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo):
            rc, raw, el, _ = call(role, row, ngl, steps, ub, timeout)
            if rc is None:
                say("  [skip] %s timeout" % row["task_id"])
                continue
            if rc != 0 and alloc_fail(raw):
                ngl = max(8, ngl - 4)
                say("  [oom] ngl 降到 %d 重试" % ngl)
                rc, raw, el, _ = call(role, row, ngl, steps, ub, timeout)
            if rc != 0:
                say("  [skip] %s rc=%d %s" % (row["task_id"], rc, raw.strip()[-160:]))
                continue
            text = SMOKE.parse_out(raw)
            ok = grade(task, row, text)
            rec = {"task_id": row["task_id"], "tag": role, "task": task,
                   "passed": bool(ok), "gen_ms": round(el * 1000),
                   "finish_reason": "length", "text": text[:4000],
                   "block": BLOCK, "steps": steps, "ub": ub, "ngl": ngl}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_ok += int(ok)
            if i < 2 or (i + 1) % 10 == 0 or i + 1 == len(todo):
                say("  %d/%d %s pass=%s acc=%.3f %.0fs elapsed=%.0fm"
                    % (i + 1, len(todo), row["task_id"], ok, n_ok / (i + 1), el,
                       (time.time() - t0) / 60))
    return ngl


def wait_ready():
    smoke = os.path.join(RES, "e_g4_smoke.json")
    while not os.path.exists(smoke):
        say("[wait] 等 e_g4_smoke.json")
        time.sleep(120)
    res = json.load(open(smoke, encoding="utf-8"))
    if not res.get("ok"):
        say("[stop] 冒烟未通过, 按降级方案 0(c) 处理, 不做质量对比")
        return None
    calm = 0
    while calm < 5:
        if SMOKE.gpu_busy():
            calm = 0
        else:
            calm += 1
        time.sleep(120)
    say("[wait] GPU 已空闲")
    return res


def auto_params(smoke):
    """按冒烟实测耗时挑 (ub, steps): steps 必须是 ub/32 的整数倍, 且不低于 64 步。"""
    el = float(smoke.get("elapsed_s") or 300)
    base = 512 * 128  # 冒烟配置 (ub=512, steps=128) 的相对开销
    opts = [(1024, 128), (512, 128), (512, 64)]
    for ub, steps in opts:
        if el * (ub * steps) / base <= 80:
            return ub, steps
    return 512, 64


def finish(steps, ub, ngsl):
    env = os.environ.copy()
    env["E_G_PREFIX"] = PREFIX
    cmd = [sys.executable, os.path.join(HERE, "e_g1_eval.py"), "score",
           "--baseline", BASELINE, "--out", "e_g4_clover_summary.json"]
    with open(os.path.join(RES, "e_g4_score.log"), "a", encoding="utf-8") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    proto = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decoder": "llama-diffusion-cli.exe (dLLM 扩散解码), 非 llama-server",
        "arch": "clover_moe",
        "block_size": BLOCK, "diffusion_steps": steps, "max_length_ub": ub,
        "algorithm": "4 (confidence-based)", "temperature": 0.0, "seed": 1234,
        "ngl": ngsl,
        "configs": {"A_off_q4km": "官方 Clover-35B-A3B-Preview-Q4_K_M.gguf (hf-mirror)",
                    "B_ad_iq4xs": "AtomicChat Clover-AD-Q4_K-IQ4_XS.gguf (ModelScope)"},
        "tasks": TOTALS,
        "capped_口径": {
            "generated_tokens": "ub - prompt_tokens (diffusion max_length = ubatch, 含 prompt)",
            "note": "自回归档为 n-predict=1024 + ctx 4096; 扩散档总长 ub, 生成上限受 prompt 长度挤压",
            "finish_reason": "固定记 length, 扩散无 EOS 提前停止",
        },
        "抽取逻辑": {
            "gsm8k": "e17.extract_number: 取最后一个 #### 之后的数字",
            "humaneval": "e17.extract_code + 隔离子进程跑单测 (8s 超时)",
            "math500": "取 #### 后最后一行, 归一化(去 $ \\left \\right \\frac 变体/空白/小写)后精确比对, 双数值再容差 1e-4; "
                        "\\frac{3}{4} 与 0.75 不互认, 属该口径的已知下限",
        },
        "与自回归的一致性差异": [
            "推理接口: 每题独立进程, 无 KV 复用, 无 server 批处理",
            "chat template: 由 GGUF 自带模板拼接 (-p/-sys), 与 server /v1/chat/completions 同源",
            "生成长度: 扩散受 ub 限制且随 prompt 变短而变长, 与 AR 固定 1024 不同",
            "耗时: 扩散每步跑满 ub 长度, 与生成长度无关, 单题耗时远高于 AR",
        ],
    }
    json.dump(proto, open(os.path.join(RES, "e_g4_protocol.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    say("[finish] 汇总 + 协议差异已写入 results/e_g4_protocol.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="按冒烟实测耗时自动挑参数")
    ap.add_argument("--wait", action="store_true", help="等冒烟通过且 GPU 空闲")
    ap.add_argument("--configs", default="A_off_q4km,B_ad_iq4xs")
    ap.add_argument("--tasks", default="gsm8k,math500,humaneval")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--ngl", type=int, default=22)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cpus", default="0-31")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--block-cap", type=int, default=0)
    args = ap.parse_args()

    roles = [r.strip() for r in args.configs.split(",") if r.strip()]
    for r in roles:
        if r not in CONFIGS:
            raise SystemExit("未知配置 %s" % r)
        for suf in ("", ".done"):
            p = CONFIGS[r] + suf
            if suf and not os.path.exists(p):
                raise SystemExit("权重未下完: %s" % p)

    steps, ub = args.steps, args.ub
    if args.wait or args.auto:
        smoke = wait_ready()
        if smoke is None:
            return 2
        if args.auto:
            ub, steps = auto_params(smoke)
            args.ngl = min(args.ngl, int(smoke.get("ngl") or args.ngl))
            say("[auto] 冒烟 %.0fs -> ub=%d steps=%d ngl=%d"
                % (float(smoke.get("elapsed_s") or 0), ub, steps, args.ngl))

    pin_self(cpus=args.cpus)
    say("[start] roles=%s tasks=%s steps=%d ub=%d block=%d ngl=%d"
        % (roles, args.tasks, steps, ub, BLOCK, args.ngl))
    ngsl = {}
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        for role in roles:
            n = run_task(role, task, ngsl.get(role, args.ngl), steps, ub,
                         args.timeout, args.block_cap)
            ngsl[role] = n
            if abs(n - args.ngl) >= 4:
                say("[warn] %s 实际 ngl=%d 与设定不一致" % (role, n))
    finish(steps, ub, ngsl)
    say("[done] G4 A/B 评测完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
