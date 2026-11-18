"""E-G2/G3/G5/G6/G7 评测矩阵 (与 G1 完全同协议): 配置 × 3 任务, 块交错 50 题。

复用 e_g1_eval.py 的 run/score, 只通过环境变量切换结果前缀与二进制目录:
  E_G_PREFIX = e_g2_minicpm_ / e_g3_xing_ / e_g5_glm47_ / e_g6_nemotron_ / e_g7_raon_
  E_G_BIN    = llama-upstream/build-cuda/bin  (G2/G3 统一二进制, 含 xing4_0)

混合推理的 -ngl 按配置给固定值 (不靠 llama.cpp 按空闲显存自动分裂):
  16 GiB 卡上放不下的 18-21 GB 档位给 18-24 层, 剩余层留在 CPU;
  15 GiB 以下给 99 (全上卡)。同一配置跨批次 -ngl 不变, 保证速度可比。

    python e_g_eval_matrix.py --suite g2 --tasks gsm8k,humaneval,gpqa --block 50
    python e_g_eval_matrix.py --suite g3 --configs A_off_iq4xs,B1_venastine_q4km --limit 20
"""
import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
EVAL = os.path.join(HERE, "e_g1_eval.py")
BIN = os.path.join(ROOT, "llama-upstream", "build-cuda", "bin")

M2 = r"E:\CUDA11.8\model\minicpm"
G2D = os.path.join(M2, "g2")
M3 = r"E:\CUDA11.8\model\xing"
G3D = os.path.join(M3, "g3")
M5 = r"E:\CUDA11.8\model\glm47"
M6 = r"E:\CUDA11.8\model\nemotron"
M7 = r"E:\CUDA11.8\model\raon"
G5D = os.path.join(M5, "g5")
G6D = os.path.join(M6, "g6")
G7D = os.path.join(M7, "g7")

SUITES = {
    "g2": {
        "prefix": "e_g2_minicpm_",
        "baseline": "A_off_q4km",
        "port": 18111,
        "configs": {
            "A_off_q4km": (os.path.join(M2, "MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf"), 30),
            "Rofficial_mixq3im": (os.path.join(M2, "MiniCPM-V-4_7-35B-A3B-mix_q3gud_im.gguf"), 99),
            "C_roleq4": (os.path.join(G2D, "e_g2_c.gguf"), 36),
            "Clayer_route": (os.path.join(G2D, "e_g2_clayer.gguf"), 40),
            "D_roleq4_im": (os.path.join(G2D, "e_g2_d.gguf"), 36),
            "R_roleq3_im": (os.path.join(G2D, "e_g2_r.gguf"), 99),
        },
    },
    "g3": {
        "prefix": "e_g3_xing_",
        "baseline": "A_off_iq4xs",
        "port": 18112,
        "configs": {
            "A_off_iq4xs": (os.path.join(M3, "xing4_0-29b-iq4_xs.gguf"), 42),
            "B1_venastine_q4km": (os.path.join(M3, "Xing4.0-29B-A4B-Q4_K_M.gguf"), 42),
            "B2_dealign_q4km": (os.path.join(M3, "Xing4.0-29B-A4B-CRACK-Q4_K_M.gguf"), 42),
            "C_roleq4": (os.path.join(G3D, "e_g3_c.gguf"), 42),
            "D_roleq4_im": (os.path.join(G3D, "e_g3_d.gguf"), 42),
            "R_roleq3_im": (os.path.join(G3D, "e_g3_r.gguf"), 99),
        },
    },
    # G5 GLM-4.7-Flash 30B-A3B (arch=deepseek2): 两档均 < 13 GiB, 全上卡; C/D/R 由 BF16 构建
    "g5": {
        "prefix": "e_g5_glm47_",
        "baseline": "A_udq2kxl",
        "port": 18113,
        "configs": {
            "A_udq2kxl": (os.path.join(M5, "GLM-4.7-Flash-UD-Q2_K_XL.gguf"), 99),
            "B_udq3kxl": (os.path.join(M5, "GLM-4.7-Flash-UD-Q3_K_XL.gguf"), 99),
            "C_roleq4": (os.path.join(G5D, "e_g5_c.gguf"), None),
            "D_roleq4_im": (os.path.join(G5D, "e_g5_d.gguf"), None),
            "R_roleq3_im": (os.path.join(G5D, "e_g5_r.gguf"), None),
        },
    },
    # G6 Nemotron-Cascade-2 30B-A3B (arch=nemotron_h_moe): 最小档 16.7 GiB 已超显存, 部分卸载
    "g6": {
        "prefix": "e_g6_nemotron_",
        "baseline": "A_i1q2k",
        "port": 18114,
        "configs": {
            "A_i1q2k": (os.path.join(M6, "Nemotron-Cascade-2-30B-A3B.i1-Q2_K.gguf"), 20),
            "B_i1q3km": (os.path.join(M6, "Nemotron-Cascade-2-30B-A3B.i1-Q3_K_M.gguf"), 16),
            "C_roleq4": (os.path.join(G6D, "e_g6_c.gguf"), None),
            "D_roleq4_im": (os.path.join(G6D, "e_g6_d.gguf"), None),
            "R_roleq3_im": (os.path.join(G6D, "e_g6_r.gguf"), None),
        },
    },
    # G7 Raon-1-mini-preview (arch=raon1): 只有 KRAFTON fork 认该 arch, 14.97 GiB 全上卡
    "g7": {
        "prefix": "e_g7_raon_",
        "baseline": "A_iq4xs",
        "port": 18115,
        "configs": {
            "A_iq4xs": (os.path.join(M7, "raon-1-mini-preview-IQ4_XS.gguf"), 99),
            "C_roleq4": (os.path.join(G7D, "e_g7_c.gguf"), None),
            "D_roleq4_im": (os.path.join(G7D, "e_g7_d.gguf"), None),
            "R_roleq3_im": (os.path.join(G7D, "e_g7_r.gguf"), None),
        },
    },
}


def n_layers(suite):
    p = os.path.join(RES, "e_%s_tensors.txt" % suite)
    if os.path.exists(p):
        mx = -1
        with open(p, encoding="utf-8") as f:
            for line in f:
                m = re.search(r"blk\.(\d+)\.", line)
                if m:
                    mx = max(mx, int(m.group(1)))
        if mx >= 0:
            return mx + 1
    return {"g2": 40, "g3": 64, "g5": 47, "g6": 52, "g7": 25}.get(suite, 40)


def auto_ngl(path, layers):
    """16 GiB 卡可用约 15 GiB: 放得下就全上卡, 放不下按体积等比卸载。"""
    if not os.path.exists(path):
        return 99
    gib = os.path.getsize(path) / 2**30
    if gib <= 15.0:
        return 99
    return max(8, int(layers * 15.0 / gib))


def say(m):
    print("%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), m), flush=True)


def n_records(prefix, tag, task):
    p = os.path.join(RES, f"{prefix}{tag}_{task}.jsonl")
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_one(prefix, tag, model, ngl, task, args):
    before = n_records(prefix, tag, task)
    cmd = [PY, EVAL, "run", "--tag", tag, "--model", model, "--task", task,
           "--port", str(args.port), "--threads", str(args.threads), "--ctx", str(args.ctx),
           "--cpus", args.cpus, "--n-predict", str(args.n_predict),
           "--ngl", str(ngl), "--max-new", str(args.block), "--block", args.block_tag or args.suite]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    env = os.environ.copy()
    env["E_G_PREFIX"] = prefix
    env["E_G_BIN"] = args.bin_dir or BIN
    env["E_G_BASELINE"] = args.baseline
    log = os.path.join(RES, f"{prefix}matrix.log")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n=== %s %s\n" % (time.strftime("%H:%M:%S"), " ".join(cmd)))
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    after = n_records(prefix, tag, task) - (0 if args.limit == 0 else 0)
    say("[%s/%s] +%d 题 (共 %d) ngl=%d rc=%d %.1f min"
        % (tag, task, after - before, after, ngl, p.returncode, (time.time() - t0) / 60))
    return after - before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["g2", "g3", "g5", "g6", "g7"])
    ap.add_argument("--configs", default="")
    ap.add_argument("--tasks", default="gsm8k,humaneval,gpqa")
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--block-tag", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--cpus", default="0-31")
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bin-dir", default="")
    args = ap.parse_args()

    s = SUITES[args.suite]
    args.port = args.port or s["port"]
    args.baseline = s["baseline"]
    tags = [t.strip() for t in args.configs.split(",") if t.strip()] or list(s["configs"])
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tags:
        if t not in s["configs"]:
            raise SystemExit("未知配置 %s, 可选 %s" % (t, list(s["configs"])))
    missing = [(t, s["configs"][t][0]) for t in tags
               if not os.path.exists(s["configs"][t][0])]
    if missing:
        raise SystemExit("缺模型文件: " + "; ".join("%s=%s" % x for x in missing))
    # ngl=None 的臂 (TCQ C/D/R) 按产物体积现场定卸载层数
    layers = n_layers(args.suite)
    for t in tags:
        model, ngl = s["configs"][t]
        if ngl is None:
            ngl = auto_ngl(model, layers)
            s["configs"][t] = (model, ngl)
            say("[ngl] %s %.2f GiB / %d 层 -> ngl=%d"
                % (t, os.path.getsize(model) / 2**30, layers, ngl))

    total = {(t, k): 200 if k == "gsm8k" else 164 if k == "humaneval" else 100
             for k in tasks for t in tags}
    if args.limit:
        total = {k: min(v, args.limit) for k, v in total.items()}

    say("[matrix] suite=%s 配置=%s | 任务=%s | block=%d | bin=%s"
        % (args.suite, tags, tasks, args.block, args.bin_dir or BIN))
    for task in tasks:
        while True:
            progressed = 0
            for tag in tags:
                if n_records(s["prefix"], tag, task) >= total[(tag, task)]:
                    continue
                model, ngl = s["configs"][tag]
                progressed += run_one(s["prefix"], tag, model, ngl, task, args)
            if not progressed:
                break
    say("[matrix] 完成")
    for tag in tags:
        for task in tasks:
            say("  %-20s %-10s %d 条" % (tag, task, n_records(s["prefix"], tag, task)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
