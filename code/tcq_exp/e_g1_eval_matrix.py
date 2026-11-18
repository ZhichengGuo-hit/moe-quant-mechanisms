"""E-G1 评测矩阵: 5 配置 × 3 任务, 按块交错 (同一块内轮换配置, 降低机器漂移影响)。

每轮每个配置只推进 --block 道题, 于是同一任务上各配置的题目在时间上均匀交错;
e_g1_eval.run 自带 (tag, task_id) 断点续跑, 重复调用不会重做。

    python e_g1_eval_matrix.py --tasks gsm8k,humaneval,gpqa --block 50
    python e_g1_eval_matrix.py --configs B_ud_q4kxl,C_roleq4 --tasks gsm8k --limit 20
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PY = sys.executable
EVAL = os.path.join(HERE, "e_g1_eval.py")
G1 = os.path.join(r"E:\CUDA11.8\model\gemma4", "g1")
M = r"E:\CUDA11.8\model\gemma4"

CONFIGS = {
    # A = 官方 QAT; B = 社区 UD-Q4_K_XL; C/C'/D/R = 本实验构建
    # C: 专家 Q4_K + 非专家 Q5_K (15.6 GiB)
    # C': 专家 Q4_K + 非专家 Q6_K (非专家比特梯度, 与 C 仅差非专家档位)
    "A_qat": os.path.join(M, "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"),
    "B_ud_q4kxl": os.path.join(M, "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"),
    "C_roleq4": os.path.join(G1, "g1_c.g4q4k.gguf"),
    "Cprime_roleq4_nx6": os.path.join(G1, "g1_cp.g4q4k_nx6.gguf"),
    "D_roleq4_im": os.path.join(G1, "g1_d.g4q4k_im.gguf"),
    "R_roleq3_im": os.path.join(G1, "g1_r.g4q3k_im.gguf"),
}


def say(m):
    print("%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), m), flush=True)


def out_path(tag, task):
    return os.path.join(RES, f"e_g1_gemma4_{tag}_{task}.jsonl")


def n_records(tag, task):
    p = out_path(tag, task)
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_one(tag, model, task, args):
    before = n_records(tag, task)
    cmd = [PY, EVAL, "run", "--tag", tag, "--model", model, "--task", task,
           "--port", str(args.port), "--threads", str(args.threads), "--ctx", str(args.ctx),
           "--cpus", args.cpus, "--n-predict", str(args.n_predict), "--block", args.block_tag]
    if args.bin_dir:
        cmd += ["--bin-dir", args.bin_dir]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    # 每轮只推进 block 题, 否则单个配置会一口气跑完全部题, 交错失效
    cmd += ["--max-new", str(args.block)]
    log = os.path.join(RES, "e_g1_matrix.log")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n=== %s %s\n" % (time.strftime("%H:%M:%S"), " ".join(cmd)))
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    after = n_records(tag, task)
    say("[%s/%s] +%d 题 (共 %d) rc=%d %.1f min"
        % (tag, task, after - before, after, p.returncode, (time.time() - t0) / 60))
    return after - before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--tasks", default="gsm8k,humaneval,gpqa")
    ap.add_argument("--block", type=int, default=50, help="每轮每配置推进的题数")
    ap.add_argument("--port", type=int, default=18101)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--cpus", default="0-15")
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="每任务只用前 N 题 (冒烟)")
    ap.add_argument("--block-tag", default="g1")
    ap.add_argument("--bin-dir", default="", help="llama.cpp 可执行目录")
    args = ap.parse_args()

    tags = [t.strip() for t in args.configs.split(",") if t.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tags:
        if t not in CONFIGS:
            raise SystemExit(f"未知配置 {t}, 可选: {list(CONFIGS)}")
    missing = [t for t in tags if not os.path.exists(CONFIGS[t])]
    if missing:
        raise SystemExit("缺模型文件: " + "; ".join(f"{t}={CONFIGS[t]}" for t in missing))
    total = {(t, k): 200 if k == "gsm8k" else 164 if k == "humaneval" else 100
             for k in tasks for t in tags}
    if args.limit:
        total = {k: min(v, args.limit) for k, v in total.items()}

    say("[matrix] 配置 %s | 任务 %s | block=%d" % (tags, tasks, args.block))
    for task in tasks:
        while True:
            progressed = 0
            for tag in tags:
                if n_records(tag, task) >= total[(tag, task)]:
                    continue
                progressed += run_one(tag, CONFIGS[tag], task, args)
            if not progressed:
                break
    say("[matrix] 全部配置/任务完成")
    for tag in tags:
        for task in tasks:
            say("  %-13s %-10s %d 条" % (tag, task, n_records(tag, task)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
