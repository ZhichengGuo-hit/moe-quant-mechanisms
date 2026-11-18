"""C1: 用一个臂跑 GSM8K200 + HumanEval164 (协议 = e17 / e_g1_eval, 与 E-H1c 完全相同)。

串行: 每个任务单独起一个 llama-server (同一时间只有一个 server), 断点续跑由
e_g1_eval.run 按 (tag, task_id) 去重; 结束后把两个任务的逐题文件合并成单文件 jsonl
(字段与 e17 输出同风格, 至少含 task/task_id/acc)。

    python c1_run_tasks.py --model E:\\CUDA11.8\\model\\e20\\h1\\e_h1_g2_imon.gguf \\
        --tag g2_im --prefix c1_ --out results/c1_g2_tasks.jsonl \\
        --log results/c1_g2.run.log --ports 18221,18222

环境: E_G_BIN / E_G_PREFIX 与 e_g1_eval 一致; bin 目录默认上游 CUDA 构建 (E-H1c 同款)。
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CUDA_BIN = os.path.abspath(os.path.join(HERE, "..", "llama-upstream", "build-cuda", "bin"))


def say(fh, msg):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def aggregate(prefix, tag, out_path, tasks):
    rows = []
    for task in tasks:
        src = os.path.join(RES, "%s%s_%s.jsonl" % (prefix, tag, task))
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append({
                    "task": task,
                    "task_id": r["task_id"],
                    "acc": bool(r.get("passed")),
                    "passed": bool(r.get("passed")),
                    "tag": r.get("tag", tag),
                    "domain": None,
                    "predicted_n": r.get("predicted_n"),
                    "gen_ms": r.get("gen_ms"),
                    "finish_reason": r.get("finish_reason"),
                    "completion": r.get("text", ""),
                    "reasoning": r.get("reasoning", ""),
                    "source": os.path.basename(src),
                })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = {}
    for r in rows:
        n, ok = per.get(r["task"], (0, 0))
        per[r["task"]] = (n + 1, ok + int(r["acc"]))
    return rows, {t: {"n": n, "passed": ok, "acc": ok / n if n else 0.0}
                  for t, (n, ok) in per.items()}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prefix", default="c1_")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--tasks", default="gsm8k,humaneval")
    ap.add_argument("--ports", default="18221,18222")
    ap.add_argument("--bin-dir", default=CUDA_BIN)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--cpus", default="0-15")
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--block", default="c1")
    a = ap.parse_args(argv)

    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    ports = [int(x) for x in a.ports.split(",")]
    if len(ports) < len(tasks):
        raise SystemExit("--ports 数量少于 --tasks")
    os.makedirs(RES, exist_ok=True)
    log_path = a.log if os.path.isabs(a.log) else os.path.join(HERE, a.log)
    t_all = time.time()
    with open(log_path, "a", encoding="utf-8") as fh:
        say(fh, "[c1] start tag=%s model=%s tasks=%s bin=%s" % (
            a.tag, a.model, tasks, a.bin_dir))
        for i, task in enumerate(tasks):
            env = dict(os.environ)
            env["E_G_PREFIX"] = a.prefix
            env["E_G_BIN"] = a.bin_dir
            cmd = [sys.executable, os.path.join(HERE, "e_g1_eval.py"), "run",
                   "--tag", a.tag, "--model", a.model, "--task", task,
                   "--port", str(ports[i]), "--threads", str(a.threads),
                   "--ctx", str(a.ctx), "--cpus", a.cpus,
                   "--n-predict", str(a.n_predict), "--bin-dir", a.bin_dir,
                   "--ngl", str(a.ngl), "--block", a.block]
            say(fh, "[c1] %s -> port %d" % (task, ports[i]))
            t0 = time.time()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, cwd=HERE).returncode
            say(fh, "[c1] %s rc=%d %.1f min" % (task, rc, (time.time() - t0) / 60))
            if rc != 0:
                say(fh, "[c1] WARN %s 失败, 继续后续任务 (可重跑续跑)" % task)

        rows, summary = aggregate(a.prefix, a.tag, a.out, tasks)
        say(fh, "[c1] aggregate %d rows -> %s  %s" % (len(rows), a.out, summary))
        say(fh, "[c1] done %.1f min" % ((time.time() - t_all) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
