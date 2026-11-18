"""等 G2 评测矩阵进程结束后: 双基线汇总 -> 写 .g2_eval_done, 让 chain_b 接着跑 G3。

chain_a 已被停掉 (它会在跑前重新做一次路由统计), 所以由本脚本补上收尾标记。
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PY = sys.executable
PREFIX = "e_g2_minicpm_"
CONFIGS = ["A_off_q4km", "Rofficial_mixq3im", "C_roleq4", "Clayer_route",
           "D_roleq4_im", "R_roleq3_im"]
TASKS = {"gsm8k": 200, "humaneval": 164, "gpqa": 100}


def say(m):
    print("%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), m), flush=True)


def matrix_alive():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object {$_.CommandLine -like '*e_g_eval_matrix.py --suite g2*'}).ProcessId"],
        capture_output=True, text=True)
    return bool(out.stdout.strip())


def counts():
    row = {}
    for tag in CONFIGS:
        for task in TASKS:
            p = os.path.join(RES, f"{PREFIX}{tag}_{task}.jsonl")
            n = 0
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    n = sum(1 for line in f if line.strip())
            row[(tag, task)] = n
    return row


def run_score(baseline, out_name):
    env = os.environ.copy()
    env["E_G_PREFIX"] = PREFIX
    cmd = [PY, os.path.join(HERE, "e_g1_eval.py"), "score", "--baseline", baseline,
           "--out", out_name]
    say("[score] " + " ".join(cmd))
    with open(os.path.join(RES, "e_g_watch_g2.log"), "a", encoding="utf-8") as f:
        f.write("\n=== %s %s\n" % (time.strftime("%H:%M:%S"), " ".join(cmd)))
        f.flush()
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    say("[score] rc=%d" % rc)
    return rc


def main():
    say("[watch] 等 G2 评测进程结束")
    while matrix_alive():
        time.sleep(300)
    say("[watch] 评测进程已结束, 统计完成题数")
    c = counts()
    complete = all(c[(t, k)] >= TASKS[k] for t in CONFIGS for k in TASKS)
    print(json.dumps({"complete": complete,
                      "counts": {"%s/%s" % k: v for k, v in c.items()}},
                     ensure_ascii=False), flush=True)
    run_score("A_off_q4km", "e_g2_minicpm_summary.json")
    run_score("Rofficial_mixq3im", "e_g2_minicpm_summary_vs_Rofficial_mixq3im.json")
    if complete:
        with open(os.path.join(RES, ".g2_eval_done"), "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        say("[watch] 已写 .g2_eval_done, chain_b 将接续 G3")
    else:
        say("[watch][WARN] 题数不齐, 未写标记; 需人工核对后再续 G3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
