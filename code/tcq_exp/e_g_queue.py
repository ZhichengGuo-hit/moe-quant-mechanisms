"""E-G2/G3 夜间队列: 等依赖 -> 构建 -> 评测, 单线程顺序, 全程落日志。

两条链:
  queue_a (G2): 等 F16 + HF 分片 -> 路由统计 -> 构建 C/C_layer/D/R -> 等 F8 让出 GPU -> 评测 6x3
                -> 写 results/.g2_eval_done
  queue_b (G3): 等 G3 四个 GGUF -> 构建 C/D/R -> 等 .g2_eval_done -> 评测 6x3
                -> 写 results/.g3_eval_done

GPU 只允许一个 llama-server: queue_a 用端口 18111, queue_b 用 18112;
queue_a 在开跑前等 18093 (F8) 关闭, queue_b 等 g2 评测结束标记。

    python e_g_queue.py --chain a
    python e_g_queue.py --chain b
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
LOG = os.path.join(ROOT, "_tmp_g_queue.log")
QUANT = os.path.join(ROOT, "llama-upstream", "build-cuda", "bin", "llama-quantize.exe")
BIN_G = os.path.join(ROOT, "llama-upstream", "build-cuda", "bin")
# Raon-1-mini 的上游支持 pending, 只有 KRAFTON 自己的 fork 分支 raon-1-mini-preview 认 arch=raon1
RAON_REPO = os.path.join(ROOT, "llama-raon")
RAON_BIN = os.path.join(RAON_REPO, "build-cuda", "bin")
# 本机 MSVC/SDK 不在 PATH 上 (SDK 在 G 盘), 构建 fork 时必须显式给环境
VC_TOOLS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
SDK_ROOT = r"G:\Windows Kits\10"
SDK_VER = "10.0.26100.0"


def build_env():
    env = os.environ.copy()
    env["INCLUDE"] = ";".join([
        os.path.join(VC_TOOLS, "include"),
        os.path.join(SDK_ROOT, "Include", SDK_VER, "ucrt"),
        os.path.join(SDK_ROOT, "Include", SDK_VER, "shared"),
        os.path.join(SDK_ROOT, "Include", SDK_VER, "um"),
        os.path.join(SDK_ROOT, "Include", SDK_VER, "winrt")])
    env["LIB"] = ";".join([
        os.path.join(VC_TOOLS, "lib", "x64"),
        os.path.join(SDK_ROOT, "Lib", SDK_VER, "ucrt", "x64"),
        os.path.join(SDK_ROOT, "Lib", SDK_VER, "um", "x64")])
    env["PATH"] = ";".join([
        os.path.join(VC_TOOLS, "bin", "Hostx64", "x64"),
        os.path.join(SDK_ROOT, "bin", SDK_VER, "x64"),
        os.path.join(ROOT, "kt_env", "Scripts"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        env.get("PATH", "")])
    return env

F16_G2 = r"E:\CUDA11.8\model\minicpm\MiniCPM-V-4_7-35B-A3B-F16.gguf"
G2_SHARDS = [r"E:\CUDA11.8\model\minicpm\safetensors\model-%05d-of-00016.safetensors" % i
             for i in range(1, 17)]
G3_FILES = [r"E:\CUDA11.8\model\xing\Xing4.0-29B-A4B-f16.gguf",
            r"E:\CUDA11.8\model\xing\Xing4.0-29B-A4B-Q4_K_M.gguf",
            r"E:\CUDA11.8\model\xing\Xing4.0-29B-A4B-CRACK-Q4_K_M.gguf",
            r"E:\CUDA11.8\model\xing\xing4_0-29b-iq4_xs.gguf"]


def say(m):
    line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(path):
    # 下载器先预分配整文件长度, 只能以 .done 标记为准
    return os.path.exists(path + ".done")


def wait_files(paths, what, timeout_h=10):
    t0 = time.time()
    while True:
        miss = [p for p in paths if not done(p)]
        if not miss:
            say("[wait] %s 全部就绪" % what)
            return True
        if time.time() - t0 > timeout_h * 3600:
            say("[wait] %s 超时, 缺 %d 个: %s" % (what, len(miss), miss[:2]))
            return False
        time.sleep(120)


def run(cmd, step, env=None):
    say("[run] %s" % " ".join(cmd)[:300])
    t0 = time.time()
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in step)
    log = os.path.join(RES, "e_g_run_%s.log" % safe)
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n=== %s %s\n" % (time.strftime("%H:%M:%S"), " ".join(cmd)))
        f.flush()
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    say("[run] %s rc=%d %.1f min" % (step, p.returncode, (time.time() - t0) / 60))
    return p.returncode


def port_open(port):
    s = socket.socket()
    s.settimeout(2)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def wait_port_free(port, timeout_h=12):
    t0 = time.time()
    while port_open(port):
        if time.time() - t0 > timeout_h * 3600:
            say("[wait] 端口 %d 仍被占用, 继续等" % port)
            t0 = time.time()
        time.sleep(180)
    say("[wait] 端口 %d 已空闲" % port)


def score_steps(prefix, baseline, extra_baseline, tag_name, env):
    run([PY, os.path.join(HERE, "e_g1_eval.py"), "score", "--baseline", baseline,
         "--out", "e_%s_summary.json" % tag_name], "score(%s)" % tag_name, env)
    if extra_baseline:
        run([PY, os.path.join(HERE, "e_g1_eval.py"), "score", "--baseline", extra_baseline,
             "--out", "e_%s_summary_vs_%s.json" % (tag_name, extra_baseline)],
            "score(%s vs %s)" % (tag_name, extra_baseline), env)


def chain_a():
    if not wait_files([QUANT], "upstream 二进制", timeout_h=3):
        return 1
    if not wait_files([F16_G2], "G2 F16", timeout_h=6):
        return 1
    task_list = "gsm8k,humaneval,gpqa"
    # 第一段: 不依赖路由统计的三个构建 (C/D/R) 先跑, 利用下载 HF 分片的空窗
    r = run([PY, os.path.join(HERE, "e_g_build.py"), "--model", "g2",
             "--only", "types,imatrix,quant_c,quant_d,quant_r,verify"], "g2 build (C/D/R)")
    if r != 0:
        return 1
    # 第二段: C_layer 需要路由统计 -> 等 HF 分片 -> 统计 -> 强制重出类型文件 -> 量化
    if not wait_files(G2_SHARDS, "G2 HF 分片", timeout_h=10):
        return 1
    r = run([PY, os.path.join(HERE, "e_g2_route_stats.py"),
             "--per-domain", "8", "--max-tokens", "256"], "g2 route")
    if r != 0:
        say("[warn] 路由统计失败, C_layer 缺失")
    r = run([PY, os.path.join(HERE, "e_g_build.py"), "--model", "g2",
             "--only", "types,quant_clayer,verify", "--force"], "g2 build (C_layer)")
    if r != 0:
        say("[warn] C_layer 构建失败, 评测降为 5 配置")
    wait_port_free(18093)
    cfgs = "A_off_q4km,Rofficial_mixq3im,C_roleq4,Clayer_route,D_roleq4_im,R_roleq3_im"
    clayer = r"E:\CUDA11.8\model\minicpm\g2\e_g2_clayer.gguf"
    if not os.path.exists(clayer):
        say("[warn] 无 C_layer 产物, 评测降为 5 配置")
        cfgs = "A_off_q4km,Rofficial_mixq3im,C_roleq4,D_roleq4_im,R_roleq3_im"
    r = run([PY, os.path.join(HERE, "e_g_eval_matrix.py"), "--suite", "g2",
             "--configs", cfgs, "--tasks", task_list, "--block", "50",
             "--n-predict", "1024"], "g2 eval")
    if r != 0:
        return 1
    env = os.environ.copy()
    env["E_G_PREFIX"] = "e_g2_minicpm_"
    score_steps("e_g2_minicpm_", "A_off_q4km", "Rofficial_mixq3im", "g2_minicpm", env)
    open(os.path.join(RES, ".g2_eval_done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    say("[chain_a] done")
    return 0


def chain_b():
    if not wait_files([QUANT], "upstream 二进制", timeout_h=3):
        return 1
    if not wait_files(G3_FILES, "G3 权重", timeout_h=10):
        return 1
    r = run([PY, os.path.join(HERE, "e_g_build.py"), "--model", "g3"], "g3 build")
    if r != 0:
        return 1
    mark = os.path.join(RES, ".g2_eval_done")
    while not os.path.exists(mark):
        time.sleep(300)
    say("[wait] g2 评测结束, 开始 g3 评测")
    wait_port_free(18111)
    r = run([PY, os.path.join(HERE, "e_g_eval_matrix.py"), "--suite", "g3",
             "--tasks", "gsm8k,humaneval,gpqa", "--block", "50", "--n-predict", "1024"], "g3 eval")
    if r != 0:
        return 1
    env = os.environ.copy()
    env["E_G_PREFIX"] = "e_g3_xing_"
    score_steps("e_g3_xing_", "A_off_iq4xs", "B1_venastine_q4km", "g3_xing", env)
    run([PY, os.path.join(HERE, "e_g3_community_meta.py"), "--attach"], "g3 meta attach", env)
    open(os.path.join(RES, ".g3_eval_done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    say("[chain_b] done")
    return 0


def wait_f8_done():
    """E-F8 硬探针 (gpqa -> aime30 两段) 未结束前不碰 GPU/CPU: 它独占 16 线程与 18093 端口。

    AIME30 由 e_f8_aime_chain.py 在 gpqa 退出后才拉起, 两者之间有 1-2 分钟空窗;
    只认 e_f8_hard_probe.py 进程会在空窗里误判"已结束", 所以同时看包装器进程,
    并在无进程时给 10 分钟宽限 + 以 e_f8_summary_aime30.json 为完成标记。
    """
    aime_done = os.path.join(RES, "e_f8_summary_aime30.json")
    while True:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object {$_.CommandLine -like '*e_f8_hard_probe.py*' -or "
             "$_.CommandLine -like '*e_f8_aime_chain.py*'}).ProcessId"],
            capture_output=True, text=True)
        pids = out.stdout.strip().split()
        if pids:
            say("[wait] E-F8 仍在跑 (pid=%s), 等待 10 分钟" % " ".join(pids))
            time.sleep(600)
            continue
        if os.path.exists(aime_done):
            break
        say("[wait] E-F8 进程已退出, 等 AIME 包装器接续 (宽限 10 分钟)")
        time.sleep(600)
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object {$_.CommandLine -like '*e_f8_hard_probe.py*' -or "
             "$_.CommandLine -like '*e_f8_aime_chain.py*'}).ProcessId"],
            capture_output=True, text=True)
        if not out.stdout.strip() and not os.path.exists(aime_done):
            say("[warn] AIME 两臂未接续 (无进程且无 e_f8_summary_aime30.json), 继续后续流程")
            break
    wait_port_free(18093)
    say("[wait] E-F8 已结束")


def smoke_one(suite, tag, model, ngl, port, out, bin_dir=None):
    import e17_task_eval as E17
    log = os.path.join(RES, "e_g_smoke_%s_%s.server.log" % (suite, tag))
    t0 = time.time()
    try:
        proc, lg = E17.start_server(bin_dir or BIN_G, model, port, 16, 2048, log,
                                    extra=("-ngl", str(ngl)))
    except BaseException as e:  # noqa: BLE001  加载失败即"不支持"的取证
        tail = ""
        if os.path.exists(log):
            with open(log, encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-12:])
        rec = {"suite": suite, "tag": tag, "model": model, "ngl": ngl,
               "loaded": False, "error": repr(e)[:300], "log_tail": tail.strip()[:1500]}
        out.append(rec)
        say("[smoke] %s/%s 加载失败: %s" % (suite, tag, rec["error"]))
        return False
    try:
        content, tim = E17.gen(port, "Question: What is 2+2?\nAnswer:", 32,
                               ["\n\nQuestion:"])
        rec = {"suite": suite, "tag": tag, "model": model, "ngl": ngl, "loaded": True,
               "load_s": round(time.time() - t0, 1), "content": content[:300],
               "predicted_per_second": (tim or {}).get("predicted_per_second")}
        out.append(rec)
        say("[smoke] %s/%s OK %.1fs, %.1f tok/s, 输出 %r"
            % (suite, tag, rec["load_s"], rec["predicted_per_second"] or 0, content[:60]))
        return True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()
        lg.close()


def ensure_raon_build():
    """KRAFTON fork (branch raon-1-mini-preview) 构建; 已存在直接复用。"""
    if os.path.exists(os.path.join(RAON_BIN, "llama-server.exe")):
        say("[raon] 复用已有构建 %s" % RAON_BIN)
        return True
    if not os.path.isdir(os.path.join(RAON_REPO, ".git")):
        if os.path.exists(os.path.join(RAON_REPO, "CMakeLists.txt")):
            say("[raon] 已有源码 (tarball 解包, 无 .git), 直接构建")
        elif os.path.isdir(RAON_REPO) and os.listdir(RAON_REPO):
            say("[raon] 目录非空且无 CMakeLists, 跳过构建")
            return False
        else:
            r = run(["git", "clone", "--depth", "1", "-b", "raon-1-mini-preview",
                     "https://github.com/krafton-ai/llama.cpp", RAON_REPO], "raon clone")
            if r != 0:
                return False
    env = build_env()
    cl = os.path.join(VC_TOOLS, "bin", "Hostx64", "x64", "cl.exe")
    r = run(["cmake", "-S", RAON_REPO, "-B", os.path.join(RAON_REPO, "build-cuda"),
             "-G", "Ninja", "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_CUDA_ARCHITECTURES=75",
             "-DCMAKE_C_COMPILER=" + cl, "-DCMAKE_CXX_COMPILER=" + cl],
            "raon cmake", env)
    if r != 0:
        return False
    r = run(["cmake", "--build", os.path.join(RAON_REPO, "build-cuda"), "-j", "16"],
            "raon build", env)
    ok = r == 0 and os.path.exists(os.path.join(RAON_BIN, "llama-server.exe"))
    say("[raon] 构建 %s" % ("成功" if ok else "失败"))
    return ok


def chain_e():
    """E-G5/G6/G7: 下载 -> 等 E-F8 收尾 -> 逐模型冒烟 -> 冒烟通过的配置跑三任务矩阵。"""
    r = run([PY, os.path.join(HERE, "e_g_dl.py"), "--only", "g5,g6,g7"], "g5-g7 download")
    if r != 0:
        say("[warn] 下载返回非 0, 用已落盘文件继续")
    wait_f8_done()
    raon_ok = ensure_raon_build()
    bins = {"g5": BIN_G, "g6": BIN_G, "g7": RAON_BIN if raon_ok else None}

    import e_g_eval_matrix as M
    smoke = []
    for suite in ("g5", "g6", "g7"):
        s = M.SUITES[suite]
        bin_dir = bins[suite]
        if not bin_dir:
            say("[%s] 无可用二进制 (fork 构建失败), 跳过" % suite)
            open(os.path.join(RES, ".%s_blocked" % suite), "w").write(
                time.strftime("%Y-%m-%dT%H:%M:%S"))
            continue
        port = s["port"]
        if port_open(port):
            wait_port_free(port, timeout_h=2)
        ok_tags = []
        for tag, (model, ngl) in s["configs"].items():
            if not (os.path.exists(model) and os.path.exists(model + ".done")):
                smoke.append({"suite": suite, "tag": tag, "model": model,
                              "loaded": False, "error": "file missing"})
                say("[smoke] %s/%s 缺文件 %s" % (suite, tag, model))
                continue
            if smoke_one(suite, tag, model, ngl, port, smoke, bin_dir):
                ok_tags.append(tag)
        with open(os.path.join(RES, "e_g567_smoke.json"), "w", encoding="utf-8") as f:
            json.dump(smoke, f, ensure_ascii=False, indent=2)
        if not ok_tags:
            say("[%s] 无可用配置, 跳过评测 (见 results/e_g567_smoke.json)" % suite)
            open(os.path.join(RES, ".%s_blocked" % suite), "w").write(
                time.strftime("%Y-%m-%dT%H:%M:%S"))
            continue
        env = os.environ.copy()
        env["E_G_PREFIX"] = s["prefix"]
        rc = run([PY, os.path.join(HERE, "e_g_eval_matrix.py"), "--suite", suite,
                  "--configs", ",".join(ok_tags), "--tasks", "gsm8k,humaneval,gpqa",
                  "--block", "50", "--n-predict", "1024", "--bin-dir", bin_dir],
                 "%s eval" % suite, env)
        if rc != 0:
            say("[warn] %s 评测 rc=%d" % (suite, rc))
        score_steps(s["prefix"], s["baseline"], "", "%s_%s" % (suite, s["baseline"]), env)
        open(os.path.join(RES, ".%s_eval_done" % suite), "w").write(
            time.strftime("%Y-%m-%dT%H:%M:%S"))
    say("[chain_e] done")
    return 0


def chain_f():
    """E-G5/G6/G7 的 TCQ 臂 (C/D/R): 下 F16/BF16 源 -> 转换 -> 分档量化 -> 三任务矩阵。

    与 chain_e 的区别: chain_e 只评官方量化档; chain_f 用同源 F16 自建 C/D/R,
    与 G2/G3 的角色模板一致 (专家 Q4_K/Q3_K + 非专家 Q5_K, D/R 挂三域 imatrix)。
    """
    r = run([PY, os.path.join(HERE, "e_g_dl.py"), "--only", "g5,g6,g7"], "g5-g7 base download")
    if r != 0:
        say("[warn] 下载返回非 0, 用已落盘文件继续")
    raon_ok = ensure_raon_build()
    bins = {"g5": BIN_G, "g6": BIN_G, "g7": RAON_BIN if raon_ok else None}

    import e_g_eval_matrix as M
    for suite in ("g5", "g6", "g7"):
        if not bins[suite]:
            say("[%s] 无可用二进制 (fork 构建失败), 跳过 TCQ 臂" % suite)
            continue
        rc = run([PY, os.path.join(HERE, "e_g_build.py"), "--model", suite],
                 "%s tcq build" % suite)
        if rc != 0:
            say("[warn] %s 构建未完成, 见 results/e_%s_*.log" % (suite, suite))
            open(os.path.join(RES, ".%s_tcq_blocked" % suite), "w").write(
                time.strftime("%Y-%m-%dT%H:%M:%S"))
            continue
        s = M.SUITES[suite]
        if port_open(s["port"]):
            wait_port_free(s["port"], timeout_h=2)
        env = os.environ.copy()
        env["E_G_PREFIX"] = s["prefix"]
        rc = run([PY, os.path.join(HERE, "e_g_eval_matrix.py"), "--suite", suite,
                  "--configs", "C_roleq4,D_roleq4_im,R_roleq3_im",
                  "--tasks", "gsm8k,humaneval,gpqa", "--block", "50",
                  "--n-predict", "1024", "--bin-dir", bins[suite]],
                 "%s tcq eval" % suite, env)
        if rc != 0:
            say("[warn] %s TCQ 评测 rc=%d" % (suite, rc))
        score_steps(s["prefix"], s["baseline"], "", "%s_%s" % (suite, s["baseline"]), env)
        open(os.path.join(RES, ".%s_tcq_eval_done" % suite), "w").write(
            time.strftime("%Y-%m-%dT%H:%M:%S"))
    say("[chain_f] done")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, choices=["a", "b", "e", "f"])
    args = ap.parse_args()
    if args.chain == "e":
        return chain_e()
    if args.chain == "f":
        return chain_f()
    return chain_a() if args.chain == "a" else chain_b()


if __name__ == "__main__":
    raise SystemExit(main())
