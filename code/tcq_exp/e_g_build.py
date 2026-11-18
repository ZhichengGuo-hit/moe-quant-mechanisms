"""E-G2/G3 构建驱动 (复用 G1 的角色对齐 llama-quantize 模板)。

G2 MiniCPM-V-4.7-35B-A3B (arch=qwen35moe, 专家融合三维张量, 无逐专家 → 层级别分档):
  C       = 专家 Q4_K + 非专家 Q5_K
  C_layer = 路由统计 top-k 层专家 Q4_K + 其余层 Q3_K + 非专家 Q5_K   (k=10, 40 层 25%)
  D       = C + --imatrix (三域语料)
  R       = 专家 Q3_K + --imatrix
  A / R_official = 官方 Q4_K_M / 官方 mix_q3gud_im, 下载即用, 不构建

G3 Xing4.0-29B-A4B (arch=xing4_0, 64 专家, blk.0-1 dense):
  C / D / R 同角色模板 (专家 Q4_K / Q4_K+im / Q3_K+im)

步骤状态写 results/e_<model>_build_state.json, 重跑只补未完成步骤。
imatrix 语料复用 G1 的三域 corpus (data/g1_calib.txt), 保证跨实验可比。

    python e_g_build.py --model g2
    python e_g_build.py --model g3 --only types,imatrix
    python e_g_build.py --model g2 --status
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BIN = os.path.join(ROOT, "llama-upstream", "build-cuda", "bin")
RAON_BIN = os.path.join(ROOT, "llama-raon", "build-cuda", "bin")
PY = sys.executable
# 转换脚本要 torch, 只能走 kt_env; gguf 从各自仓库的 gguf-py 取
PY_CONV = os.path.join(ROOT, "kt_env", "Scripts", "python.exe")
HEADER = os.path.join(HERE, "e_g_gguf_header.py")
CORPUS = os.path.join(DATA, "g1_calib.txt")

PRESETS = {
    "g2": {
        "name": "MiniCPM-V-4.7-35B-A3B",
        "src": r"E:\CUDA11.8\model\minicpm\MiniCPM-V-4_7-35B-A3B-F16.gguf",
        "outdir": r"E:\CUDA11.8\model\minicpm\g2",
        "route": os.path.join(RES, "e_g2_route_usage.json"),
        "typefiles": os.path.join(HERE, "e_g_typefiles.py"),
        "typefiles_args": [],
        "quant": [
            ("c", "e_g2_types_c.txt", "Q4_K", False),
            ("clayer", "e_g2_types_clayer.txt", "Q3_K", False),
            ("d", "e_g2_types_c.txt", "Q4_K", True),
            ("r", "e_g2_types_r.txt", "Q3_K", True),
        ],
        "refs": {"a": "官方 Q4_K_M (openbmb 官方 GGUF)",
                 "r_official": "官方 mix_q3gud_im (官方 imatrix 混档)"},
    },
    "g3": {
        "name": "Xing4.0-29B-A4B",
        "src": r"E:\CUDA11.8\model\xing\Xing4.0-29B-A4B-f16.gguf",
        "outdir": r"E:\CUDA11.8\model\xing\g3",
        "route": "",
        "typefiles": os.path.join(HERE, "e_g_typefiles.py"),
        "typefiles_args": [],
        "quant": [
            ("c", "e_g3_types_c.txt", "Q4_K", False),
            ("d", "e_g3_types_c.txt", "Q4_K", True),
            ("r", "e_g3_types_r.txt", "Q3_K", True),
        ],
        "refs": {"a": "官方 xing4_0-29b-iq4_xs",
                 "b1": "社区 Venastine Q4_K_M", "b2": "社区 dealignai CRACK Q4_K_M"},
    },
    # G5: 官方 BF16 GGUF 分片直接当 F16 源 (llama-quantize 自动读同目录分片)
    "g5": {
        "name": "GLM-4.7-Flash 30B-A3B",
        "src": r"E:\CUDA11.8\model\glm47\BF16\GLM-4.7-Flash-BF16-00001-of-00002.gguf",
        "outdir": r"E:\CUDA11.8\model\glm47\g5",
        "route": "",
        "typefiles": os.path.join(HERE, "e_g_typefiles.py"),
        "typefiles_args": [],
        "bin": BIN,
        "quant": [
            ("c", "e_g5_types_c.txt", "Q4_K", False),
            ("d", "e_g5_types_c.txt", "Q4_K", True),
            ("r", "e_g5_types_r.txt", "Q3_K", True),
        ],
        "refs": {"a": "官方 unsloth UD-Q2_K_XL", "b": "官方 unsloth UD-Q3_K_XL"},
    },
    # G6: nvidia 原始 safetensors -> upstream convert (arch nemotron_h_moe)
    "g6": {
        "name": "Nemotron-Cascade-2 30B-A3B",
        "src": r"E:\CUDA11.8\model\nemotron\f16\Nemotron-Cascade-2-30B-A3B-F16.gguf",
        "outdir": r"E:\CUDA11.8\model\nemotron\g6",
        "route": "",
        "typefiles": os.path.join(HERE, "e_g_typefiles.py"),
        "typefiles_args": [],
        "bin": BIN,
        "convert": {"script": os.path.join(ROOT, "llama-upstream", "convert_hf_to_gguf.py"),
                    "model": r"E:\CUDA11.8\model\nemotron\src",
                    "gguf_py": os.path.join(ROOT, "llama-upstream", "gguf-py")},
        "quant": [
            ("c", "e_g6_types_c.txt", "Q4_K", False),
            ("d", "e_g6_types_c.txt", "Q4_K", True),
            ("r", "e_g6_types_r.txt", "Q3_K", True),
        ],
        "refs": {"a": "i1-Q2_K", "b": "i1-Q3_K_M"},
    },
    # G7: KRAFTON safetensors (专家 MXFP8) -> fork convert (arch raon1), 二进制也只能用 fork
    "g7": {
        "name": "Raon-1-mini-preview",
        "src": r"E:\CUDA11.8\model\raon\f16\Raon-1-mini-preview-F16.gguf",
        "outdir": r"E:\CUDA11.8\model\raon\g7",
        "route": "",
        "typefiles": os.path.join(HERE, "e_g_typefiles.py"),
        "typefiles_args": [],
        "bin": RAON_BIN,
        "convert": {"script": os.path.join(ROOT, "llama-raon", "convert_hf_to_gguf.py"),
                    "model": r"E:\CUDA11.8\model\raon\src",
                    "gguf_py": os.path.join(ROOT, "llama-raon", "gguf-py")},
        "quant": [
            ("c", "e_g7_types_c.txt", "Q4_K", False),
            ("d", "e_g7_types_c.txt", "Q4_K", True),
            ("r", "e_g7_types_r.txt", "Q3_K", True),
        ],
        "refs": {"a": "官方 IQ4_XS"},
    },
}


def say(m):
    print("%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), m), flush=True)


def paths(p):
    out = {}
    for tag, *_ in p["quant"]:
        out[tag] = os.path.join(p["outdir"], "e_%s_%s.gguf" % (p["outdir"].rstrip("\\").split("\\")[-1], tag))
    return out


def state_path(model):
    return os.path.join(RES, "e_%s_build_state.json" % model)


HEAVY_LOCK = os.path.join(RES, ".g_heavy.lock")


@contextlib.contextmanager
def heavy_lock(tag):
    """imatrix / quantize / 路由统计都要把 60-70 GB 权重拉进内存(共 128 GB),
    全局互斥, 避免两条链同时跑把内存打爆。"""
    os.makedirs(RES, exist_ok=True)
    while True:
        try:
            fd = os.open(HEAVY_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            pid = 0
            try:
                pid = int(open(HEAVY_LOCK).read().strip())
            except Exception:
                pass
            if pid and not _alive(pid):
                try:
                    os.remove(HEAVY_LOCK)
                except OSError:
                    pass
                continue
            say("[lock] %s: 等 pid=%d 释放" % (tag, pid))
            time.sleep(60)
    try:
        yield
    finally:
        try:
            os.remove(HEAVY_LOCK)
        except OSError:
            pass


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def load_state(model):
    path = state_path(model)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"steps": {}}


def save_state(model, st):
    with open(state_path(model), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def run(cmd, log_name, step, env=None):
    os.makedirs(RES, exist_ok=True)
    log = os.path.join(RES, log_name + ".log")
    t0 = time.time()
    say("[cmd] %s" % " ".join(cmd)[:400])
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n=== %s %s\n" % (time.strftime("%H:%M:%S"), " ".join(cmd)))
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError("步骤 %s 失败 rc=%d, 见 %s" % (step, rc, log))
    say("[cmd] %s done %.1f min" % (step, (time.time() - t0) / 60))
    return time.time() - t0


def need(path, what):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("缺文件: %s (%s)" % (path, what))


def step_types(p, st, model):
    args = [PY, p["typefiles"], "--model", model]
    if model == "g2":
        if os.path.exists(p["route"]):
            args += ["--usage", p["route"], "--k", "10"]
        else:
            say("[types][WARN] 无路由统计, 跳过 C_layer (只出 C/R)")
    run(args, "e_%s_typefiles" % model, "types")


def step_convert(p, st, model):
    """G6/G7 只有原始 safetensors: 用对应仓库的 convert_hf_to_gguf.py 出 F16 GGUF。

    G7 的专家是 MXFP8, fork 的 raon.py 会在转换时反量化, 所以必须用 fork 的脚本 + fork 的 gguf-py。
    """
    c = p.get("convert")
    if not c:
        return
    out = p["src"]
    if os.path.exists(out) and os.path.getsize(out) > 1 << 30:
        say("[convert] 复用 %s" % out)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    need(os.path.join(c["model"], "config.json"), "HF config.json")
    env = os.environ.copy()
    env["PYTHONPATH"] = c["gguf_py"]
    with heavy_lock("convert_" + model):
        run([PY_CONV, c["script"], c["model"], "--outfile", out, "--outtype", "f16"],
            "e_%s_convert" % model, "convert", env)
    need(out, "F16 GGUF")


def step_dump(p, st, model):
    """张量清单来自真实 GGUF 头 (本地文件), 供 e_g_typefiles.py 生成分档规则。"""
    need(p["src"], "F16 GGUF")
    out = os.path.join(RES, "e_%s_tensors.txt" % model)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        say("[dump] 复用 %s" % out)
        return
    run([PY, HEADER, p["src"], "--dump", out], "e_%s_dump" % model, "dump")
    need(out, "张量清单")


def step_imatrix(p, st, model):
    need(p["src"], "F16 GGUF")
    need(CORPUS, "三域 imatrix 语料")
    out = os.path.join(RES, "e_%s_imatrix.gguf" % model)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        say("[imatrix] 复用 %s" % out)
        return
    with heavy_lock("imatrix"):
        run([os.path.join(p.get("bin", BIN), "llama-imatrix.exe"), "-m", p["src"], "-f", CORPUS,
             "-o", out, "--output-format", "gguf", "-c", "512", "-t", "16", "-b", "512",
             "--output-frequency", "10"],
            "e_%s_imatrix" % model, "imatrix")
    need(out, "imatrix 输出")


def quant(p, model, tag, typefile, base, use_im, st):
    need(p["src"], "F16 GGUF")
    outs = paths(p)
    tf = os.path.join(RES, typefile)
    need(tf, "类型文件")
    # 选项必须写在位置参数之前, 否则会被静默忽略
    cmd = [os.path.join(p.get("bin", BIN), "llama-quantize.exe"), "--tensor-type-file", tf]
    if use_im:
        im = os.path.join(RES, "e_%s_imatrix.gguf" % model)
        need(im, "imatrix")
        cmd += ["--imatrix", im]
    cmd += [p["src"], outs[tag], base, "16"]
    if os.path.exists(outs[tag]) and os.path.getsize(outs[tag]) > 0 and not st.get("force_step"):
        say("[quant] 已存在 %s, 跳过" % outs[tag])
        return
    with heavy_lock("quant_" + tag):
        run(cmd, "e_%s_quant_%s" % (model, tag), "quant_" + tag)
    need(outs[tag], "量化输出 " + tag)


def step_verify(p, model, st):
    meta_path = os.path.join(RES, "e_%s_typefiles_meta.json" % model)
    need(meta_path, "typefiles meta")
    with open(meta_path, encoding="utf-8") as f:
        tfm = json.load(f)
    outs = paths(p)
    rows = []
    for tag, tfile, base, use_im in p["quant"]:
        key = {"c": "c", "clayer": "clayer", "d": "c", "r": "r"}[tag]
        if key not in tfm["variants"]:
            rows.append({"tag": tag, "file": outs[tag], "note": "无估算(类型文件缺失)"})
            continue
        size = os.path.getsize(outs[tag])
        est = tfm["variants"][key]["est_bytes"]
        dev = (size - est) / est * 100
        rows.append({"tag": tag, "file": outs[tag], "gib": round(size / 2**30, 3),
                     "est_gib": round(est / 2**30, 3), "dev_pct": round(dev, 2),
                     "within_5pct": abs(dev) <= 5})
        say("[verify] %-7s %.2f GiB (估 %.2f, 偏差 %+.1f%%)"
            % (tag, size / 2**30, est / 2**30, dev))
    meta = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": p["name"],
            "builds": rows, "refs": p["refs"], "bin": p.get("bin", BIN),
            "imatrix": {"file": os.path.join(RES, "e_%s_imatrix.gguf" % model),
                        "size_gib": round(os.path.getsize(os.path.join(RES, "e_%s_imatrix.gguf" % model)) / 2**30, 3)
                        if os.path.exists(os.path.join(RES, "e_%s_imatrix.gguf" % model)) else 0,
                        "corpus": CORPUS, "corpus_bytes": os.path.getsize(CORPUS)},
            "note": "角色对齐 llama-quantize; 专家为融合张量 → 只能整层/整张量分档"}
    with open(os.path.join(RES, "e_%s_build_meta.json" % model), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["g2", "g3", "g5", "g6", "g7"])
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    p = PRESETS[args.model]
    steps = (["convert"] if p.get("convert") else []) + \
            (["dump"] if args.model in ("g5", "g6", "g7") else []) + \
            ["types", "imatrix"] + ["quant_" + t for t, *_ in p["quant"]] + ["verify"]
    st = load_state(args.model)
    if args.status:
        for s in steps:
            info = st["steps"].get(s)
            print("%-12s %s%s" % (s, "done" if info else "pending", ("  " + str(info)) if info else ""))
        return 0

    only = [x.strip() for x in args.only.split(",") if x.strip()] or steps
    os.makedirs(p["outdir"], exist_ok=True)
    say("[build] %s 步骤 %s" % (args.model, only))
    for s in only:
        if st["steps"].get(s) and not args.force:
            say("[build] skip %s (已完成)" % s)
            continue
        t0 = time.time()
        if s == "convert":
            step_convert(p, st, args.model)
        elif s == "dump":
            step_dump(p, st, args.model)
        elif s == "types":
            step_types(p, st, args.model)
        elif s == "imatrix":
            step_imatrix(p, st, args.model)
        elif s == "verify":
            step_verify(p, args.model, st)
        elif s.startswith("quant_"):
            tag = s[len("quant_"):]
            row = [r for r in p["quant"] if r[0] == tag][0]
            quant(p, args.model, *row, st=st)
        else:
            raise SystemExit("未知步骤 %s" % s)
        st["steps"][s] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "min": round((time.time() - t0) / 60, 1)}
        save_state(args.model, st)
    say("[build] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
