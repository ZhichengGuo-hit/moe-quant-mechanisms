"""E-G1 构建驱动: 路由统计 -> 类型规则 -> imatrix 语料 -> llama-imatrix -> 三份 GGUF -> 体积校验

gemma4 的专家是融合三维张量 (ffn_gate_up_exps), tcmoe 容器不可用, 因此这里的 C/D/R
都是"角色对齐的 llama-quantize":
  C  = 专家均匀 Q4_K (types_c), 角色对齐 (attn/ffn 内层 Q5_K, 归一化 F32)
  D  = C + --imatrix (三域语料)
  R  = 专家均匀 Q3_K + --imatrix (types_r)
  A/B 为官方 QAT 与社区 UD-Q4_K_XL, 不构建, 直接评测。

步骤状态写在 results/e_g1_build_state.json, 重跑只做未完成的步骤。

    python e_g1_build.py                # 全流程
    python e_g1_build.py --only types,corpus
    python e_g1_build.py --status
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BIN_CPU = os.path.join(ROOT, "llama.cpp", "build-tcmoe", "bin", "Release")
BIN_CUDA = os.path.join(ROOT, "llama.cpp", "build-cuda", "bin")
PY = sys.executable


def pick_bin(tool):
    """普通量化类型优先用 CUDA 构建 (imatrix 走 GPU), tcmoe 类型仍用 build-tcmoe。"""
    exe = tool + ".exe"
    for cand in (BIN_CUDA, BIN_CPU):
        if os.path.exists(os.path.join(cand, exe)):
            return os.path.join(cand, exe)
    return os.path.join(BIN_CPU, exe)

MROOT = os.path.join(r"E:\CUDA11.8\model\gemma4")
BF16_1 = os.path.join(MROOT, "BF16", "gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf")
BF16_2 = os.path.join(MROOT, "BF16", "gemma-4-26B-A4B-it-BF16-00002-of-00002.gguf")
OUT_DIR = os.path.join(MROOT, "g1")
STATE = os.path.join(RES, "e_g1_build_state.json")
META = os.path.join(RES, "e_g1_build_meta.json")
USAGE = os.path.join(RES, "e_g1_route_usage.json")
TYPES_C = os.path.join(RES, "e_g1_types_c.txt")
TYPES_CP = os.path.join(RES, "e_g1_types_cprime.txt")
TYPES_R = os.path.join(RES, "e_g1_types_r.txt")
TYPEFILES_META = os.path.join(RES, "e_g1_typefiles_meta.json")
CORPUS = os.path.join(DATA, "g1_calib.txt")
IMATRIX = os.path.join(RES, "e_g1_imatrix.gguf")

OUTS = {"c": os.path.join(OUT_DIR, "g1_c.g4q4k.gguf"),
        "cprime": os.path.join(OUT_DIR, "g1_cp.g4q4k_nx6.gguf"),
        "d": os.path.join(OUT_DIR, "g1_d.g4q4k_im.gguf"),
        "r": os.path.join(OUT_DIR, "g1_r.g4q3k_im.gguf")}

STEPS = ["route", "types", "corpus", "imatrix", "quant_c", "quant_cprime", "quant_d",
         "quant_r", "verify"]


def say(m):
    line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), m)
    print(line, flush=True)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"steps": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def save_state(st):
    os.makedirs(RES, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def run(cmd, log_name, state, step):
    """子进程输出写 results/<log_name>.log, 失败抛异常。"""
    os.makedirs(RES, exist_ok=True)
    log_path = os.path.join(RES, log_name + ".log")
    t0 = time.time()
    say("[cmd] %s" % " ".join(cmd)[:400])
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n=== %s %s\n" % (time.strftime("%H:%M:%S"), " ".join(cmd)))
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError("step %s 失败 rc=%d, 见 %s" % (step, p.returncode, log_path))
    say("[cmd] %s done in %.1f min" % (step, (time.time() - t0) / 60))
    return time.time() - t0


# ---------------- 各步骤 ----------------

def step_route(st):
    """轻量路由统计: 每域 8 条 / 最多 256 token。失败或超时降级为全零 stub。"""
    if os.path.exists(USAGE):
        say("[route] 复用已有 %s" % USAGE)
        return
    try:
        run([PY, os.path.join(HERE, "e_g1_route_stats.py"),
             "--model-dir", os.path.join(MROOT, "safetensors"),
             "--per-domain", "8", "--max-tokens", "256", "--out", USAGE],
            "e_g1_route_stats", st, "route")
    except Exception as e:  # noqa: BLE001  路由统计只影响 C′ 与补丁表, 不阻塞主流程
        say("[route] 失败, 用全零 stub 继续: %r" % e)
        stub_usage()


def stub_usage():
    """没有路由统计时的占位: 用 config.json 描述结构, 计数全零 (均匀分档)。"""
    cfg_path = os.path.join(MROOT, "safetensors", "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        raw = json.load(f)
    cfg = raw.get("text_config", raw)
    n_layer = int(cfg["num_hidden_layers"])
    n_exp = int(cfg["num_experts"])
    zeros = [[0] * n_exp for _ in range(n_layer)]
    out = {"config": {k: cfg[k] for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
                                          "moe_intermediate_size", "num_experts", "vocab_size",
                                          "num_attention_heads", "head_dim", "global_head_dim",
                                          "num_key_value_heads", "num_global_key_value_heads",
                                          "layer_types", "tie_word_embeddings") if k in cfg},
           "top_k_experts": cfg.get("num_experts_per_tok") or cfg.get("top_k_experts"),
           "domains": {}, "combined": {"counts": zeros},
           "stub": True, "note": "route stats 未产出, 用全零计数 (只影响 meta, 不影响 C/R 类型文件)"}
    with open(USAGE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    say("[route] stub 写入 %s" % USAGE)


def step_types(st):
    run([PY, os.path.join(HERE, "e_g1_typefiles.py"), "--usage", USAGE],
        "e_g1_typefiles", st, "types")


def step_corpus(st):
    run([PY, os.path.join(HERE, "e_g1_imatrix_corpus.py")], "e_g1_corpus", st, "corpus")


def need(path, what):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("缺文件: %s (%s)" % (path, what))


def step_imatrix(st):
    need(BF16_1, "BF16 GGUF 主分片")
    need(CORPUS, "imatrix 语料")
    run([pick_bin("llama-imatrix"), "-m", BF16_1, "-f", CORPUS,
         "-o", IMATRIX, "--output-format", "gguf", "-c", "512", "-t", "16", "-b", "512",
         "--output-frequency", "10"],
        "e_g1_imatrix", st, "imatrix")
    need(IMATRIX, "imatrix 输出")


def quant(name, base_type, typefile, imatrix, st):
    need(BF16_1, "BF16 GGUF 主分片")
    # 注意: llama-quantize 的选项循环只吃"程序名之后、第一个位置参数之前"的 -- 参数,
    # 选项写在位置参数后面会被静默忽略 (类型文件与 imatrix 全部失效)。
    cmd = [pick_bin("llama-quantize"), "--tensor-type-file", typefile]
    if imatrix:
        need(IMATRIX, "imatrix 输出")
        cmd += ["--imatrix", IMATRIX]
    cmd += [BF16_1, OUTS[name], base_type, "16"]
    run(cmd, "e_g1_quant_" + name, st, "quant_" + name)
    need(OUTS[name], "量化输出 " + name)


def step_verify(st):
    with open(TYPEFILES_META, encoding="utf-8") as f:
        tf = json.load(f)
    rows = []
    for name, key in (("c", "c"), ("cprime", "cprime"), ("d", "c"), ("r", "r")):
        size = os.path.getsize(OUTS[name])
        est = tf["variants"][key]["est_bytes"]
        dev = (size - est) / est * 100
        rows.append({"tag": name, "file": OUTS[name], "gib": round(size / 2**30, 3),
                     "est_gib": round(est / 2**30, 3), "dev_pct": round(dev, 2),
                     "within_5pct": abs(dev) <= 5})
        say("[verify] %s %.2f GiB (估 %.2f, 偏差 %+.1f%%)" % (name, size / 2**30, est / 2**30, dev))
    meta = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "builds": rows,
            "imatrix": {"file": IMATRIX, "size_gib": round(os.path.getsize(IMATRIX) / 2**30, 3),
                        "corpus": CORPUS, "corpus_bytes": os.path.getsize(CORPUS)
                        if os.path.exists(CORPUS) else 0, "model": BF16_1},
            "refs": {"a_qat": "官方 QAT (外部下载)", "b_ud": "UD-Q4_K_XL (外部下载)"},
            "note": "gemma4 专家为融合三维张量, tcmoe 容器不可用; C/D/R 均为角色对齐 "
                    "llama-quantize (专家整张量均匀档位)"}
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    bad = [r["tag"] for r in rows if not r["within_5pct"]]
    if bad:
        say("[verify] WARN 偏差 >5%%: %s (记录在 meta, 不中断)" % bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的步骤子集, 例: types,corpus")
    ap.add_argument("--force", action="store_true", help="忽略状态文件, 重跑选中步骤")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    st = load_state()
    if args.status:
        for s in STEPS:
            info = st["steps"].get(s)
            print("%-10s %s%s" % (s, "done" if info else "pending",
                                  ("  " + str(info)) if info else ""))
        return 0

    only = [x.strip() for x in args.only.split(",") if x.strip()] or STEPS
    os.makedirs(OUT_DIR, exist_ok=True)
    fns = {"route": step_route, "types": step_types, "corpus": step_corpus,
           "imatrix": step_imatrix, "verify": step_verify}
    say("[build] 步骤 %s" % only)
    for s in only:
        if s in st["steps"] and not args.force:
            say("[build] skip %s (已完成)" % s)
            continue
        t0 = time.time()
        if s == "quant_c":
            quant("c", "Q4_K", TYPES_C, False, st)
        elif s == "quant_cprime":
            quant("cprime", "Q4_K", TYPES_CP, False, st)
        elif s == "quant_d":
            quant("d", "Q4_K", TYPES_C, True, st)
        elif s == "quant_r":
            quant("r", "Q3_K", TYPES_R, True, st)
        else:
            fns[s](st)
        st["steps"][s] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "min": round((time.time() - t0) / 60, 1)}
        save_state(st)
    say("[build] all selected steps done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
