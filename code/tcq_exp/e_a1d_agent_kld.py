"""E-A1d: 带 tool schema 的 agent 上下文里, tcmoe 的 token 级分布误差是否显著变大。

两个条件, 同一套打分协议 (复用 e_f9_kld 的 server/打分/指标实现):
  agent  2 条 BFCL 风格带 tools 的请求 -> /apply-template 渲染 -> ref 生成 8 token
         -> 对 8 个位置逐点 ref/target top-64 分布 -> KLD/JSD/top-1
  plain  data/math_eval64.jsonl 前 2 条 text, 同样 ref 生成 8 token -> 逐点 KLD

目标配置: base2 (tcmoe, domain=base) / tcq_nl_global (同文件, domain=global)
          / iq2xxs / ud_q2kxl

    python e_a1d_agent_kld.py --smoke
    python e_a1d_agent_kld.py

输出: results/e_a1d_agent_kld.json (+ results/.a1d_done)
      --smoke 写 results/e_a1d_smoke.json, 不写 done 标记
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e_f9_kld as F9  # noqa: E402  复用 server/打分/指标实现

RES = os.path.join(HERE, "results")
MODEL_ROOT = r"E:\CUDA11.8\model"
REF = os.path.join(MODEL_ROOT, "e20", "qwen36_f16.gguf")
TCMOE = os.path.abspath(os.path.join(HERE, "..", "tcmoe", "out", "model-tcmoe-q4k-nl.gguf"))
PLAIN_DATA = os.path.join(HERE, "data", "math_eval64.jsonl")

REF_PORT = 18181
TGT_PORT = 18182
THREADS = 16
N_CTX = 16384
K = 64
N_POS = 8

CONFIGS = {
    "base2": {"path": TCMOE, "domain": "base"},
    "tcq_nl_global": {"path": TCMOE, "domain": "global"},
    "iq2xxs": {"path": os.path.join(MODEL_ROOT, "baselines",
                                    "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"), "domain": None},
    "ud_q2kxl": {"path": os.path.join(MODEL_ROOT, "ud",
                                      "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"), "domain": None},
}
ORDER = ["base2", "tcq_nl_global", "iq2xxs", "ud_q2kxl"]

SYSTEM = ("You are a helpful assistant with access to tools. "
          "Use the provided tools when the user's request requires file operations.")


def _tool(name, desc, props, required):
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


def _p(desc):
    return {"type": "string", "description": desc}


TOOLS = [
    _tool("cd", "Change the current working directory.", {"path": _p("directory path")},
          ["path"]),
    _tool("ls", "List files and directories under a path.",
          {"path": _p("directory path")}, ["path"]),
    _tool("mv", "Move a file or directory to a new location.",
          {"source": _p("source path"), "destination": _p("destination path")},
          ["source", "destination"]),
    _tool("mkdir", "Create a new directory.", {"path": _p("directory path to create")},
          ["path"]),
    _tool("grep", "Search for a pattern in a file.",
          {"pattern": _p("regular expression"), "path": _p("file path")},
          ["pattern", "path"]),
    _tool("find", "Find files matching a name pattern under a directory.",
          {"pattern": _p("name pattern"), "path": _p("root directory")},
          ["pattern", "path"]),
    _tool("cat", "Print the contents of a file.", {"path": _p("file path")}, ["path"]),
    _tool("diff", "Compare two files line by line.",
          {"file1": _p("first file"), "file2": _p("second file")}, ["file1", "file2"]),
    _tool("sort", "Sort the lines of a file.",
          {"file": _p("file path"), "order": _p("ascending or descending")}, ["file"]),
    _tool("cp", "Copy a file or directory.",
          {"source": _p("source path"), "destination": _p("destination path")},
          ["source", "destination"]),
]

AGENT_REQUESTS = [
    "Move 'final_report.pdf' within document directory to 'temp' directory in document. "
    "Make sure to create the directory",
    "Sort 'final_report.pdf' in document directory.",
]


# ---------------------------------------------------------------- 序列构造
def apply_template(port, user_text):
    r = F9.post_json("http://127.0.0.1:%d/apply-template" % port,
                     {"messages": [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": user_text}],
                      "tools": TOOLS,
                      "add_generation_prompt": True}, timeout=120)
    prompt = r.get("prompt")
    if not prompt:
        raise RuntimeError("apply-template 未返回 prompt: %s" % str(r)[:300])
    return prompt


def generate(port, ids, n):
    """参考模型贪心生成 n 个 token, 返回生成 token id (return_tokens)"""
    r = F9.post_json("http://127.0.0.1:%d/completion" % port,
                     {"prompt": list(ids), "n_predict": n, "temperature": 0.0,
                      "top_k": 1, "seed": 1234, "ignore_eos": True,
                      "cache_prompt": True, "stream": False, "return_tokens": True},
                     timeout=1800)
    toks = [int(t) for t in (r.get("tokens") or [])]
    if len(toks) < n:
        print("[warn] 只生成 %d/%d 个 token" % (len(toks), n), flush=True)
    return toks[:n]


def build_seq(port, name, ctx_ids, npos):
    """ref 生成轨迹, 再逐位置取 ref top-k"""
    gen = generate(port, ctx_ids, npos)
    if not gen:
        raise RuntimeError("%s: 参考模型没有生成任何 token" % name)
    full = list(ctx_ids) + gen
    tops = []
    for i in range(len(gen)):
        tops.append(F9.score_position(port, full, len(ctx_ids) + i, K, True, None))
    print("[ref] %s ctx=%d gen=%d scored=%d" %
          (name, len(ctx_ids), len(gen), len(tops)), flush=True)
    return {"name": name, "ctx": list(ctx_ids), "gen": gen, "full": full, "tops": tops}


def load_plain(n, target_tokens=1000):
    """拼接到 ~1000 token, 与 agent 条件(≈1010 token)对齐上下文长度"""
    rows, buf, toks = [], [], 0
    with open(PLAIN_DATA, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            t = d.get("text") or d.get("prompt") or d.get("question")
            if not t:
                raise SystemExit("math_eval64.jsonl 字段不认识: %s" % list(d))
            buf.append(t)
            toks += int(d.get("n_tok") or 64)
            if toks >= target_tokens:
                rows.append({"name": "plain_%d" % len(rows), "text": "\n\n".join(buf)})
                buf, toks = [], 0
                if len(rows) >= n:
                    break
    return rows


# ---------------------------------------------------------------- 单配置
def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def eval_config(port, cfg, seqs, n_vocab):
    per = []
    for r in seqs:
        for i, target in enumerate(r["gen"]):
            idx = len(r["ctx"]) + i
            q_top = F9.score_position(port, r["full"], idx, K, True, cfg["domain"])
            m = F9.position_metrics(r["tops"][i], q_top, target, n_vocab)
            per.append({"seq": r["name"], "pos": i, "target_token": target,
                        "kld": m["kld"], "jsd": m["jsd"],
                        "top1_agree": m["top1_agree"]})
    return {"mean_kld": _mean([x["kld"] for x in per]),
            "mean_top1": _mean([x["top1_agree"] for x in per]),
            "per_pos": per}


def group_by_path(names):
    """同一 GGUF 只起一次 server (base2/tcq_nl_global 靠 domain 切换)"""
    groups, index = [], {}
    for n in names:
        key = os.path.normcase(os.path.abspath(CONFIGS[n]["path"]))
        if key not in index:
            index[key] = len(groups)
            groups.append([CONFIGS[n]["path"], []])
        groups[index[key]][1].append(n)
    return groups


def write_out(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="E-A1d agent(tools) vs plain token-level KLD")
    ap.add_argument("--smoke", action="store_true",
                    help="1 条 agent 请求 / 2 个位置 / 只跑 iq2xxs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    t_start = time.time()

    if args.smoke:
        names = ["iq2xxs"]
        npos = 2
        agent_texts = AGENT_REQUESTS[:1]
        plain_rows = []
        out_path = os.path.join(RES, "e_a1d_smoke.json")
    else:
        names = list(ORDER)
        npos = N_POS
        agent_texts = list(AGENT_REQUESTS)
        plain_rows = load_plain(2)
        out_path = args.out or os.path.join(RES, "e_a1d_agent_kld.json")

    for t in names:
        if not os.path.exists(CONFIGS[t]["path"]):
            raise SystemExit("模型文件不存在: %s" % CONFIGS[t]["path"])
    if not os.path.exists(REF):
        raise SystemExit("参考模型不存在: %s" % REF)

    out = {
        "protocol": {
            "experiment": "E-A1d agent(tool schema) vs plain token-level KLD",
            "ref": REF,
            "k": K, "positions_per_seq": npos,
            "agent_requests": agent_texts,
            "tools": [t["function"]["name"] for t in TOOLS],
            "plain_source": "%s (每条拼接到 ~1000 token, 共 %d 条)" % (PLAIN_DATA, len(plain_rows)),
            "alignment": "ref 贪心生成 npos 个 token 作轨迹; 逐位置 prompt=ids[:idx] "
                         "+ n_predict=1 取 top-k, 两侧同一前缀",
            "metrics": "KLD/JSD/top1_agree, 与 E-F9 同一 position_metrics "
                       "(top-k 并支撑 + __tail__ 桶)",
            "cache_prompt": True, "threads": THREADS, "n_ctx": N_CTX,
            "ref_port": REF_PORT, "target_port": TGT_PORT,
            "configs": {n: {"file": CONFIGS[n]["path"], "domain": CONFIGS[n]["domain"]}
                        for n in names},
            "smoke": bool(args.smoke),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "agent": {},
        "plain": {},
    }

    ref_proc = ref_fh = None
    tgt_proc = tgt_fh = None
    try:
        print("[ref] 启动 %s" % os.path.basename(REF), flush=True)
        t0 = time.time()
        ref_proc, ref_fh = F9.start_server(REF, REF_PORT, THREADS, N_CTX,
                                           os.path.join(RES, "e_a1d_ref.server.log"),
                                           3600)
        ref_load_s = time.time() - t0
        info = F9.server_info(REF_PORT)
        n_vocab = int(info.get("n_vocab") or F9.DEFAULT_N_VOCAB)
        print("[ref] 加载 %.1f 分钟, n_vocab=%d" % (ref_load_s / 60, n_vocab), flush=True)
        out["protocol"]["ref_load_s"] = round(ref_load_s, 1)
        out["protocol"]["ref_server_meta"] = info
        out["protocol"]["n_vocab"] = n_vocab

        # agent 条件
        agent_seqs = []
        for j, text in enumerate(agent_texts):
            prompt = apply_template(REF_PORT, text)
            ids = F9.tokenize(REF_PORT, prompt)
            print("[ref] agent_%d prompt_tokens=%d" % (j, len(ids)), flush=True)
            agent_seqs.append(build_seq(REF_PORT, "agent_%d" % j, ids, npos))

        # plain 条件
        plain_seqs = []
        for row in plain_rows:
            ids = F9.tokenize(REF_PORT, row["text"])
            plain_seqs.append(build_seq(REF_PORT, row["name"], ids, npos))

        out["protocol"]["agent_seq_tokens"] = [
            {"name": s["name"], "ctx": len(s["ctx"]), "gen": s["gen"]} for s in agent_seqs]
        out["protocol"]["plain_seq_tokens"] = [
            {"name": s["name"], "ctx": len(s["ctx"]), "gen": s["gen"]} for s in plain_seqs]
        out["protocol"]["ref_dist_done_s"] = round(time.time() - t_start, 1)
        write_out(out, out_path)

        for path, cfgs in group_by_path(names):
            print("[tgt] 启动 %s (%s)" % (os.path.basename(path), ",".join(cfgs)), flush=True)
            t0 = time.time()
            tgt_proc, tgt_fh = F9.start_server(
                path, TGT_PORT, THREADS, N_CTX,
                os.path.join(RES, "e_a1d_%s.server.log" % cfgs[0]), 1800)
            load_s = time.time() - t0
            tinfo = F9.server_info(TGT_PORT)
            tn_vocab = int(tinfo.get("n_vocab") or n_vocab)
            same_tok = True
            if agent_seqs:
                probe = F9.tokenize(TGT_PORT, apply_template(TGT_PORT, agent_texts[0]))
                same_tok = probe == agent_seqs[0]["ctx"]
            print("[tgt] %s 加载 %.1f 分钟, n_vocab=%d, tokenizer_match=%s" %
                  (cfgs[0], load_s / 60, tn_vocab, same_tok), flush=True)
            for cfg in cfgs:
                out["protocol"].setdefault("target_load_s", {})[cfg] = round(load_s, 1)
                out["protocol"].setdefault("tokenizer_match", {})[cfg] = bool(same_tok)
                if not same_tok:
                    raise SystemExit("[%s] tokenizer 与参考不一致, KLD 不可比" % cfg)
                if agent_seqs:
                    out["agent"][cfg] = eval_config(TGT_PORT, CONFIGS[cfg],
                                                    agent_seqs, n_vocab)
                    print("[%s/agent] mean_kld=%.4f mean_top1=%.3f" %
                          (cfg, out["agent"][cfg]["mean_kld"],
                           out["agent"][cfg]["mean_top1"]), flush=True)
                if plain_seqs:
                    out["plain"][cfg] = eval_config(TGT_PORT, CONFIGS[cfg],
                                                    plain_seqs, n_vocab)
                    print("[%s/plain] mean_kld=%.4f mean_top1=%.3f" %
                          (cfg, out["plain"][cfg]["mean_kld"],
                           out["plain"][cfg]["mean_top1"]), flush=True)
                write_out(out, out_path)
            F9.stop_server(tgt_proc, tgt_fh)
            tgt_proc = tgt_fh = None
            print("[tgt] 已停止 %s" % os.path.basename(path), flush=True)
    finally:
        F9.stop_server(tgt_proc, tgt_fh)
        F9.stop_server(ref_proc, ref_fh)
        out["protocol"]["elapsed_s"] = round(time.time() - t_start, 1)
        write_out(out, out_path)

    if not args.smoke:
        with open(os.path.join(RES, ".a1d_done"), "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))

    print("-" * 78, flush=True)
    for cond in ("agent", "plain"):
        for cfg, rec in out[cond].items():
            print("%-6s %-14s KLD=%.4f top1=%.3f" %
                  (cond, cfg, rec["mean_kld"], rec["mean_top1"]), flush=True)
    print("[done] %s (%.1f min)" % (out_path, (time.time() - t_start) / 60), flush=True)


if __name__ == "__main__":
    main()
