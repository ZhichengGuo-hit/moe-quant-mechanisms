#!/usr/bin/env python
"""E-A1b: LongMemEval 前 100 题, 6 配置 (纯文本长程记忆, 零环境依赖).

数据: data/longmemeval_oracle.json (默认, 证据会话) / longmemeval_s.json (全量 115k 上下文).
协议: 官方 LONGMEMEVAL_PROMPT_DIRECT + json 历史格式; 六臂统一 token 预算 (默认 8192),
      超出部分从**最旧**的历史截断 (保留 Current Date/Question/Answer: 尾部).
      生成用各配置自己的 llama-server; 判定用固定本地 judge (gemma-4-26B-A4B-UD-Q4_K_XL),
      官方 anscheck prompt, yes/no. 本地 judge 与官方 GPT-4o judge 不完全同口径, 仅作臂间比较
      (跨研究数字只作方向锚点).

    python e_a1b_longmemeval.py --phase gen --configs iq2xxs --limit 3
    python e_a1b_longmemeval.py --phase all
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
MODEL_ROOT = r"E:\CUDA11.8\model"
CUDA_BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-server.exe"
CPU_BIN = r"E:\CUDA11.8\train\llama.cpp\build-tcmoe\bin\Release\llama-server.exe"
TCMOE = os.path.abspath(os.path.join(HERE, "..", "tcmoe", "out", "model-tcmoe-q4k-nl.gguf"))
JUDGE = os.path.join(MODEL_ROOT, "gemma4", "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")

CONFIGS = {
    "iq2xxs": {"path": os.path.join(MODEL_ROOT, "baselines", "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"),
               "bin": CUDA_BIN, "ngl": 99, "domain": None},
    "ud_q2kxl": {"path": os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
                 "bin": CUDA_BIN, "ngl": 99, "domain": None},
    "ud_q4km": {"path": os.path.join(MODEL_ROOT, "Qwen3.6-35B-A3B",
                                     "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
                "bin": CUDA_BIN, "ngl": 20, "domain": None},
    "tcq_nl_global": {"path": TCMOE, "bin": CPU_BIN, "ngl": 0, "domain": "global"},
    "tcq_nl_math": {"path": TCMOE, "bin": CPU_BIN, "ngl": 0, "domain": "math"},
    "base2": {"path": TCMOE, "bin": CPU_BIN, "ngl": 0, "domain": "base"},
}
PORT = 18145
JPORT = 18149
PROMPT_DIRECT = (
    'I will give you several history chats between you and a user. Please answer the question based on the '
    'relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:'
)


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ---------------------------------------------------------------- 数据 / prompt
def load_items(limit):
    path = os.path.join(DATA, "longmemeval_oracle.json")
    if not os.path.exists(path):
        raise SystemExit("缺少数据: %s (从 hf-mirror 下载 xiaowu0162/longmemeval)" % path)
    rows = json.load(open(path, encoding="utf-8"))
    return rows[:limit]


def _clean(value):
    """去掉官方数据里的 has_answer 标注字段 (与 evalscope longmemeval/utils.py 一致)."""
    if isinstance(value, list):
        return [_clean(x) for x in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k != "has_answer"}
    return value


def history_string(entry):
    """官方 history_format='json': 每个 session 一行 JSON."""
    out = ""
    for i, (date, sess) in enumerate(zip(entry["haystack_dates"], entry["haystack_sessions"]), 1):
        out += ("\n### Session %d:\nSession Date: %s\nSession Content:\n%s\n"
                % (i, date, "\n" + json.dumps(_clean(sess), ensure_ascii=False)))
    return out


def judge_prompt(qtype, question, answer, response):
    """官方 LongMemEval anscheck prompt (与 evalscope longmemeval/utils.py 同文)."""
    if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        tpl = ('I will give you a question, a correct answer, and a response from a model. Please answer yes if '
               'the response contains the correct answer. Otherwise, answer no. If the response is equivalent to '
               'the correct answer or contains all the intermediate steps to get the correct answer, you should '
               'also answer yes. If the response only contains a subset of the information required by the '
               'answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n'
               'Is the model response correct?')
    elif qtype == "temporal-reasoning":
        tpl = ('I will give you a question, a correct answer, and a response from a model. Please answer yes if '
               'the response contains the correct answer. Otherwise, answer no. If the response is equivalent to '
               'the correct answer or contains all the intermediate steps to get the correct answer, you should '
               'also answer yes. If the response only contains a subset of the information required by the '
               'answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the '
               'question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors '
               '(e.g., predicting 19 days when the answer is 18), the model\'s response is still correct. \n\n'
               'Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct?')
    elif qtype == "knowledge-update":
        tpl = ('I will give you a question, a correct answer, and a response from a model. Please answer yes if '
               'the response contains the correct answer. Otherwise, answer no. If the response contains some '
               'previous information along with an updated answer, the response should be considered as correct '
               'as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\n'
               'Model Response: {}\n\nIs the model response correct?')
    elif qtype == "single-session-preference":
        tpl = ('I will give you a question, a rubric for desired personalized response, and a response from a '
               'model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. '
               'The model does not need to reflect all the points in the rubric. The response is correct as long '
               "as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
               'Rubric: {}\n\nModel Response: {}\n\nIs the model response correct?')
    else:
        raise ValueError("unknown question_type: %s" % qtype)
    return tpl.format(question, answer, response)


# ---------------------------------------------------------------- server
def wait_port_free(port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2)
        except Exception:  # noqa: BLE001
            return True
        time.sleep(1)
    return False


def start_server(cfg, port, ctx, log_path, timeout=1800):
    assert wait_port_free(port), "端口 %d 仍被占用" % port
    cmd = [cfg["bin"], "-m", cfg["path"], "-c", str(ctx), "-ngl", str(cfg["ngl"]),
           "-t", "16", "--host", "127.0.0.1", "--port", str(port), "-np", "1",
           "--jinja", "--chat-template-kwargs", '{"enable_thinking": false}',
           "--no-webui", "--verbosity", "1"]
    fh = open(log_path, "w", encoding="utf-8")
    fh.write(" ".join(cmd) + "\n")
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit("llama-server 退出 code=%s (见 %s)" % (proc.returncode, log_path))
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as r:
                if r.status == 200:
                    log("server up %.0fs (%s)" % (time.time() - t0, os.path.basename(cfg["path"])))
                    return proc, fh
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise SystemExit("llama-server 启动超时")


def stop_server(proc, fh):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        fh.close()
    except Exception:  # noqa: BLE001
        pass


def post_json(port, path, payload, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tokenize(port, text):
    return post_json(port, "/tokenize", {"content": text, "add_special": False},
                     timeout=120)["tokens"]


def render(port, user_text):
    try:
        r = post_json(port, "/apply-template",
                      {"messages": [{"role": "user", "content": user_text}]}, timeout=120)
        if r.get("prompt"):
            return r["prompt"]
    except Exception:  # noqa: BLE001
        pass
    return "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % user_text


# ---------------------------------------------------------------- 生成
def gen_one(name, items, ctx_tokens, limit):
    cfg = CONFIGS[name]
    out_path = os.path.join(RES, "a1b_gen_%s.jsonl" % name)
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                done[rec["qid"]] = rec
    todo = [it for it in items if it["question_id"] not in done]
    if not todo:
        log("[%s] gen 已完成 %d 题" % (name, len(done)))
        return
    proc = fh = None
    try:
        proc, fh = start_server(cfg, PORT, ctx_tokens + 1024,
                                os.path.join(RES, "a1b_%s.server.log" % name))
        budget = ctx_tokens - 512
        with open(out_path, "a", encoding="utf-8") as f:
            for i, it in enumerate(todo, 1):
                user = PROMPT_DIRECT.format(history_string(it), it["question_date"], it["question"])
                prompt = render(PORT, user)
                ids = tokenize(PORT, prompt)
                if len(ids) > budget:
                    ids = ids[-budget:]  # 保留尾部 (问题 + 最近会话)
                payload = {"prompt": ids, "n_predict": 256, "temperature": 0.0,
                           "top_k": 1, "seed": 20261111, "cache_prompt": False,
                           "stop": ["\n\nQuestion:", "### Session", "<|im_end|>"], "stream": False}
                if cfg.get("domain"):
                    payload["domain"] = cfg["domain"]
                r = post_json(PORT, "/completion", payload)
                ans = (r.get("content") or "").strip()
                rec = {"qid": it["question_id"], "qtype": it["question_type"],
                       "question": it["question"], "gold": it["answer"], "response": ans,
                       "prompt_tokens": len(ids), "truncated": len(tokenize(PORT, prompt)) > budget}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                log("[%s] %d/%d %s (%s) pt=%d" % (name, i, len(todo), rec["qid"], rec["qtype"], len(ids)))
    finally:
        stop_server(proc, fh)


# ---------------------------------------------------------------- 判定
def judge_one(name):
    gen_path = os.path.join(RES, "a1b_gen_%s.jsonl" % name)
    out_path = os.path.join(RES, "a1b_judge_%s.jsonl" % name)
    if not os.path.exists(gen_path):
        log("[%s] 无生成文件, 跳过判定" % name)
        return
    rows = [json.loads(l) for l in open(gen_path, encoding="utf-8") if l.strip()]
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                done[rec["qid"]] = rec
    todo = [r for r in rows if r["qid"] not in done]
    if not todo:
        log("[%s] judge 已完成 %d 题" % (name, len(rows)))
        return
    jcfg = {"bin": CUDA_BIN, "path": JUDGE, "ngl": 99, "domain": None}
    proc = fh = None
    try:
        proc, fh = start_server(jcfg, JPORT, 8192, os.path.join(RES, "a1b_judge.server.log"))
        with open(out_path, "a", encoding="utf-8") as f:
            for r in todo:
                jp = judge_prompt(r["qtype"], r["question"], r["gold"], r["response"])
                d = post_json(JPORT, "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": jp}],
                    "temperature": 0.0, "max_tokens": 8, "cache_prompt": False})
                txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                ok = 1 if txt.strip().lower().startswith("yes") else 0
                rec = {"qid": r["qid"], "qtype": r["qtype"], "correct": ok, "judge_raw": txt.strip()[:20]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                log("[%s] judge %s -> %d" % (name, r["qid"], ok))
    finally:
        stop_server(proc, fh)


# ---------------------------------------------------------------- 汇总
def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, p)


def boot_ci(diffs, n_boot=1000, seed=20261104):
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return [None, None]
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return [means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]]


def summarize(names, items, baseline="iq2xxs"):
    res, per_q = {}, {}
    for name in names:
        p = os.path.join(RES, "a1b_judge_%s.jsonl" % name)
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        by_qid = {r["qid"]: r["correct"] for r in rows}
        per_q[name] = by_qid
        types = {}
        for r in rows:
            types.setdefault(r["qtype"], []).append(r["correct"])
        res[name] = {
            "n": len(rows),
            "acc": sum(r["correct"] for r in rows) / len(rows),
            "per_type": {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in types.items()},
        }
    if baseline in per_q:
        for name, by_qid in per_q.items():
            if name == baseline:
                continue
            common = sorted(set(by_qid) & set(per_q[baseline]))
            b = sum(1 for q in common if by_qid[q] == 1 and per_q[baseline][q] == 0)
            c = sum(1 for q in common if by_qid[q] == 0 and per_q[baseline][q] == 1)
            diffs = [by_qid[q] - per_q[baseline][q] for q in common]
            res[name]["vs_%s" % baseline] = {
                "paired_n": len(common), "b": b, "c": c, "mcnemar_p": mcnemar(b, c),
                "acc_diff": sum(diffs) / len(diffs) if diffs else None,
                "boot_ci95": boot_ci(diffs),
            }
    out = {"protocol": {"data": "longmemeval_oracle first %d" % len(items),
                        "judge": os.path.basename(JUDGE),
                        "prompts": "official LONGMEMEVAL_PROMPT_DIRECT + anscheck (yes/no)",
                        "notes": "本地 judge, 与官方 GPT-4o 不完全同口径, 仅作臂间比较"},
           "configs": res}
    path = os.path.join(RES, "a1b_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("summary -> %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ctx-tokens", type=int, default=8192)
    ap.add_argument("--phase", default="all", choices=["gen", "judge", "all"])
    args = ap.parse_args()
    names = [x for x in args.configs.split(",") if x]
    items = load_items(args.limit)
    os.makedirs(RES, exist_ok=True)
    if args.phase in ("gen", "all"):
        for name in names:
            gen_one(name, items, args.ctx_tokens, args.limit)
    if args.phase in ("judge", "all"):
        for name in names:
            judge_one(name)
        summarize(names, items)


if __name__ == "__main__":
    main()
