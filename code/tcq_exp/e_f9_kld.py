"""E-F9: token 级输出分布 KLD 对照 (量化模型 vs 全精度参考)。

服务的论点: "KLD 指标是否携带任务级信号" —— 最终产出一张 KLD–NLL–acc 三层对应表。

协议:
  语料   data/{domain}_eval64.jsonl 每域取前 --seqs 条; 每条取前 --ctx-tokens 个位置
  对齐   用 /tokenize 拿到 token id, 逐点提交 "prompt": ids[:i] + "n_predict": 1
         (server 不对数字数组加 BOS, 位置严格对齐; 不是拿整段 prompt 只取首位置)
  两侧   参考模型 (全精度) 与量化模型在同一 token 前缀上各自取 top-k
  指标   KLD(P_ref || Q_model) / JSD / top-1 一致率 / teacher-forcing NLL

llama.cpp fork 的实际能力 (源码核实, 见 METADATA["server_capability"]):
  * 无 --logits-all / 无全词表 logits 输出; 只有 /completion 的 n_probs (top-k logprobs)
  * n_probs 无上限, 返回 completion_probabilities[].top_logprobs (默认 post_sampling_probs
    =false 时字段是 logprob, 不是 prob; exp() 还原)
  * top-k 概率是 **全词表 softmax 的绝对值** (server-common.cpp get_token_probabilities
    对整条候选向量归一), 因此 tail 质量 = 1 - sum(top-k) 是有意义的桶
  * n_probs>0 且 post_sampling_probs=false 时强制关闭 backend sampling, 概率取原始
    logits, 不受 temperature/top_k 截断 -> n_predict=1 + temperature=0 可用于纯前向打分
  * "prompt" 支持 token id 数组; cache_prompt=true/false 都支持 (脚本默认 true,
    只复用前 i-1 个 token 的前缀, 与 false 数学等价, 但省掉 ~64x prefill)

用法:
    python e_f9_kld.py --dry-run
    python e_f9_kld.py
    python e_f9_kld.py --models iq2xxs,ud_q2kxl,nl_global --domains math,code,general

输出: results/e_f9_kld.json
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
LLAMA_DIR = os.path.abspath(os.path.join(HERE, "..", "llama.cpp"))
LLAMA_BIN = os.path.join(LLAMA_DIR, "build-tcmoe", "bin", "Release")
SERVER_EXE = os.path.join(LLAMA_BIN, "llama-server.exe")
MODEL_ROOT = r"E:\CUDA11.8\model"
DOMAINS = ("math", "code", "general")   # 与 tcq_lib.DOMAINS 相同
SPLIT = "eval64"                        # 与 tcq_lib.load_domain(dom, "eval64") 相同
DEFAULT_N_VOCAB = 151936                # Qwen3 词表; 仅当 server 不报告 n_vocab 时兜底

MODELS = {
    "iq2xxs": {
        "path": os.path.join(MODEL_ROOT, "baselines", "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"),
        "domain": None, "acc_tag": "iq2xxs", "alias": ("iq2xxs",),
    },
    "ud_q2kxl": {
        "path": os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
        "domain": None, "acc_tag": "ud_q2kxl", "alias": ("ud_q2kxl",),
    },
    "ud_q4kxl": {
        "path": os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
        "domain": None, "acc_tag": "ud_q4kxl", "alias": ("ud_q4kxl",),
    },
    "nl_global": {
        "path": os.path.abspath(os.path.join(HERE, "..", "tcmoe", "out", "model-tcmoe-q4k-nl.gguf")),
        "domain": "global", "acc_tag": "tcmoe_nl_global",
        "alias": ("tcmoe_nl_global", "nl_global"),
    },
}

# 参考模型优先级: 全精度 F16/F16 转换件 > 可用的最高精度 Q4
REF_CANDIDATES = [
    (os.path.join(MODEL_ROOT, "e20", "qwen36_f16.gguf"), "F16"),
    (os.path.join(MODEL_ROOT, "Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"), "Q4_K_M"),
    (os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"), "Q4_K_XL"),
]

# 量级估计 (1-token 前向 + 采样, CPU 16 线程, 仅用于 dry-run 提示, 不是协议的一部分)
EST_SEC_PER_POS = {"ref_full": 0.45, "model": 0.16}

SERVER_CAPABILITY = {
    "verified_by": "static source inspection on the local build-tcmoe fork",
    "sources": [
        "llama.cpp/tools/server/server-schema.cpp",
        "llama.cpp/tools/server/server-common.cpp",
        "llama.cpp/tools/server/server-context.cpp",
        "llama.cpp/tools/server/server-task.cpp",
        "llama.cpp/tools/server/README.md",
    ],
    "logits_all": False,
    "logits_all_note": "grep 'logits-all|logits_all|--logits' 在 tools/server 下无命中; "
                       "该 fork 没有返回全词表 logits 的选项, 只有 n_probs 的 top-k",
    "max_top_k": "无显式上限 (schema 只要求 >0); 实际上限是候选向量长度/词表大小, 默认 0 表示不返回",
    "request_field": "n_probs (别名 logprobs); OAI chat 侧要求 logprobs=true 且用 top_logprobs",
    "response_shape": {
        "path": "completion_probabilities[i]  (i = 第 i 个生成 token, 长度 n_predict)",
        "entry": {"id": "token id", "token": "str", "bytes": "list[int]",
                  "logprob": "ln p (post_sampling_probs=false)",
                  "top_logprobs": "[{id, token, bytes, logprob}]"},
        "note": "post_sampling_probs=true 时字段改为 prob/top_probs (经过采样链, 不是原始分布)",
    },
    "probability_semantics": "get_token_probabilities() 对整条候选向量做 softmax "
                             "(server-common.cpp:1664-1711, cum_sum 覆盖全部 cur), "
                             "返回的 top-k 概率是绝对概率; tail 桶 = 1 - sum(top-k) 可直接用于 KLD",
    "pre_sampling_logits": "n_probs>0 且 post_sampling_probs=false 时 need_pre_sample_logits "
                           "置位并关闭 backend sampling (server-context.cpp:2127), "
                           "概率来自原始 logits, 与 temperature/top_k 无关",
    "n_predict_1": "支持; 每个请求只产出 1 个位置的分布, 配合 token 前缀实现逐点 teacher forcing",
    "prompt_token_ids": "server-common.cpp tokenize_mixed(): \"prompt\": [12,34,...] 直接按 id 使用, "
                        "数字数组不加 BOS",
    "cache_prompt": "支持 true/false; 默认 true, 前缀复用与 false 数学等价",
    "tokenize_endpoint": "/tokenize (content/add_special=false/parse_special=true), 用于严格对齐 token",
    "context": "tcmoe 补丁: /completion 的 \"domain\" 字段选择专家补丁, 缺省 = 首个声明的域; "
               "非 tcmoe 模型忽略该字段 (server-context.cpp resolve_tcmoe_domain)",
    "runtime_probe": "未做 (机器上有 12h UD 评测队列在跑, 再起 server 会污染其耗时)",
}


# ---------------------------------------------------------------- 数据
def load_domain(domain, split=SPLIT):
    """等价于 tcq_lib.load_domain(domain, split): data/{domain}_{split}.jsonl"""
    p = os.path.join(DATA, f"{domain}_{split}.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- HTTP
def http_json(url, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url, payload, timeout=300):
    return http_json(url, payload, timeout)


def get_json(url, timeout=30):
    return http_json(url, None, timeout)


def port_alive(port, timeout=3):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status in (200, 503)
    except Exception:  # noqa: BLE001
        return False


def wait_health(port, proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server 退出, code={proc.returncode} (见 server log)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise SystemExit(f"llama-server 启动超时 ({timeout}s)")


def wait_port_free(port, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not port_alive(port):
            return True
        time.sleep(1)
    return False


def start_server(model_path, port, threads, n_ctx, log_path, timeout):
    """启动 llama-server; 可用环境变量覆盖 (缺省保持原行为):
      F9_BIN  server 可执行文件路径 (例如 CUDA 构建的 llama-server.exe)
      F9_NGL  GPU 卸载层数; 未设置时不传 -ngl (纯 CPU, 与既有 E-F9 结果一致)
    """
    if not wait_port_free(port):
        raise SystemExit(f"端口 {port} 仍被占用, 无法启动 llama-server")
    exe = os.environ.get("F9_BIN", "").strip() or SERVER_EXE
    ngl = os.environ.get("F9_NGL", "").strip()
    cmd = [exe, "-m", model_path, "--host", "127.0.0.1", "--port", str(port),
           "-c", str(n_ctx), "-np", "1", "-t", str(threads),
           "--no-webui", "--verbosity", "1"]
    if ngl:
        cmd += ["-ngl", ngl]
    log = open(log_path, "w", encoding="utf-8")
    log.write(" ".join(cmd) + "\n")
    log.flush()
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        wait_health(port, proc, timeout)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        log.close()
        raise
    return proc, log


def stop_server(proc, log):
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    finally:
        try:
            log.close()
        except Exception:  # noqa: BLE001
            pass


def server_info(port):
    info = {}
    try:
        r = get_json(f"http://127.0.0.1:{port}/v1/models", timeout=20)
        d = r.get("data") or []
        if d:
            meta = d[0].get("meta") or {}
            info["n_vocab"] = int(meta.get("n_vocab")) if meta.get("n_vocab") else None
            info["n_ctx_train"] = meta.get("n_ctx_train")
            info["ftype"] = meta.get("ftype")
    except Exception:  # noqa: BLE001
        pass
    if not info.get("n_vocab"):
        try:
            r = get_json(f"http://127.0.0.1:{port}/props", timeout=20)
            info["n_vocab"] = r.get("n_vocab") or \
                (r.get("default_generation_settings") or {}).get("n_vocab")
            if info["n_vocab"]:
                info["n_vocab"] = int(info["n_vocab"])
        except Exception:  # noqa: BLE001
            pass
    return info


def tokenize(port, text):
    r = post_json(f"http://127.0.0.1:{port}/tokenize",
                  {"content": text, "add_special": False, "parse_special": True}, timeout=60)
    return [int(t) for t in r.get("tokens", [])]


# ---------------------------------------------------------------- 分布
def _prob(logp):
    if logp is None:
        return 0.0
    if logp < -1e30:  # logarithm(0) -> lowest float, json 里是一个巨大负数
        return 0.0
    return math.exp(float(logp))


def extract_top(entry):
    """completion_probabilities[i] -> [(token_id, p)] 按 p 降序, 丢弃 p<=0"""
    out = []
    for it in entry.get("top_logprobs") or entry.get("top_probs") or []:
        v = it.get("logprob")
        if v is None:
            v = it.get("prob")
            p = float(v) if v is not None else 0.0
        else:
            p = _prob(v)
        if p > 0:
            out.append((int(it["id"]), p))
    out.sort(key=lambda x: -x[1])
    return out


def build_partition(p_top, q_top, n_vocab):
    """把两侧 top-k 拼到同一支撑上, 词表剩余质量作 tail 桶。

    对 P 不认识的 token (Q 的 top-k 中 P 没有) 用 P 的剩余质量均匀分摊,
    反之亦然; 分摊后剩余部分进 "__tail__"。这样两侧质量都严格归一到 1,
    KLD/JSD 在同一划分上可比。
    """
    P = dict(p_top)
    Q = dict(q_top)
    rest_p = max(0.0, 1.0 - math.fsum(P.values()))
    rest_q = max(0.0, 1.0 - math.fsum(Q.values()))
    n_unp = max(1, n_vocab - len(P))
    n_unq = max(1, n_vocab - len(Q))
    cp = rest_p / n_unp
    cq = rest_q / n_unq
    only_p = set(P) - set(Q)
    only_q = set(Q) - set(P)

    keys = set(P) | set(Q) | {"__tail__"}
    p = {}
    q = {}
    for k in keys - {"__tail__"}:
        p[k] = P[k] if k in P else cp
        q[k] = Q[k] if k in Q else cq
    p["__tail__"] = max(0.0, rest_p - cp * len(only_q))
    q["__tail__"] = max(0.0, rest_q - cq * len(only_p))
    return p, q


def kl(p, q):
    s = 0.0
    for k, pk in p.items():
        if pk <= 0.0:
            continue
        qk = q.get(k, 0.0)
        if qk <= 0.0:
            qk = 1e-12
        s += pk * math.log(pk / qk)
    return s


def jsd(p, q):
    m = {k: 0.5 * (p[k] + q.get(k, 0.0)) for k in p}
    for k in q:
        if k not in m:
            m[k] = 0.5 * q[k]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def position_metrics(p_top, q_top, target, n_vocab):
    P, Q = build_partition(p_top, q_top, n_vocab)
    rest_p = max(0.0, 1.0 - math.fsum(v for _, v in p_top))
    rest_q = max(0.0, 1.0 - math.fsum(v for _, v in q_top))
    p_gold = dict(p_top).get(target, rest_p / max(1, n_vocab - len(p_top)))
    q_gold = dict(q_top).get(target, rest_q / max(1, n_vocab - len(q_top)))
    return {
        "kld": kl(P, Q),
        "jsd": jsd(P, Q),
        "top1_agree": 1.0 if p_top and q_top and p_top[0][0] == q_top[0][0] else 0.0,
        "nll_ref": -math.log(p_gold) if p_gold > 0 else 20.0,
        "nll_model": -math.log(q_gold) if q_gold > 0 else 20.0,
        "cov_ref": 1.0 if target in dict(p_top) else 0.0,
        "cov_model": 1.0 if target in dict(q_top) else 0.0,
        "mass_ref": math.fsum(v for _, v in p_top),
        "mass_model": math.fsum(v for _, v in q_top),
        "nk_ref": len(p_top),
        "nk_model": len(q_top),
    }


def score_position(port, ids, target_idx, k, cache_prompt, domain, retries=3):
    """提交 ids[:target_idx] 作上下文, 请求 1 个 token 的 top-k 分布"""
    payload = {
        "prompt": list(ids[:target_idx]),
        "n_predict": 1,
        "n_probs": k,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1234,
        "ignore_eos": True,
        "cache_prompt": bool(cache_prompt),
        "stream": False,
    }
    if domain:
        payload["domain"] = domain
    last = None
    for _ in range(retries):
        try:
            r = post_json(f"http://127.0.0.1:{port}/completion", payload, timeout=300)
            cp = r.get("completion_probabilities") or []
            if not cp:
                raise RuntimeError(f"响应无 completion_probabilities: {str(r)[:200]}")
            return extract_top(cp[0])
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"打分请求失败: {last}")


# ---------------------------------------------------------------- 汇总
def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def _median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    return float(ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2]))


def summarize(rows, per_seq):
    """rows: 每个位置的 metrics 列表"""
    keys = ("kld", "jsd", "top1_agree", "nll_ref", "nll_model",
            "cov_ref", "cov_model", "mass_ref", "mass_model")
    out = {"n_pos": len(rows), "per_seq_mean_kld": per_seq}
    for k in keys:
        out["mean_" + k] = _mean([r[k] for r in rows])
    out["median_kld"] = _median([r["kld"] for r in rows])
    out["std_kld"] = _std([r["kld"] for r in rows])
    out["mean_top_k_returned"] = _mean(
        [0.5 * (r["nk_ref"] + r["nk_model"]) for r in rows])
    return out


def read_e17_acc():
    """e17_summary.json -> {(tag, task): acc}"""
    p = os.path.join(RES, "e17_summary.json")
    if not os.path.exists(p):
        return {}, None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {(r["tag"], r["task"]): float(r["acc"]) for r in data}, p


def read_existing_nll(aliases):
    """从 pareto.json / e21_nll_v1.json 里找同名配置的现成 NLL"""
    for fn in ("pareto.json", "e21_nll_v1.json"):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        cfg = data.get("configs")
        if isinstance(cfg, list):  # pareto.json: [ {...name, nll} ]
            for c in cfg:
                if isinstance(c, dict) and c.get("name") in aliases \
                        and isinstance(c.get("nll"), dict):
                    return {"source": fn, "config": c["name"], "per_domain": c["nll"],
                            "mean": c.get("nll_mean")}
        elif isinstance(cfg, dict):  # e21_nll_v1.json: {name: {...}}
            for name in aliases:
                rec = cfg.get(name)
                if isinstance(rec, dict) and isinstance(rec.get("nll"), dict):
                    return {"source": fn, "config": name, "per_domain": rec["nll"],
                            "mean": _mean(list(rec["nll"].values()))}
        nll = data.get("nll") if isinstance(data.get("nll"), dict) else None
        if nll:
            for name in aliases:
                if name in nll and isinstance(nll[name], dict):
                    return {"source": fn, "config": name, "per_domain": nll[name],
                            "mean": _mean(list(nll[name].values()))}
    return None


def git_commit(path):
    try:
        r = subprocess.run(["git", "-C", path, "log", "-1", "--format=%H %ci"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or f"unknown (exit {r.returncode})"
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


# ---------------------------------------------------------------- 主流程
def _guess_prec(path):
    b = os.path.basename(path).lower()
    if "bf16" in b:
        return "BF16"
    if "f16" in b or "fp16" in b:
        return "F16"
    if "q8" in b:
        return "Q8_0"
    if "q6_k" in b:
        return "Q6_K"
    if "q5_k" in b:
        return "Q5_K"
    if "q4_k_m" in b:
        return "Q4_K_M"
    if "q4_k_xl" in b or "q4_k" in b:
        return "Q4_K"
    return "custom"


def pick_ref(arg):
    if arg and arg != "auto":
        return os.path.abspath(arg), _guess_prec(arg), None
    for path, prec in REF_CANDIDATES:
        if os.path.exists(path):
            return path, prec, None
    return None, None, "找不到全精度参考 GGUF (试过: " + \
        ", ".join(os.path.basename(p) for p, _ in REF_CANDIDATES) + ")"


def plan(models, domains, seqs, ctx_tokens):
    """dry-run / 正式跑共用: 每个 (模型, 域) 的计划请求数"""
    n_req = []
    for tag in models:
        for dom in domains:
            n_seq = min(seqs, len(load_domain(dom)))
            n_req.append({"model": tag, "domain": dom, "seqs": n_seq,
                          "positions_per_seq": ctx_tokens,
                          "requests": n_seq * ctx_tokens})
    return n_req


def run_one(tag, spec, ref_tag, ref_data, domains, seqs, ctx_tokens, k, port,
            threads, cache_prompt, n_ctx, timeout):
    log_path = os.path.join(RES, f"e_f9_{tag}.server.log")
    out = {"file": spec["path"], "domain_field": spec["domain"],
           "port": port, "server_log": os.path.basename(log_path),
           "domains": {}, "vocab_check": {}}
    proc, log = None, None
    try:
        proc, log = start_server(spec["path"], port, threads, n_ctx, log_path, timeout)
        info = server_info(port)
        n_vocab = info.get("n_vocab") or DEFAULT_N_VOCAB
        out["n_vocab"] = n_vocab
        out["n_vocab_reported"] = bool(info.get("n_vocab"))
        out["server_meta"] = info

        # token 对齐: 与参考模型必须逐 id 相同
        first_text = load_domain(domains[0])[0]["text"]
        first_ids = tokenize(port, first_text)
        same = first_ids == ref_data["first_ids"]
        out["vocab_check"] = {
            "first_text_token_ids_equal_ref": same,
            "n_vocab_equal_ref": n_vocab == ref_data["n_vocab"],
            "ref_n_vocab": ref_data["n_vocab"],
            "self_n_vocab": n_vocab,
        }
        if not same:
            raise SystemExit(f"[{tag}] tokenizer 与参考不一致, KLD 不可比: "
                             f"{first_ids[:8]} vs {ref_data['first_ids'][:8]}")

        for dom in domains:
            ds = load_domain(dom)[:seqs]
            ids_all = [tokenize(port, r["text"]) for r in ds]
            for j, ids in enumerate(ids_all):
                if ids != ref_data["tokens"][dom][j]:
                    raise SystemExit(f"[{tag}/{dom}] 第 {j} 条 token 与参考不一致")
            metrics, per_seq = [], []
            for j, ids in enumerate(ids_all):
                npos = min(ctx_tokens, len(ids) - 1)
                seq_klds = []
                for i in range(1, npos + 1):
                    ref_top = ref_data["top"][dom][j][i - 1]
                    model_top = score_position(port, ids, i, k, cache_prompt,
                                               spec["domain"])
                    m = position_metrics(ref_top, model_top, ids[i], n_vocab)
                    metrics.append(m)
                    seq_klds.append(m["kld"])
                    if (len(metrics) % 64) == 0:
                        print(f"  [{tag}/{dom}] {len(metrics)} pos, "
                              f"mean KLD {_mean([x['kld'] for x in metrics]):.4f}",
                              flush=True)
                m_seq = _mean(seq_klds)
                per_seq.append(m_seq if m_seq is not None else 0.0)
                print(f"  [{tag}/{dom}] seq {j + 1}/{len(ids_all)} n_pos={npos} "
                      f"KLD={per_seq[-1]:.4f}", flush=True)
            out["domains"][dom] = summarize(metrics, per_seq)
        out["mean_kld_over_domains"] = _mean(
            [v["mean_kld"] for v in out["domains"].values()])
        return out
    finally:
        stop_server(proc, log)


def run_ref(ref_path, ref_tag, domains, seqs, ctx_tokens, k, port, threads,
            cache_prompt, n_ctx, timeout):
    """参考侧: 一次性把所有域/序列/位置的 top-k 分布缓存下来"""
    log_path = os.path.join(RES, "e_f9_ref.server.log")
    data = {"tag": ref_tag, "file": ref_path, "tokens": {}, "top": {}, "first_ids": None,
            "n_vocab": DEFAULT_N_VOCAB, "server_meta": {}, "n_pos": 0}
    proc, log = None, None
    try:
        proc, log = start_server(ref_path, port, threads, n_ctx, log_path, timeout)
        info = server_info(port)
        data["n_vocab"] = info.get("n_vocab") or DEFAULT_N_VOCAB
        data["server_meta"] = info
        for dom in domains:
            ds = load_domain(dom)[:seqs]
            ids_all = [tokenize(port, r["text"]) for r in ds]
            data["tokens"][dom] = ids_all
            if data["first_ids"] is None:
                data["first_ids"] = ids_all[0]
            data["top"][dom] = []
            for j, ids in enumerate(ids_all):
                npos = min(ctx_tokens, len(ids) - 1)
                tops = []
                for i in range(1, npos + 1):
                    tops.append(score_position(port, ids, i, k, cache_prompt, None))
                    data["n_pos"] += 1
                    if (data["n_pos"] % 64) == 0:
                        print(f"  [ref/{dom}] {data['n_pos']} pos", flush=True)
                data["top"][dom].append(tops)
                print(f"  [ref/{dom}] seq {j + 1}/{len(ids_all)} n_pos={npos}", flush=True)
        return data
    finally:
        stop_server(proc, log)


def build_three_layer(results, acc, acc_src, ref_tag):
    rows = []
    for tag, rec in results.items():
        aliases = MODELS[tag]["alias"]
        existing = read_existing_nll(aliases)
        a_tag = MODELS[tag]["acc_tag"]
        for dom in rec["domains"]:
            d = rec["domains"][dom]
            rows.append({
                "model": tag, "domain": dom,
                "kld": d.get("mean_kld"), "jsd": d.get("mean_jsd"),
                "top1_agree": d.get("mean_top1_agree"),
                "nll_ref": d.get("mean_nll_ref"),
                "nll_model_teacher_forcing": d.get("mean_nll_model"),
                "nll_existing": (existing or {}).get("per_domain", {}).get(dom),
                "nll_existing_source": (existing or {}).get("source"),
                "acc_gsm8k": acc.get((a_tag, "gsm8k")),
                "acc_humaneval": acc.get((a_tag, "humaneval")),
                "acc_source": os.path.basename(acc_src) if acc_src else None,
            })
        vals = [v for v in (rec["domains"][d].get("mean_jsd") for d in rec["domains"])
                if v is not None]
        rows.append({
            "model": tag, "domain": "ALL",
            "kld": rec.get("mean_kld_over_domains"),
            "jsd": _mean(vals),
            "top1_agree": _mean([v for v in
                                 (x.get("mean_top1_agree") for x in rec["domains"].values())
                                 if v is not None]),
            "nll_ref": _mean([v for v in
                              (x.get("mean_nll_ref") for x in rec["domains"].values())
                              if v is not None]),
            "nll_model_teacher_forcing": _mean(
                [v for v in (x.get("mean_nll_model") for x in rec["domains"].values())
                 if v is not None]),
            "nll_existing": (existing or {}).get("mean"),
            "nll_existing_source": (existing or {}).get("source"),
            "acc_gsm8k": acc.get((a_tag, "gsm8k")),
            "acc_humaneval": acc.get((a_tag, "humaneval")),
            "acc_source": os.path.basename(acc_src) if acc_src else None,
        })
    return rows


def write_out(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description="E-F9 token 级输出分布 KLD 对照")
    ap.add_argument("--ref", default="auto", help="参考 GGUF; auto = 自动挑全精度件")
    ap.add_argument("--models", default="iq2xxs,ud_q2kxl,nl_global")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--ctx-tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--port", type=int, default=18095)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-cache-prompt", action="store_true",
                    help="每个位置重算整个前缀 (慢 ~64x, 结果等价)")
    ap.add_argument("--server-timeout", type=int, default=600, help="server 加载超时(s)")
    ap.add_argument("--out", default="e_f9_kld.json",
                    help="results/ 下的输出文件名 (默认 e_f9_kld.json)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    for d in domains:
        if d not in DOMAINS:
            raise SystemExit(f"未知域 {d}; 可选 {DOMAINS}")
    tags = [t.strip() for t in args.models.split(",") if t.strip()]
    for t in tags:
        if t not in MODELS:
            raise SystemExit(f"未知模型 {t}; 可选 {tuple(MODELS)}")

    ref_path, ref_prec, ref_err = pick_ref(args.ref)
    if ref_path is None:
        raise SystemExit(ref_err)
    if not os.path.exists(ref_path):
        raise SystemExit(f"参考 GGUF 不存在: {ref_path}")

    # 缺文件的量化侧: 跳过 (ud_q2kxl 允许不存在)
    usable = []
    for t in tags:
        if os.path.exists(MODELS[t]["path"]):
            usable.append(t)
        else:
            print(f"[warn] 跳过 {t}: 文件不存在 {MODELS[t]['path']}", flush=True)
    if not usable:
        raise SystemExit("没有任何可用的量化模型文件")

    cache_prompt = not args.no_cache_prompt
    n_ctx = max(128, args.ctx_tokens + 8)
    out_path = os.path.join(RES, args.out) if not os.path.isabs(args.out) else args.out
    acc, acc_src = read_e17_acc()

    if args.dry_run:
        print("=" * 78)
        print(f"[dry-run] 参考: {ref_prec}  {ref_path}")
        if ref_prec not in ("BF16", "F16"):
            print("          注意: 参考不是 BF16, 结论按相对 KLD 解读")
        print(f"[dry-run] 量化: {', '.join(usable)}")
        print(f"[dry-run] 域={domains} seqs={args.seqs} ctx_tokens={args.ctx_tokens} "
              f"k={args.k} port={args.port} threads={args.threads} "
              f"cache_prompt={cache_prompt}")
        print(f"[dry-run] 输出: {out_path}")
        print("-" * 78)
        print(f"{'model':10s} {'domain':10s} {'seqs':>5s} {'pos/seq':>8s} {'requests':>9s}")
        total = 0
        est_s = 0.0
        for r in plan(usable, domains, args.seqs, args.ctx_tokens):
            total += r["requests"]
            rate = EST_SEC_PER_POS["model"]
            est_s += r["requests"] * rate
            print(f"{r['model']:10s} {r['domain']:10s} {r['seqs']:5d} "
                  f"{r['positions_per_seq']:8d} {r['requests']:9d}")
        ref_req = sum(min(args.seqs, len(load_domain(d))) * args.ctx_tokens for d in domains)
        est_s += ref_req * EST_SEC_PER_POS["ref_full"]
        print("-" * 78)
        print(f"量化侧请求 {total}  参考侧请求 {ref_req}  合计 {total + ref_req}")
        print(f"量级估计 {est_s / 60:.1f} 分钟 (仅供参考; 机器上 12h UD 队列占用 16 线程, "
              f"实际会明显更慢)")
        if os.path.getsize(ref_path) / 2 ** 30 > 60:
            print(f"注意: 参考 GGUF {os.path.getsize(ref_path) / 2 ** 30:.1f} GiB, "
                  f"本机 128 GiB RAM 且已有评测队列在跑, 起 server 前确认内存余量")
        print("[dry-run] 不加载任何模型, 未启动 server。")
        return

    if port_alive(args.port):
        raise SystemExit(f"端口 {args.port} 已被占用 (可能是别的 llama-server), 换 --port")
    server_exe = os.environ.get("F9_BIN", "").strip() or SERVER_EXE
    if not os.path.exists(server_exe):
        raise SystemExit(f"找不到 llama-server: {server_exe}")

    t_start = time.time()
    out = {
        "protocol": {
            "experiment": "E-F9 token-level output-distribution KLD (quantized vs full-precision)",
            "claim": "KLD 是否携带任务级信号: KLD-NLL-acc 三层对应",
            "corpus": f"data/{{domain}}_{SPLIT}.jsonl, 每域前 {args.seqs} 条, "
                      f"每条前 {args.ctx_tokens} 个位置",
            "alignment": "逐位置提交 prompt=ids[:i] + n_predict=1, 取该位置下一个 token 的 top-k",
            "metrics": ["KLD(P_ref||Q_model)", "JSD(nats)", "top1_agree",
                        "NLL_ref", "NLL_model(teacher-forcing, top-k 覆盖或 tail 分摊)"],
            "kld_method": "两侧 top-k 并到同一支撑; 对侧未出现的 token 用该侧剩余质量均匀分摊, "
                          "剩余进 __tail__ 桶; p=0 项跳过, q<=0 时 clamp 1e-12",
            "mass_sanity": "top-k 概率应为全词表 softmax 的绝对值 (mass<1, 随 k 增大上升); "
                           "若 mean_topk_mass 恒等于 1.000, 说明被候选内归一, 本次结果作废",
            "k": args.k, "domains": domains, "seqs": args.seqs,
            "ctx_tokens": args.ctx_tokens, "threads": args.threads,
            "cache_prompt": cache_prompt, "n_ctx": n_ctx, "port": args.port,
            "backend": ("CPU only (build-tcmoe, type45 tcmoe patches); 无 CUDA 设备"
                        if not (os.environ.get("F9_NGL", "").strip())
                        else "CUDA hybrid offload (F9_BIN/F9_NGL)"),
            "server_bin": os.environ.get("F9_BIN", "").strip() or SERVER_EXE,
            "ngl": os.environ.get("F9_NGL", "").strip() or None,
            "llama_cpp_commit": git_commit(LLAMA_DIR),
            "elapsed_s": 0,
        },
        "server_capability": SERVER_CAPABILITY,
        "ref": {
            "file": ref_path, "precision": ref_prec,
            "size_gib": round(os.path.getsize(ref_path) / 2 ** 30, 2),
            "is_bf16": ref_prec == "BF16",
            "note": ("参考是 BF16 原地权重" if ref_prec == "BF16" else
                     "参考非 BF16 原地权重 (用全精度转换件 F16 或最高精度 Q4_K_M); "
                     "绝对 KLD 会略偏小/偏大, 结论按相对 KLD 解读"),
        },
        "models": {t: {"file": MODELS[t]["path"], "domain": MODELS[t]["domain"],
                       "acc_tag": MODELS[t]["acc_tag"]} for t in usable},
        "results": {},
        "three_layer": [],
    }

    try:
        print(f"[ref] 启动 {ref_prec} {os.path.basename(ref_path)}", flush=True)
        ref_data = run_ref(ref_path, ref_prec, domains, args.seqs, args.ctx_tokens,
                           args.k, args.port, args.threads, cache_prompt, n_ctx,
                           args.server_timeout)
        out["ref"].update({"n_vocab": ref_data["n_vocab"],
                           "n_pos": ref_data["n_pos"],
                           "server_meta": ref_data["server_meta"]})
        print(f"[ref] 完成 {ref_data['n_pos']} 个位置", flush=True)

        for t in usable:
            print(f"[{t}] 启动 {os.path.basename(MODELS[t]['path'])}", flush=True)
            out["results"][t] = run_one(t, MODELS[t], ref_prec, ref_data, domains,
                                        args.seqs, args.ctx_tokens, args.k, args.port,
                                        args.threads, cache_prompt, n_ctx,
                                        args.server_timeout)
            masses = [(d.get("mean_mass_ref") or 0.0) for d in out["results"][t]["domains"].values()]
            masses += [(d.get("mean_mass_model") or 0.0) for d in out["results"][t]["domains"].values()]
            if masses and max(masses) > 1.005:
                print(f"[warn] {t}: mean top-k mass={max(masses):.3f} > 1.005, "
                      f"概率可能被候选内归一, 该模型 KLD 不可信", flush=True)
            out["three_layer"] = build_three_layer(out["results"], acc, acc_src, ref_prec)
            out["protocol"]["elapsed_s"] = round(time.time() - t_start, 1)
            write_out(out, out_path)
            print(f"[{t}] mean KLD {out['results'][t]['mean_kld_over_domains']:.4f} "
                  f"-> {out_path}", flush=True)
    finally:
        out["protocol"]["elapsed_s"] = round(time.time() - t_start, 1)
        if out["results"]:
            out["three_layer"] = build_three_layer(out["results"], acc, acc_src, ref_prec)
            write_out(out, out_path)

    print("-" * 78)
    print(f"{'model':10s} {'domain':10s} {'KLD':>8s} {'JSD':>8s} {'top1':>6s} "
          f"{'NLL_ref':>8s} {'NLL_q':>8s} {'gsm8k':>6s} {'HE':>6s}")
    for r in out["three_layer"]:
        print(f"{r['model']:10s} {r['domain']:10s} "
              f"{(r['kld'] if r['kld'] is not None else float('nan')):8.4f} "
              f"{(r['jsd'] or 0):8.4f} {(r['top1_agree'] or 0):6.3f} "
              f"{(r['nll_ref'] or 0):8.4f} {(r['nll_model_teacher_forcing'] or 0):8.4f} "
              f"{(r['acc_gsm8k'] or 0):6.3f} {(r['acc_humaneval'] or 0):6.3f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
