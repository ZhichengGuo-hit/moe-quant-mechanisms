#!/usr/bin/env python
"""E-H5: 长上下文评测（NIAH 长度扫描 + RULER 同构合成精简集，CUDA 三臂）

臂（用户锁定方案 D）:
  iq2xxs   baselines/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf   -ngl 99
  ud_q2kxl ud/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf          -ngl 99
  ud_q4km  Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 20（超显存，部分卸载）
同一 binary（llama-upstream build-cuda b8746），同一 -ngl 口径内比较；上下文 ≤ 64k，远低于原生 262k。

任务:
  NIAH     4k/16k/64k × 每档 10 题（8 单针 + 2 三针变体），针深度均匀分布
  RULER*   8k 定长 × 3 类 × 50 题：变量追踪 / 常见词提取 / 多跳方位
           （原版 RULER 数据不在本地，这里是与官方任务同构的合成替代，条目在臂间完全相同；
            只能作臂间相对比较，不得当作官方 RULER 分数）

主指标: 检索/任务准确率 × 长度；附带输出长度与 4-gram 最大重复率。
判定: 与 iq2xxs 的配对 bootstrap（条目级，90%/95% CI）。

用法:
  python e_h5_eval.py build            # 生成条目（需起一次 server 做 tokenize）
  python e_h5_eval.py run --arms iq2xxs,ud_q2kxl,ud_q4km
  python e_h5_eval.py report
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-server.exe"
BIN_CPU = r"E:\CUDA11.8\train\llama.cpp\build-tcmoe\bin\Release\llama-server.exe"
TCMOE = r"E:\CUDA11.8\train\tcmoe\out\model-tcmoe-q4k-nl.gguf"
RES = os.path.join(HERE, "results")
WORK = os.path.join(HERE, "h5_work")
ITEMS = os.path.join(WORK, "h5_items_smoke.json" if os.environ.get("H5_SMOKE") else "h5_items.json")
LOG = os.path.join(RES, "h5_logs")
# max_len: 该臂只跑不超过此长度的条目（CPU 臂按用户方案 D 限 ≤8k）
ARMS = {
    "iq2xxs": {"path": r"E:\CUDA11.8\model\baselines\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
               "ngl": 99, "bin": BIN, "max_len": 65536, "ctx": 65536 + 4096, "domain": None},
    "ud_q2kxl": {"path": r"E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",
                 "ngl": 99, "bin": BIN, "max_len": 65536, "ctx": 65536 + 4096, "domain": None},
    "ud_q4km": {"path": r"E:\CUDA11.8\model\Qwen3.6-35B-A3B\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                "ngl": 20, "bin": BIN, "max_len": 65536, "ctx": 65536 + 4096, "domain": None},
    # CPU 三臂同一文件、同一 binary，仅靠请求级 domain 切换（base = 关补丁退化对照）
    "tcq_nl_global": {"path": TCMOE, "ngl": 0, "bin": BIN_CPU, "max_len": 8192,
                      "ctx": 8192 + 2048, "domain": "global"},
    "tcq_nl_math": {"path": TCMOE, "ngl": 0, "bin": BIN_CPU, "max_len": 8192,
                    "ctx": 8192 + 2048, "domain": "math"},
    "base2": {"path": TCMOE, "ngl": 0, "bin": BIN_CPU, "max_len": 8192,
              "ctx": 8192 + 2048, "domain": "base"},
}
CUDA_ARMS = ["iq2xxs", "ud_q2kxl", "ud_q4km"]
CPU_ARMS = ["tcq_nl_global", "tcq_nl_math", "base2"]
LENS = [4096, 16384, 65536]
CTX = 65536 + 4096  # 64k 条目实测 prompt 达 ~68070 token，512 余量会 400
SEED = 20261108
BOS_PROSE = [os.path.join(HERE, "data", "prose_test.jsonl"),
             os.path.join(HERE, "data", "general_eval.jsonl")]


def log(msg):
    print(msg, flush=True)
    os.makedirs(LOG, exist_ok=True)
    with open(os.path.join(LOG, "h5_eval.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------- server ----------------
class Server:
    def __init__(self, arm, port):
        self.arm = arm
        self.port = port
        self.proc = None

    def start(self, timeout=600):
        cfg = ARMS[self.arm]
        os.makedirs(LOG, exist_ok=True)
        lf = open(os.path.join(LOG, "server_%s.log" % self.arm), "w", encoding="utf-8")
        cmd = [cfg["bin"], "-m", cfg["path"], "-c", str(cfg["ctx"]),
               "-ngl", str(cfg["ngl"]), "-t", "16",
               "--host", "127.0.0.1", "--port", str(self.port), "-np", "1"]
        log("$ " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % self.port, timeout=5) as r:
                    if json.loads(r.read().decode()).get("status") == "ok":
                        log("[server] up %.0fs" % (time.time() - t0))
                        return True
            except Exception:
                time.sleep(5)
        return False

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
            self.proc = None

    def post(self, path, payload, timeout=1200):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path),
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def tokenize(self, text):
        return self.post("/tokenize", {"content": text})["tokens"]

    def chat(self, system, user, max_tokens=256):
        payload = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0, "top_k": 1, "max_tokens": max_tokens,
            "cache_prompt": False,
            # 关闭思维通道（与 e_g1_eval.py 同口径），否则 content 为空
            "chat_template_kwargs": {"enable_thinking": False}}
        dom = ARMS[self.arm].get("domain")
        if dom:
            payload["domain"] = dom
        d = self.post("/v1/chat/completions", payload)
        msg = d["choices"][0]["message"]
        ch = msg.get("content") or ""
        if not ch:
            ch = msg.get("reasoning_content") or ""
        return ch, int(d.get("usage", {}).get("prompt_tokens") or 0)


# ---------------- 条目生成 ----------------
def prose():
    parts = []
    for p in BOS_PROSE:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts.append(json.loads(line)["text"].strip())
    return parts


def haystack_tokens(srv_tok, target, corpus, rng):
    """逐句累积 token 数到 ~target（每句单独 tokenize，避免 O(n²) 往返）"""
    out, n = [], 0
    while n < target:
        s = corpus[rng.randrange(len(corpus))]
        out.append(s)
        n += len(srv_tok(s))
    return "\n".join(out), n


def insert_at(text, needle, frac, tok_fn):
    """按 token 比例插到最近的句边界"""
    sent = [s for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip() and s != "\n"]
    total = sum(len(tok_fn(s)) for s in sent) or 1
    k = int(total * frac)
    acc = 0
    for i, s in enumerate(sent):
        acc += len(tok_fn(s))
        if acc >= k:
            sent.insert(i + 1, needle)
            return "\n".join(sent)
    return text + "\n" + needle


def niah_items(tok_fn, rng):
    corpus = prose()
    smoke = bool(os.environ.get("H5_SMOKE"))
    items = []
    for L in ([4096] if smoke else LENS):
        for i in range(1 if smoke else 10):
            multi = i >= 8
            hay, ntok = haystack_tokens(tok_fn, L, corpus, rng)
            vals = ["ZKQ%d-%04d" % (i, rng.randrange(10000)) for _ in range(3 if multi else 1)]
            if multi:
                depths = [0.2, 0.5, 0.8]
                txt = hay
                for v, d in zip(vals, depths):
                    txt = insert_at(txt, "The secret passphrase for compartment %s is %s." % (v[3:], v), d, tok_fn)
                q = ("Three passphrase records are hidden in the document above. "
                     "List all three passphrases, comma separated, in the order they appear.")
                gold = list(vals)
            else:
                d = rng.uniform(0.1, 0.9)
                txt = insert_at(hay, "The secret passphrase for compartment %s is %s." % (vals[0][3:], vals[0]), d, tok_fn)
                q = "What is the secret passphrase hidden in the document above? Answer with the passphrase only."
                gold = list(vals)
            assert all(v in txt for v in vals)
            items.append({"task": "niah", "len": L, "idx": i, "multi": multi,
                          "prompt": txt, "question": q, "gold": gold,
                          "gold_mode": "all"})
            log("[item] niah L=%d i=%d multi=%s" % (L, i, multi))
    return items


def pad_to(tok_fn, body, target, corpus, rng):
    """把正文随机插入到 ~target token 的填充文本中部"""
    hay, _ = haystack_tokens(tok_fn, max(512, target - len(tok_fn(body))), corpus, rng)
    parts = hay.split("\n")
    frac = rng.uniform(0.3, 0.7)
    k = int(len(parts) * frac)
    return "\n".join(parts[:k] + [body] + parts[k:])


def ruler_items(tok_fn, rng):
    corpus = prose()
    smoke = bool(os.environ.get("H5_SMOKE"))
    N = 1 if smoke else 50
    items = []
    T = 8192
    # 1) 变量追踪
    for i in range(N):
        n = 10
        vals = [rng.randrange(2, 99)]
        lines = ["Initial value: X0 = %d." % vals[0]]
        for j in range(1, n):
            op = rng.choice(["+", "-", "*"])
            c = rng.randrange(1, 9) if op != "*" else rng.randrange(2, 4)
            vals.append(vals[-1] + c if op == "+" else (vals[-1] - c if op == "-" else vals[-1] * c))
            lines.append("X%d = X%d %s %d." % (j, j - 1, op, c))
        body = "\n".join(lines)
        tgt = rng.randrange(0, n)
        q = ("In the document above there is a block of variable definitions X0..X%d. "
             "What is the final integer value of X%d? Answer with the integer only." % (n - 1, tgt))
        items.append({"task": "vt", "len": T, "idx": i, "prompt": pad_to(tok_fn, body, T, corpus, rng),
                      "question": q, "gold": [str(vals[tgt])], "gold_mode": "num"})
    # 2) 常见词提取
    words = ["blorf", "quane", "zibet", "marl", "daxil", "sunep", "torix", "vundy", "klast", "pemro",
             "gresk", "holt", "avora", "nixle", "durmo"]
    for i in range(N):
        cnt = {w: rng.choice([1, 1, 2, 3, 4, 6, 7, 8]) for w in words}
        body = "Word frequency list: " + ", ".join("%s(%d)" % (w, c) for w, c in cnt.items()) + "."
        th = rng.choice([4, 5, 6])
        gold = [w for w, c in cnt.items() if c >= th]
        q = ("In the word frequency list above, which words appear at least %d times? "
             "List them comma separated." % th)
        items.append({"task": "cw", "len": T, "idx": i, "prompt": pad_to(tok_fn, body, T, corpus, rng),
                      "question": q, "gold": gold, "gold_mode": "set", "cand": words})
    # 3) 多跳方位（3 跳向量合成，8 向映射；无法映射的合成丢弃重抽）
    vec = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
    back = {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west",
            (1, 1): "northeast", (-1, 1): "northwest", (1, -1): "southeast", (-1, -1): "southwest"}
    for i in range(N):
        places = ["PX%d" % j for j in range(6)]
        for _try in range(200):
            facts, cur = [], places[0]
            ax = ay = 0
            ok = True
            for j in range(1, 4):
                d = rng.choice(list(vec))
                facts.append("%s is %s of %s." % (cur, d, places[j]))
                ax += vec[d][0]
                ay += vec[d][1]
                cur = places[j]
            # (ax, ay) 是 places[0] 相对 places[3] 的位移，题目问的是 places[0] -> cur 的方向，需取反
            acc = back.get((-ax, -ay))
            if acc:
                break
        else:
            acc = "north"
        body = " ".join(facts)
        q = "According to the facts above, what is the direction from %s to %s? Answer with one word." % (places[0], cur)
        items.append({"task": "mh", "len": T, "idx": i, "prompt": pad_to(tok_fn, body, 8192, corpus, rng),
                      "question": q, "gold": [acc], "gold_mode": "str"})
    return items


def build_items(port):
    srv = Server("iq2xxs", port)
    if not srv.start():
        raise SystemExit("server failed")
    try:
        rng = random.Random(SEED)
        tok = srv.tokenize
        items = niah_items(tok, rng) + ruler_items(tok, rng)
        os.makedirs(WORK, exist_ok=True)
        json.dump(items, open(ITEMS, "w"), ensure_ascii=False)
        lens = {}
        for it in items:
            lens.setdefault(it["task"], []).append(len(tok(it["prompt"])))
        log("[items] %d saved; prompt tokens avg=%s" % (
            len(items), {k: int(np.mean(v)) for k, v in lens.items()}))
    finally:
        srv.stop()


# ---------------- 推理 ----------------
def score(item, out):
    it = item
    lo = out.lower()
    if it["gold_mode"] == "num":
        m = re.findall(r"-?\d+", out)
        return int(m[0]) == int(it["gold"][0]) if m else False
    if it["gold_mode"] == "set":
        # 从候选词表中检出模型列举的词，要求与 gold 完全一致（召回+精确）
        pred = {w for w in it.get("cand", []) if re.search(r"\b%s\b" % w, lo)}
        return pred == set(g.lower() for g in it["gold"])
    return all(g.lower() in lo for g in it["gold"])


def repeat_max(text, n=4):
    w = text.split()
    if len(w) < n:
        return 0
    g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    from collections import Counter
    return max(Counter(g).values()) if g else 0


def run_arm(arm, port, limit=0):
    out_p = os.path.join(RES, "e_h5_%s%s.jsonl" % (arm, "_smoke" if os.environ.get("H5_SMOKE") else ""))
    done = set()
    if os.path.exists(out_p):
        for line in open(out_p, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                done.add((d["task"], d["len"], d["idx"]))
    cfg = ARMS[arm]
    items = json.load(open(ITEMS, encoding="utf-8"))
    # 超长条目跳过（CPU 臂 ≤8k），limit 限制每类条目数（用于降成本的子集臂）
    items = [it for it in items if it["len"] <= cfg["max_len"] and (not limit or it["idx"] < limit)]
    srv = Server(arm, port)
    if not srv.start():
        log("[%s] server failed" % arm)
        return False
    t0 = time.time()
    try:
        with open(out_p, "a", encoding="utf-8") as f:
            n_new = 0
            for it in items:
                key = (it["task"], it["len"], it["idx"])
                if key in done:
                    continue
                try:
                    out, ptok = srv.chat("You are a precise assistant. Follow the requested answer format exactly.",
                                         it["prompt"] + "\n\n" + it["question"], 256)
                except Exception as e:
                    log("[%s] item %s error %s" % (arm, key, str(e)[:100]))
                    continue
                rec = {"arm": arm, "task": it["task"], "len": it["len"], "idx": it["idx"],
                       "multi": it.get("multi", False), "ok": bool(score(it, out)),
                       "out": out[:800], "out_tokens": len(out.split()),
                       "prompt_tokens": ptok, "repeat4": repeat_max(out),
                       "dur_s": round(time.time() - t0, 1)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_new += 1
                if n_new % 10 == 0:
                    log("[%s] %d/%d done %.0f min" % (arm, len(done) + n_new, len(items), (time.time() - t0) / 60))
    finally:
        srv.stop()
    log("[%s] FINISHED %d new, %.0f min" % (arm, n_new, (time.time() - t0) / 60))
    return True


# ---------------- 汇总 ----------------
def report():
    suffix = "_smoke" if os.environ.get("H5_SMOKE") else ""
    arms = [a for a in ARMS if os.path.exists(os.path.join(RES, "e_h5_%s%s.jsonl" % (a, suffix)))]
    data = {a: {} for a in arms}
    for a in arms:
        for line in open(os.path.join(RES, "e_h5_%s%s.jsonl" % (a, suffix)), encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                data[a][(d["task"], d["len"], d["idx"])] = d
    out = {"arms": arms, "tasks": {}}
    rng = np.random.default_rng(SEED)
    for task in ("niah", "vt", "cw", "mh"):
        lens = sorted({k[1] for a in arms for k in data[a] if k[0] == task})
        for L in lens:
            cells = {}
            for a in arms:
                ks = [k for k in data[a] if k[0] == task and k[1] == L]
                if not ks:
                    continue
                oks = [data[a][k]["ok"] for k in ks]
                cells[a] = {"n": len(ks), "acc": float(np.mean(oks)),
                            "out_tokens": float(np.mean([data[a][k]["out_tokens"] for k in ks])),
                            "repeat4": float(np.mean([data[a][k]["repeat4"] for k in ks]))}
            out["tasks"].setdefault(task, {})[str(L)] = cells
            for ref in ("iq2xxs", "base2"):
                if ref not in cells or len(cells) < 2:
                    continue
                for a in cells:
                    if a == ref:
                        continue
                    ks = sorted(set(k for k in data[a] if k[0] == task and k[1] == L) &
                                set(k for k in data[ref] if k[0] == task and k[1] == L))
                    if len(ks) < 5:
                        continue
                    d = np.array([int(data[a][k]["ok"]) - int(data[ref][k]["ok"]) for k in ks], dtype=float)
                    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
                    out.setdefault("pairs_vs_%s" % ref, {})["%s|%d|%s" % (task, L, a)] = {
                        "n_pairs": len(ks), "diff": float(d.mean()),
                        "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
                        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        "mcnemar_b": int(np.sum(d > 0)), "mcnemar_c": int(np.sum(d < 0))}
    json.dump(out, open(os.path.join(RES, "e_h5_summary%s.json" % suffix), "w"),
              indent=1, ensure_ascii=False)
    log("wrote results/e_h5_summary%s.json" % suffix)
    for task, byL in out["tasks"].items():
        for L, cells in sorted(byL.items(), key=lambda x: int(x[0])):
            log("  %-5s %6s  %s" % (task, L, " ".join("%s=%.3f(n=%d)" % (a, c["acc"], c["n"]) for a, c in cells.items())))
    for ref in ("iq2xxs", "base2"):
        for k, v in out.get("pairs_vs_%s" % ref, {}).items():
            log("  vs %-8s %-26s d=%+.3f ci90=[%+.3f,%+.3f] b=%d c=%d" % (
                ref, k, v["diff"], v["ci90"][0], v["ci90"][1], v["mcnemar_b"], v["mcnemar_c"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "report", "all"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--port", type=int, default=18301)
    ap.add_argument("--limit", type=int, default=0, help="每类任务最多跑前 N 条（0=全部）")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.cmd in ("build", "all") and not os.path.exists(ITEMS):
        build_items(a.port)
    if a.cmd in ("run", "all"):
        if not os.path.exists(ITEMS):
            build_items(a.port)
        port = a.port
        for arm in a.arms.split(","):
            arm = arm.strip()
            if arm:
                run_arm(arm, port, a.limit)
                port += 1
    if a.cmd in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
