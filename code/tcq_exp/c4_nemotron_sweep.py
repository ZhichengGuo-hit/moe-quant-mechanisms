"""C4: Nemotron 位宽扫描 + CoT 长度曲线。

同一 i1 系列、同一量化器 (imatrix?) 的 5 个 GGUF 档位, 统一 llama-server 协议跑
GSM8K 前 100 题, 逐题记录 acc / predicted_n / finish_reason / 行为分类, 最后出
summary + CoT 长度曲线 (PNG/CSV) + "渐变 vs 断崖" 判定。

    # 串行跑全部档位 (不存在的档位跳过), 日志 results/c4.run.log
    python c4_nemotron_sweep.py run
    # 抽 20 题做人工校准候选
    python c4_nemotron_sweep.py calib
    # 汇总 (需人工校准结果 results/c4_calib_manual.json; 缺失则一致率记 null)
    python c4_nemotron_sweep.py summarize

协议: /v1/chat/completions, temperature 0, top_k 1, n_predict(max_tokens) 1024,
cache_prompt false; system 与判分复用 e17_task_eval.SYSTEM_MATH / extract_number。
Nemotron 自带 chat template, 不传 enable_thinking 等 Qwen 专用参数。
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-server.exe"
MODEL_DIR = r"E:\CUDA11.8\model\nemotron"
RUN_LOG = os.path.join(RES, "c4.run.log")
OUT_JSONL = os.path.join(RES, "c4_nemotron_sweep.jsonl")
SUMMARY = os.path.join(RES, "c4_summary.json")
CURVE_PNG = os.path.join(RES, "c4_cot_curve.png")
CURVE_CSV = os.path.join(RES, "c4_cot_curve.csv")
CALIB_CAND = os.path.join(RES, "c4_calib_candidates.jsonl")
CALIB_MANUAL = os.path.join(RES, "c4_calib_manual.json")

# 档位按 bpw 升序; size_gib 为任务书给出的目标大小, bpw 为 llama.cpp 标称值
QUANTS = [
    {"tag": "Q2_K", "file": "Nemotron-Cascade-2-30B-A3B.i1-Q2_K.gguf",
     "size_gib": 16.68, "bpw": 2.63, "port": 18301},
    {"tag": "Q3_K_S", "file": "Nemotron-Cascade-2-30B-A3B.i1-Q3_K_S.gguf",
     "size_gib": 16.70, "bpw": 3.50, "port": 18302},
    {"tag": "Q3_K_M", "file": "Nemotron-Cascade-2-30B-A3B.i1-Q3_K_M.gguf",
     "size_gib": 18.49, "bpw": 3.91, "port": 18303},
    {"tag": "Q4_K_M", "file": "Nemotron-Cascade-2-30B-A3B.i1-Q4_K_M.gguf",
     "size_gib": 22.83, "bpw": 4.83, "port": 18304},
    {"tag": "Q6_K", "file": "Nemotron-Cascade-2-30B-A3B.i1-Q6_K.gguf",
     "size_gib": 31.21, "bpw": 6.56, "port": 18305},
]
HIGH_ANCHORS = {"Q4_K_M", "Q6_K"}
N_PREDICT = 1024
STOPS = ["\n\nQuestion:", "\n\nQ:"]
CTX = 4096
NGL = 20
THREADS = 16
BUDGET_H = 10.0

from e17_task_eval import SYSTEM_MATH, extract_number  # noqa: E402


def say(msg):
    line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    # E: 为 HDD RAID, 偶发路径解析失败 (FileNotFoundError), 写日志需重试
    for attempt in range(20):
        try:
            os.makedirs(RES, exist_ok=True)
            with open(RUN_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return
        except OSError:
            time.sleep(2)
    print("WARN 日志写入失败", flush=True)


def retry_open(path, mode="w", tries=30):
    last = None
    for _ in range(tries):
        try:
            return open(path, mode, encoding="utf-8")
        except OSError as e:  # 文件系统瞬断
            last = e
            time.sleep(2)
    raise last


def retry_stat(path, tries=30):
    last = None
    for _ in range(tries):
        try:
            if os.path.exists(path):
                return os.path.getsize(path)
            last = FileNotFoundError(path)
        except OSError as e:
            last = e
        time.sleep(2)
    raise last


def _exists(path, tries=10):
    for _ in range(tries):
        try:
            if os.path.exists(path):
                return True
        except OSError:
            pass
        time.sleep(1)
    return False



# ---------------- HTTP ----------------

def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start_server(model_path, port, log_path, timeout=1800):
    cmd = [BIN, "-m", model_path, "--host", "127.0.0.1", "--port", str(port),
           "-c", str(CTX), "-t", str(THREADS), "-np", "1", "-ngl", str(NGL),
           "--no-webui", "--verbosity", "1"]
    log = retry_open(log_path, "w")
    log.write(" ".join(cmd) + "\n")
    log.flush()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server 退出 code={proc.returncode}")
            try:
                if get_json(f"http://127.0.0.1:{port}/health", 3).get("status") == "ok":
                    return proc, log, time.time() - t0
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3)
        raise RuntimeError("llama-server 启动超时")
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        raise


def stop_server(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.close()


# ---------------- 生成与判分 ----------------

def finish_of(resp, choice, pn):
    fr = choice.get("finish_reason")
    if fr:
        return fr
    if resp.get("stopped_limit") or resp.get("truncated"):
        return "length"
    if resp.get("stopped_eos"):
        return "eos"
    return "length" if pn >= N_PREDICT else "stop"


def gen(port, question):
    payload = {"messages": [{"role": "system", "content": SYSTEM_MATH},
                            {"role": "user", "content": question}],
               "max_tokens": N_PREDICT, "temperature": 0.0, "top_k": 1, "seed": 1234,
               "cache_prompt": False, "stop": STOPS, "stream": False}
    resp = post_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    tim = resp.get("timings") or {}
    pn = int(tim.get("predicted_n") or (resp.get("usage") or {}).get("completion_tokens") or 0)
    return content, reasoning, pn, finish_of(resp, choice, pn), tim


# ---------------- 行为分类 (规则版) ----------------

RE_MATH = re.compile(r"\d\s*[-+*/×÷]\s*=?\s*\d")
LOOP_WIN = 64


def is_loop(text):
    if len(text) >= 2 * LOOP_WIN:
        for i in range(0, len(text) - 2 * LOOP_WIN + 1, 16):
            if text[i:i + LOOP_WIN] == text[i + LOOP_WIN:i + 2 * LOOP_WIN]:
                return True
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    run, prev = 1, None
    for l in lines:
        if l == prev:
            run += 1
            if run >= 3 and len(l) >= 8:
                return True
        else:
            run, prev = 1, l
    cnt = {}
    for l in lines:
        cnt[l] = cnt.get(l, 0) + 1
    return any(c >= 3 and len(l) >= 8 for l, c in cnt.items())


def classify(text, pn, finish):
    """truncated > loop > reasoning > direct。"""
    if finish == "length" or pn >= N_PREDICT:
        return "truncated"
    if is_loop(text):
        return "loop"
    prefix = text.split("####")[0] if "####" in text else text
    nlines = len([l for l in prefix.splitlines() if l.strip()])
    if RE_MATH.search(prefix) or nlines >= 2 or len(prefix.strip()) >= 100:
        return "reasoning"
    return "direct"


# ---------------- 数据 ----------------

def load_tasks(n):
    with retry_open(os.path.join(DATA, "gsm8k_200.jsonl"), "r") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows[:n]


def done_ids():
    done = {}
    if os.path.exists(OUT_JSONL):
        with retry_open(OUT_JSONL, "r") as f:
            for l in f:
                try:
                    r = json.loads(l)
                    done.setdefault(r["tag"], set()).add(r["task_id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


def check_quant(q):
    path = os.path.join(MODEL_DIR, q["file"])
    try:
        actual = retry_stat(path, tries=3)
    except OSError:
        # 说明是否仍在下载 (.part*)
        partial = 0
        for i in range(8):
            pp = path + f".part{i}"
            try:
                if _exists(pp, tries=1):
                    partial += os.path.getsize(pp)
            except OSError:
                pass
        extra = (f"; 下载中 .part* 已 {partial/1024**3:.2f} GiB" if partial else "")
        return None, f"文件不存在 ({q['file']}{extra})"
    if actual == 0:
        return None, f"文件不存在 ({q['file']})"
    target = q["size_gib"] * (1024 ** 3)
    if actual < target * 0.95:
        return None, f"文件不完整: {actual/1024**3:.2f} GiB < 目标 {q['size_gib']} GiB 的 95%"
    return path, actual


# ---------------- 运行 ----------------

def run_quant(q, path, actual, limit, tag_done):
    rows = load_tasks(limit)
    todo = [r for r in rows if r["task_id"] not in tag_done]
    if not todo:
        say(f"[{q['tag']}] 已完成 {len(tag_done)} 题, 跳过")
        return None
    log_path = os.path.join(RES, f"c4_{q['tag']}.server.log")
    say(f"[{q['tag']}] 开始 待跑 {len(todo)}/{len(rows)} 题 port={q['port']}")
    t0 = time.time()
    proc, log, load_s = start_server(path, q["port"], log_path)
    say(f"[{q['tag']}] server ready in {load_s:.1f}s")
    n_ok = 0
    try:
        with retry_open(OUT_JSONL, "a") as f:
            for i, row in enumerate(todo):
                ts = time.time()
                try:
                    content, reasoning, pn, finish, _ = gen(q["port"], row["question"].strip())
                except Exception as e:  # noqa: BLE001  单题失败不中断
                    say(f"  [skip] {q['tag']} {row['task_id']}: {e!r}")
                    continue
                # 模板默认保留思维链, CoT 落在 reasoning_content, 统计与分类都用全文
                full = (reasoning + "\n" + content) if reasoning else content
                pred = extract_number(full)
                gold = float(row["gold"])
                ok = pred is not None and abs(pred - gold) < 1e-4
                rec = {"tag": q["tag"], "bpw": q["bpw"], "size_gib": q["size_gib"],
                       "task_id": row["task_id"], "acc": bool(ok), "pred": pred,
                       "gold": row["gold"], "predicted_n": pn,
                       "finish_reason": finish,
                       "behavior": classify(full, pn, finish),
                       "out_len": len(full), "reasoning_len": len(reasoning),
                       "answer": content[-200:],
                       "head200": full[:200], "tail200": full[-200:],
                       "gen_ms": round((time.time() - ts) * 1000)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += int(ok)
                if i < 3 or (i + 1) % 20 == 0 or i + 1 == len(todo):
                    el = time.time() - t0
                    say(f"  {i+1}/{len(todo)} {row['task_id']} acc={n_ok/(i+1):.3f} "
                        f"pn={pn} {finish} {rec['behavior']} {rec['gen_ms']/1000:.1f}s "
                        f"elapsed={el/60:.1f}m")
    finally:
        stop_server(proc, log)
    el = time.time() - t0
    say(f"[{q['tag']}] 结束 {len(todo)} 题 acc={n_ok/max(1,len(todo)):.3f} "
        f"用时 {el/60:.1f} min")
    return {"elapsed_s": el, "load_s": load_s, "n": len(todo), "actual_bytes": actual}


def cmd_run(args):
    os.makedirs(RES, exist_ok=True)
    say(f"==== C4 位宽扫描开始 (ngl={NGL} ctx={CTX} t={THREADS} n_predict={N_PREDICT}) ====")
    done = done_ids()
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    plan = []
    for q in QUANTS:
        if only and q["tag"] not in only:
            continue
        path, info = check_quant(q)
        if path is None:
            plan.append({"tag": q["tag"], "status": "skipped", "reason": info, **q})
            say(f"[{q['tag']}] SKIP: {info}")
        else:
            plan.append({"tag": q["tag"], "status": "pending", "path": path,
                         "actual_bytes": info, **q})
    meta = {}
    ran = []
    for item in plan:
        if item["status"] == "skipped":
            continue
        tag = item["tag"]
        # 时间预算: 已跑档位实测每题耗时 × 文件大小比 推剩余档位
        limit = args.limit
        if ran:
            measured = [(m["avg_s"], m["size_gib"]) for m in ran if m["avg_s"]]
            if measured:
                base_s, base_gib = measured[-1]
                remaining = [p for p in plan if p["status"] == "pending"
                             and p["tag"] >= tag]
                proj = sum(base_s * (p["size_gib"] / base_gib) * args.limit
                           for p in remaining)
                used = sum(m["elapsed_s"] for m in ran) / 3600.0
                if used + proj / 3600.0 > BUDGET_H and tag in HIGH_ANCHORS:
                    limit = min(limit, 50)
                    say(f"[{tag}] 预算保护: 预计总耗时 {used + proj/3600:.1f}h > {BUDGET_H}h,"
                        f" 高锚降到 {limit} 题")
        r = run_quant(item, item["path"], item["actual_bytes"], limit, done.get(tag, set()))
        if r:
            r.update({"tag": tag, "bpw": item["bpw"], "size_gib": item["size_gib"],
                      "limit": limit, "avg_s": r["elapsed_s"] / max(1, r["n"])})
            ran.append(r)
            item["status"] = "done"
            item["elapsed_s"] = r["elapsed_s"]
    say("==== C4 位宽扫描结束 ====")
    with retry_open(os.path.join(RES, "c4_run_meta.json"), "w") as f:
        json.dump({"plan": plan, "ran": ran}, f, indent=2, ensure_ascii=False)


# ---------------- 人工校准 ----------------

def cmd_calib(args):
    recs = {}
    with retry_open(OUT_JSONL, "r") as f:
        for l in f:
            r = json.loads(l)
            recs.setdefault(r["tag"], []).append(r)
    tags = [q["tag"] for q in QUANTS if q["tag"] in recs]
    n_per = {t: 4 for t in tags}
    extra = max(0, 20 - 4 * len(tags))
    for i in range(extra):  # 每档先保底 4 题, 余量轮转补齐到 20 题
        n_per[tags[i % len(tags)]] += 1
    out = []
    for t in tags:
        rows = recs[t]
        per = n_per[t]
        picks = []
        seen = set()
        # 稀有类别优先各取 1 题, 保证 truncated/loop/direct 都被人工核对到
        for r in rows:
            if r["behavior"] != "reasoning" and r["behavior"] not in seen and len(picks) < per - 1:
                picks.append(r)
                seen.add(r["behavior"])
        common = [r for r in rows if r["behavior"] == "reasoning"]
        need = per - len(picks)
        step = max(1, len(common) // max(1, need))
        picks += common[::step][:need]
        picks.sort(key=lambda r: r["task_id"])
        for r in picks:
            out.append({"tag": t, "task_id": r["task_id"], "rule": r["behavior"],
                        "predicted_n": r["predicted_n"], "finish_reason": r["finish_reason"],
                        "head200": r["head200"], "tail200": r["tail200"],
                        "out_len": r["out_len"], "answer": r.get("answer", "")})
    with retry_open(CALIB_CAND, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    say(f"[calib] 候选 {len(out)} 题 ({', '.join(f'{t}:{n_per[t]}' for t in tags)}) -> {CALIB_CAND}")


# ---------------- 汇总 + 曲线 ----------------

def quant_stats(tag):
    rows = []
    with retry_open(OUT_JSONL, "r") as f:
        for l in f:
            r = json.loads(l)
            if r["tag"] == tag:
                rows.append(r)
    if not rows:
        return None
    pns = [r["predicted_n"] for r in rows]
    beh = {k: 0 for k in ("reasoning", "direct", "truncated", "loop")}
    for r in rows:
        beh[r["behavior"]] = beh.get(r["behavior"], 0) + 1
    n = len(rows)
    return {"n": n, "acc": sum(r["acc"] for r in rows) / n,
            "avg_pn": statistics.mean(pns), "median_pn": statistics.median(pns),
            "behavior": {k: round(v / n, 4) for k, v in beh.items()},
            "behavior_n": beh,
            "reasoning_frac": round(sum(1 for r in rows if r.get("reasoning_len", 0) > 0) / n, 4),
            "avg_out_len": statistics.mean([r["out_len"] for r in rows])}


def load_manual():
    if not os.path.exists(CALIB_MANUAL):
        return None
    with retry_open(CALIB_MANUAL, "r") as f:
        m = json.load(f)
    return {(x["tag"], x["task_id"]): x["manual"] for x in m}


def draw_chart(points):
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1280, 720
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        f_t = ImageFont.truetype("arial.ttf", 22)
        f_l = ImageFont.truetype("arial.ttf", 16)
    except Exception:  # noqa: BLE001
        f_t = ImageFont.load_default()
        f_l = ImageFont.load_default()
    L, R, T, B = 110, W - 160, 70, H - 90
    d.text((L, 20), "C4 Nemotron sweep: GSM8K acc vs CoT length by bpw", fill="black", font=f_t)
    d.line([(L, T), (L, B)], fill="black", width=2)
    d.line([(R, T), (R, B)], fill="black", width=2)
    d.line([(L, B), (R, B)], fill="black", width=2)
    bps = [p["bpw"] for p in points]
    x0, x1 = min(bps) - 0.4, max(bps) + 0.4
    pmax = max(p["avg_pn"] for p in points) * 1.15 or 1
    for i in range(6):
        acc = i / 5.0
        y = B - (B - T) * acc
        d.line([(L, y), (R, y)], fill=(225, 225, 225), width=1)
        d.text((40, y - 8), f"{acc:.1f}", fill="black", font=f_l)
        pnv = pmax * i / 5.0
        d.text((R + 12, y - 8), f"{pnv:.0f}", fill=(180, 60, 60), font=f_l)
    def px(b):
        return L + (R - L) * (b - x0) / (x1 - x0)
    def py_acc(a):
        return B - (B - T) * a
    def py_pn(p):
        return B - (B - T) * (p / pmax)
    for b in bps:
        d.line([(px(b), B), (px(b), B + 6)], fill="black", width=1)
        d.text((px(b) - 14, B + 12), f"{b:.2f}", fill="black", font=f_l)
    d.text((L - 20, B + 42), "bpw", fill="black", font=f_l)
    d.text((10, T - 34), "acc", fill=(30, 80, 180), font=f_l)
    rt = "avg predicted_n"
    d.text((W - 10 - d.textlength(rt, font=f_l), T - 34), rt,
           fill=(180, 60, 60), font=f_l)
    seq = sorted(points, key=lambda p: p["bpw"])
    d.line([(px(p["bpw"]), py_acc(p["acc"])) for p in seq], fill=(30, 80, 180), width=3)
    d.line([(px(p["bpw"]), py_pn(p["avg_pn"])) for p in seq], fill=(180, 60, 60), width=3)
    for p in seq:
        x, y = px(p["bpw"]), py_acc(p["acc"])
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(30, 80, 180))
        d.text((x + 8, y - 26), f"{p['acc']:.3f}", fill=(30, 80, 180), font=f_l)
        x, y = px(p["bpw"]), py_pn(p["avg_pn"])
        d.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(180, 60, 60))
        d.text((x + 8, y + 6), f"{p['avg_pn']:.0f}", fill=(180, 60, 60), font=f_l)
        d.text((x - 10, B + 26), p["tag"], fill=(90, 90, 90), font=f_l)
    img.save(CURVE_PNG)
    with open(CURVE_CSV, "w", encoding="utf-8") as f:
        f.write("tag,bpw,acc,avg_predicted_n,median_predicted_n\n")
        for p in seq:
            f.write(f"{p['tag']},{p['bpw']},{p['acc']:.4f},{p['avg_pn']:.2f},"
                    f"{p['median_pn']:.1f}\n")


def verdict(points):
    seq = sorted(points, key=lambda p: p["bpw"])
    if len(seq) < 2:
        return "样本不足 (仅 %d 档), 无法判定" % len(seq), {}
    pns = [p["avg_pn"] for p in seq]
    accs = [p["acc"] for p in seq]
    rel_pn_range = (max(pns) - min(pns)) / max(1.0, max(pns))
    acc_range = max(accs) - min(accs)
    # 相邻档位突变: acc 大幅下降且 CoT 长度同步大幅缩短才算断崖
    gaps = []
    for a, b in zip(seq, seq[1:]):
        gaps.append({"from": a["tag"], "to": b["tag"],
                     "acc_delta": round(b["acc"] - a["acc"], 4),
                     "pn_ratio": round(b["avg_pn"] / max(1.0, a["avg_pn"]), 3)})
    cliff_gap = next((g for g in gaps
                      if g["acc_delta"] >= 0.15 and g["pn_ratio"] <= 0.70), None)
    span = "bpw %.2f–%.2f (%s→%s)" % (seq[0]["bpw"], seq[-1]["bpw"],
                                      seq[0]["tag"], seq[-1]["tag"])
    basis = {"n_quants": len(seq), "acc_range": round(acc_range, 4),
             "rel_pn_range": round(rel_pn_range, 4), "adjacent_gaps": gaps,
             "cliff_gap": cliff_gap}
    if cliff_gap:
        v = ("断崖: %s→%s (bpw %.2f→%.2f) acc %+.1fpp 且 CoT 长度 ×%.2f, "
             "其余相邻档位变化平缓; %s 全档 acc 极差 %.1fpp, avg CoT %.0f–%.0f tokens"
             % (cliff_gap["from"], cliff_gap["to"],
                [p for p in seq if p["tag"] == cliff_gap["from"]][0]["bpw"],
                [p for p in seq if p["tag"] == cliff_gap["to"]][0]["bpw"],
                cliff_gap["acc_delta"] * 100, cliff_gap["pn_ratio"], span,
                acc_range * 100, min(pns), max(pns)))
    else:
        v = ("渐变而非断崖: %s 全档 acc 极差 %.1fpp, avg CoT 长度 %.0f–%.0f tokens "
             "(相对极差 %.1f%%), 相邻档位无 acc≥15pp 且 CoT 缩短≥30%% 的突变"
             % (span, acc_range * 100, min(pns), max(pns), rel_pn_range * 100))
    hi, rest = seq[-1], seq[:-1]
    rest_acc = max(p["acc"] for p in rest)
    rest_pn = statistics.mean([p["avg_pn"] for p in rest])
    if hi["acc"] > rest_acc + 0.005:
        v += "; 最高位宽档 %s acc %.3f, 比其余档最高 %.3f 高 %.1fpp" % (
            hi["tag"], hi["acc"], rest_acc, (hi["acc"] - rest_acc) * 100)
    elif hi["acc"] < rest_acc - 0.005:
        v += "; 最高位宽档 %s acc %.3f, 比其余档最高 %.3f 低 %.1fpp" % (
            hi["tag"], hi["acc"], rest_acc, (rest_acc - hi["acc"]) * 100)
    else:
        v += "; 最高位宽档 %s acc %.3f, 与其余档最高 %.3f 持平" % (
            hi["tag"], hi["acc"], rest_acc)
    if hi["avg_pn"] >= rest_pn * 1.15:
        v += ", 且 CoT 明显更长 (avg %.0f vs 其余均值 %.0f tokens)" % (hi["avg_pn"], rest_pn)
    elif hi["avg_pn"] <= rest_pn * 0.85:
        v += ", 且 CoT 明显更短 (avg %.0f vs 其余均值 %.0f tokens)" % (hi["avg_pn"], rest_pn)
    else:
        v += ", CoT 长度与其余档持平 (avg %.0f vs 均值 %.0f tokens)" % (hi["avg_pn"], rest_pn)
    return v, basis


def cmd_summarize(args):
    meta_p = os.path.join(RES, "c4_run_meta.json")
    meta = {}
    if os.path.exists(meta_p):
        with retry_open(meta_p, "r") as f:
            meta = json.load(f)
    manual = load_manual()
    by_tag = {r["tag"]: r for r in meta.get("ran", [])}
    skipped = {p["tag"]: p["reason"] for p in meta.get("plan", []) if p["status"] == "skipped"}
    calib = None
    if manual is not None and _exists(CALIB_CAND):
        with retry_open(CALIB_CAND, "r") as f:
            cands = [json.loads(l) for l in f if l.strip()]
        match, pairs = [], []
        for c in cands:
            m = manual.get((c["tag"], c["task_id"]))
            if m is None:
                continue
            match.append(1 if m == c["rule"] else 0)
            pairs.append({"tag": c["tag"], "task_id": c["task_id"],
                          "rule": c["rule"], "manual": m})
        if match:
            calib = {"n": len(match), "agreement": sum(match) / len(match),
                     "by_rule_class": {k: sum(1 for p in pairs if p["rule"] == k)
                                       for k in ("reasoning", "direct", "truncated", "loop")},
                     "disagreements": [p for p in pairs if p["rule"] != p["manual"]]}
            per_tag = {}
            for p in pairs:
                per_tag.setdefault(p["tag"], []).append(1 if p["rule"] == p["manual"] else 0)
            calib["by_tag"] = {t: {"n": len(v), "agreement": sum(v) / len(v)}
                              for t, v in per_tag.items()}
    out = {"protocol": {"bin": BIN, "ngl": NGL, "ctx": CTX, "threads": THREADS,
                        "n_predict": N_PREDICT, "temperature": 0.0, "top_k": 1,
                        "cache_prompt": False, "endpoint": "/v1/chat/completions",
                        "task": "GSM8K first %d" % (args.limit or 100),
                        "chat_template": "Nemotron 自带模板, 默认保留思维链 "
                                         "(CoT 在 reasoning_content), 未传 enable_thinking",
                        "system": SYSTEM_MATH},
           "quants": [], "skipped": {}, "calibration": calib,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    points = []
    for q in QUANTS:
        st = quant_stats(q["tag"])
        entry = {"tag": q["tag"], "bpw": q["bpw"], "size_gib": q["size_gib"]}
        if st:
            entry.update(st)
            entry["elapsed_s"] = round(by_tag.get(q["tag"], {}).get("elapsed_s", 0), 1)
            entry["limit"] = by_tag.get(q["tag"], {}).get("limit")
            entry["skipped"] = False
            if calib:
                ct = (calib.get("by_tag") or {}).get(q["tag"]) or {}
                entry["calib_n"] = ct.get("n", 0)
                entry["calib_agreement"] = ct.get("agreement")
            points.append({"tag": q["tag"], "bpw": q["bpw"], "acc": st["acc"],
                           "avg_pn": st["avg_pn"], "median_pn": st["median_pn"]})
        else:
            _, why = check_quant(q)
            entry.update({"skipped": True,
                          "reason": why or skipped.get(q["tag"], "无记录")})
        out["quants"].append(entry)
    out["skipped"] = {e["tag"]: e["reason"] for e in out["quants"] if e.get("skipped")}
    total_s = sum(e.get("elapsed_s", 0) for e in out["quants"] if not e.get("skipped"))
    done_tags = [e["tag"] for e in out["quants"] if not e.get("skipped")]
    out["note"] = ("已就绪 %d/%d 档按 100 题协议完成 (%s); 实测总耗时 %.1f h "
                   "(预算 %.0f h); 未就绪档位跳过并在 skipped 中记录原因"
                   % (len(done_tags), len(QUANTS), ", ".join(done_tags),
                      total_s / 3600.0, BUDGET_H))
    if points:
        v, basis = verdict(points)
        out["verdict"] = v
        out["verdict_basis"] = basis
        try:
            draw_chart(points)
            out["curve"] = {"png": CURVE_PNG, "csv": CURVE_CSV}
        except Exception as e:  # noqa: BLE001  matplotlib/PIL 缺失时退化为 CSV
            with open(CURVE_CSV, "w", encoding="utf-8") as f:
                f.write("tag,bpw,acc,avg_predicted_n,median_predicted_n\n")
                for p in sorted(points, key=lambda x: x["bpw"]):
                    f.write(f"{p['tag']},{p['bpw']},{p['acc']:.4f},{p['avg_pn']:.2f},"
                            f"{p['median_pn']:.1f}\n")
            out["curve"] = {"png": None, "csv": CURVE_CSV, "note": "绘图库缺失: %r" % (e,)}
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    say(f"[summarize] -> {SUMMARY} verdict={out.get('verdict')}")
    for e in out["quants"]:
        if e.get("skipped"):
            say(f"  {e['tag']:7s} skipped: {e['reason']}")
        else:
            say(f"  {e['tag']:7s} bpw={e['bpw']:.2f} n={e['n']} acc={e['acc']:.3f} "
                f"avg_pn={e['avg_pn']:.1f} med={e['median_pn']:.0f} "
                f"trunc={e['behavior']['truncated']:.2f} loop={e['behavior']['loop']:.2f} "
                f"{e['elapsed_s']/60:.0f}m")


def main(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--only", default=None, help="只跑这些档位 (逗号分隔, 冒烟用)")
    sub.add_parser("calib")
    s = sub.add_parser("summarize")
    s.add_argument("--limit", type=int, default=100)
    a = ap.parse_args(argv)
    {"run": cmd_run, "calib": cmd_calib, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
