"""E-A1g: BFCL-v3 multi_turn_base (20 题) 两个干预实验, 对 tcq_nl_global 基线 acc 0.15.

    A1G_MODE=single_call  上游响应只保留第一个 tool_call (检验"计划先行/一次多步"假说)
    A1G_MODE=cwd_hint     每次请求末尾追加 cwd 状态信念 system 提示 (检验"cwd 状态信念"假说)

两模式都用 a1a 的同一臂/同一 llama-server 参数/同一 EvalScope 参数, 顺序执行, 各自独占
results/a1g/<mode>/; 报告 json -> results/a1g_<mode>.json; 汇总 -> results/a1g_summary.json.

E-A1h 复用同一代理: --arm/--subset 可覆盖, 产物隔离到
results/a1h/<arm>/<subset>/ 与 results/a1h_<arm>_<subset>.json (+ .server.log), 不动 a1g 产物.

    E:\\CUDA11.8\\train\\a1_env\\Scripts\\python.exe e_a1g_intervene.py --mode single_call
    E:\\CUDA11.8\\train\\a1_env\\Scripts\\python.exe e_a1g_intervene.py --mode cwd_hint
    E:\\CUDA11.8\\train\\a1_env\\Scripts\\python.exe e_a1g_intervene.py --mode cwd_hint --arm base2 --subset multi_turn_long_context
    E:\\CUDA11.8\\train\\a1_env\\Scripts\\python.exe e_a1g_intervene.py --summary
"""
import argparse
import glob
import json
import os
import time

import e_a1a_bfcl as A

ARM = "tcq_nl_global"
SUBSET = "multi_turn_base"
SUBSETS = ["multi_turn_base", "multi_turn_long_context"]
OUT_ROOT = os.path.join(A.RES, "a1g")        # 旧 A1g 产物 (summarize 用, 保留不覆盖)
OUT_ROOT_A1H = os.path.join(A.RES, "a1h")    # A1h: 六臂 x 两子集 cwd_hint
BASE_REVIEW = os.path.join(A.RES, "a1a", ARM, "reviews", ARM, "bfcl_v3_%s.jsonl" % SUBSET)


# ---------------------------------------------------------------- 单模式运行
def run_mode(mode, limit, subset=None, arm=None):
    assert mode in ("single_call", "cwd_hint"), mode
    arm = arm or ARM
    subset = subset or SUBSET
    cfg = A.CONFIGS[arm]
    work_dir = os.path.join(OUT_ROOT_A1H, arm, subset)
    os.makedirs(work_dir, exist_ok=True)
    server_log = os.path.join(A.RES, "a1h_%s_%s.server.log" % (arm, subset))
    os.environ["A1G_MODE"] = mode
    rec = {"mode": mode, "config": arm, "path": os.path.basename(cfg["path"]),
           "domain": cfg["domain"], "subset": subset, "limit": limit,
           "threads": A.THREADS, "ctx": A.CTX, "status": "failed", "error": None}
    proc = fh = proxy = None
    t0 = time.time()
    try:
        proc, fh = A.start_server(cfg, server_log)
        proxy = A._DomainProxy(A.PORT, cfg["domain"])
        from evalscope import TaskConfig, run_task
        tcfg = TaskConfig(
            model=arm, model_id=arm,
            api_url="http://127.0.0.1:%d/v1" % A.PROXY_PORT, api_key="EMPTY",
            eval_type="openai_api",
            datasets=["bfcl_v3"],
            dataset_args={"bfcl_v3": {
                "subset_list": [subset],
                "extra_params": {"underscore_to_dot": True, "is_fc_model": True},
            }},
            generation_config={"temperature": 0.0, "max_tokens": 4096,
                               "parallel_tool_calls": True, "timeout": 900, "stream": False},
            eval_batch_size=1, limit=limit, seed=20261111,
            work_dir=work_dir, no_timestamp=True,
        )
        res = run_task(tcfg)
        try:
            report = {k: (json.loads(v.model_dump_json()) if hasattr(v, "model_dump_json") else str(v))
                      for k, v in (res or {}).items()}
        except Exception:  # noqa: BLE001
            report = {"raw": str(res)[:2000]}
        beh = {}
        for p in glob.glob(os.path.join(work_dir, "predictions", "*", "bfcl_v3_*.jsonl")):
            beh[os.path.basename(p)] = A.behavior_metrics(p)
        rec.update({"status": "ok", "report": report, "behavior": beh})
    except BaseException as e:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        raise
    finally:
        rec["minutes"] = round((time.time() - t0) / 60.0, 1)
        if proxy is not None:
            proxy.stop()
        A.stop_server(proc, fh)
        if not A.wait_port_free(A.PORT, timeout=120):
            rec["server_still_up"] = True
        out = os.path.join(A.RES, "a1h_%s_%s.json" % (arm, subset))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        A.log("[a1h/%s/%s/%s] %s %.1f min -> %s" % (
            arm, subset, mode, rec["status"], rec["minutes"], out))
    return rec


# ---------------------------------------------------------------- 轨迹指标 (同 e_a1f_decomp 口径)
def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def find_review(work_root, model_id, subset):
    c = sorted(glob.glob(os.path.join(work_root, "reviews", model_id, "bfcl_v3_%s.jsonl*" % subset)))
    if not c:
        return None
    for x in c:
        if "rerun" in os.path.basename(x):
            return x
    return c[0]


def ck(tc):
    fn = (tc or {}).get("function") or {}
    return (fn.get("name"), json.dumps(fn.get("arguments"), sort_keys=True, ensure_ascii=False))


def analyze(p):
    rows = []
    if p:
        for r in read_jsonl(p):
            sc = ((r.get("sample_score") or {}).get("score") or {}).get("value") or {}
            msgs = r.get("messages") or []
            asst = [m for m in msgs if m.get("role") == "assistant"]
            tools = [m for m in msgs if m.get("role") == "tool"]
            calls, multi = [], 0
            for m in asst:
                t = m.get("tool_calls") or []
                if len(t) >= 2:
                    multi += 1
                calls += [ck(x) for x in t]
            rep = sum(1 for i in range(1, len(calls)) if calls[i] == calls[i - 1])
            err = sum(1 for t in tools if "error" in (t.get("content") or "").lower())
            cd_err = sum(1 for t in tools if "cd:" in (t.get("content") or "")
                         and "No such file or directory" in (t.get("content") or ""))
            rows.append({"index": r.get("index"), "acc": float(sc.get("acc") or 0.0),
                         "turns": len(asst), "n_calls": len(calls),
                         "multi": multi, "repeat_pairs": rep, "tool_errors": err,
                         "cd_errors": cd_err,
                         "terminated": bool(asst) and not (asst[-1].get("tool_calls") or [])})
    n = len(rows)
    if not n:
        return {"n": 0, "source": p}

    def avg(k, s=None):
        s = rows if s is None else s
        return round(sum(x[k] for x in s) / len(s), 3) if s else None

    fail = [x for x in rows if x["acc"] < 1.0]
    return {"n": n, "source": p, "acc": round(sum(x["acc"] for x in rows) / n, 4),
            "turns_mean": avg("turns"), "calls_mean": avg("n_calls"),
            "multi_call_rate": round(sum(1 for x in rows if x["multi"]) / n, 3),
            "repeat_pairs_mean": avg("repeat_pairs"),
            "tool_errors_mean": avg("tool_errors"),
            "tool_errors_total": sum(x["tool_errors"] for x in rows),
            "cd_errors_mean": avg("cd_errors"),
            "cd_errors_total": sum(x["cd_errors"] for x in rows),
            "terminated_rate": round(sum(1 for x in rows if x["terminated"]) / n, 3),
            "fail": {"n": len(fail), "turns_mean": avg("turns", fail),
                     "repeat_pairs_mean": avg("repeat_pairs", fail),
                     "tool_errors_mean": avg("tool_errors", fail),
                     "cd_errors_mean": avg("cd_errors", fail)},
            "rows": rows}


# ---------------------------------------------------------------- 汇总
def verdict(d_base, d_mode, hypothesis):
    """d_base/d_mode: analyze() 结果; 返回 (delta 描述, 是否支持假说)."""
    da = round(d_mode["acc"] - d_base["acc"], 3)
    if da >= 0.10:
        return "支持(acc +%.3f)" % da, True
    if da <= -0.05:
        return "反向(acc %.3f)" % da, False
    return "无显著变化(acc %+.3f)" % da, False


def summarize():
    base = analyze(BASE_REVIEW if os.path.exists(BASE_REVIEW) else find_review(
        os.path.join(A.RES, "a1a", ARM), ARM, SUBSET))
    out = {"arm": ARM, "subset": SUBSET, "baseline_acc_ref": 0.15,
           "baseline": base, "modes": {}, "delta_vs_baseline": {}}
    for mode in ("single_call", "cwd_hint"):
        p = find_review(os.path.join(OUT_ROOT, mode), ARM, SUBSET)
        if not p:
            out["modes"][mode] = {"n": 0, "skipped": True,
                                  "reason": "no review jsonl (run failed?), see results/a1g_%s.run.log" % mode}
            out["delta_vs_baseline"][mode] = {"skipped": True}
            continue
        m = analyze(p)
        out["modes"][mode] = m
        bi = {r["index"]: r["acc"] for r in base.get("rows", [])}
        out["delta_vs_baseline"][mode] = {
            "acc": round(m["acc"] - base["acc"], 3),
            "cd_errors_total": m["cd_errors_total"] - base["cd_errors_total"],
            "repeat_pairs_mean": round(m["repeat_pairs_mean"] - base["repeat_pairs_mean"], 3),
            "turns_mean": round(m["turns_mean"] - base["turns_mean"], 3),
            "multi_call_rate": round(m["multi_call_rate"] - base["multi_call_rate"], 3),
            "flipped_to_ok": [r["index"] for r in m.get("rows", []) if r["acc"] >= 1.0 and bi.get(r["index"], 0) < 1.0],
            "flipped_to_fail": [r["index"] for r in m.get("rows", []) if r["acc"] < 1.0 and bi.get(r["index"], 0) >= 1.0],
        }
    s1 = out["modes"].get("single_call") or {}
    s2 = out["modes"].get("cwd_hint") or {}
    parts, hits = [], []
    if s1.get("n"):
        t1, ok1 = verdict(base, s1, "计划先行")
        hits.append("计划先行" if ok1 else "")
        parts.append("single_call acc %.2f->%.2f(%s, 多调用轮 %.2f->%.2f, cd 错误 %d->%d, 重复对 %.2f->%.2f)" % (
            base["acc"], s1["acc"], t1, base["multi_call_rate"], s1["multi_call_rate"],
            base["cd_errors_total"], s1["cd_errors_total"],
            base["repeat_pairs_mean"], s1["repeat_pairs_mean"]))
    else:
        parts.append("single_call 未产出结果")
    if s2.get("n"):
        t2, ok2 = verdict(base, s2, "状态信念")
        hits.append("状态信念" if ok2 else "")
        parts.append("cwd_hint acc %.2f->%.2f(%s, cd 错误 %d->%d, 重复对 %.2f->%.2f, 轮数 %.1f->%.1f)" % (
            base["acc"], s2["acc"], t2, base["cd_errors_total"], s2["cd_errors_total"],
            base["repeat_pairs_mean"], s2["repeat_pairs_mean"],
            base["turns_mean"], s2["turns_mean"]))
    else:
        parts.append("cwd_hint 未产出结果")
    hits = [h for h in hits if h]
    if hits:
        tail = " → 支持%s假说" % "/".join(hits)
    else:
        tail = " → 两个假说均未获支持"
    out["supports"] = hits
    out["judgement"] = "; ".join(parts) + tail + "。"
    path = os.path.join(A.RES, "a1g_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    def line(name, s):
        if not s.get("n"):
            print("%-12s NO DATA" % name)
            return
        print("%-12s acc=%.3f turns=%.1f multi=%.2f rep=%.2f cd_err=%d tool_err=%d term=%.2f" % (
            name, s["acc"], s["turns_mean"], s["multi_call_rate"],
            s["repeat_pairs_mean"], s["cd_errors_total"], s["tool_errors_total"],
            s["terminated_rate"]))

    line("baseline", base)
    line("single_call", s1)
    line("cwd_hint", s2)
    print(out["judgement"])
    print("-> %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, choices=["single_call", "cwd_hint"])
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--subset", default=SUBSET, choices=SUBSETS)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    globals()["ARM"] = args.arm
    if args.summary:
        summarize()
        return
    if not args.mode:
        ap.error("--mode or --summary required")
    run_mode(args.mode, args.limit, subset=args.subset, arm=args.arm)


if __name__ == "__main__":
    main()
