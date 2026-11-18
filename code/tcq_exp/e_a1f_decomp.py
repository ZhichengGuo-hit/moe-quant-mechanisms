"""E-A1f: BFCL 多轮失败轨迹机制分解（零模型调用，只读 A1a reviews 产物）。

每臂 x 子集：acc、轮数、调用数、首轮调用数、多调用轮占比、相邻重复调用对、工具报错数、终止率。
配对：iq2xxs 成功而目标失败的题上，目标的轮数/重复/报错特征。
输出 results/e_a1f_traj_decomp.json
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BASE = os.path.join(RES, "a1a")
ARMS = ["iq2xxs", "ud_q2kxl", "ud_q4km", "tcq_nl_global", "tcq_nl_math", "base2"]
SUBSETS = ["multi_turn_base", "multi_turn_long_context"]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def find_review(arm, subset):
    c = sorted(glob.glob(os.path.join(BASE, arm, "reviews", arm, "bfcl_v3_%s.jsonl*" % subset)))
    if not c:
        return None
    for x in c:
        if "rerun" in os.path.basename(x):
            return x
    return c[0]


def ck(tc):
    fn = (tc or {}).get("function") or {}
    return (fn.get("name"), json.dumps(fn.get("arguments"), sort_keys=True, ensure_ascii=False))


def analyze(arm, subset):
    p = find_review(arm, subset)
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
            rows.append({"index": r.get("index"), "acc": float(sc.get("acc") or 0.0),
                         "turns": len(asst), "n_calls": len(calls),
                         "multi": multi, "repeat_pairs": rep, "tool_errors": err,
                         "terminated": bool(asst) and not (asst[-1].get("tool_calls") or [])})
    n = len(rows)
    if not n:
        return {"n": 0, "source": p}

    def avg(k, s=None):
        s = rows if s is None else s
        return round(sum(x[k] for x in s) / len(s), 3) if s else None

    succ = [x for x in rows if x["acc"] >= 1.0]
    fail = [x for x in rows if x["acc"] < 1.0]
    return {"n": n, "source": p, "acc": round(sum(x["acc"] for x in rows) / n, 4),
            "turns_mean": avg("turns"), "calls_mean": avg("n_calls"),
            "multi_call_rate": round(sum(1 for x in rows if x["multi"]) / n, 3),
            "repeat_pairs_mean": avg("repeat_pairs"), "tool_errors_mean": avg("tool_errors"),
            "terminated_rate": round(sum(1 for x in rows if x["terminated"]) / n, 3),
            "succ": {"n": len(succ), "turns_mean": avg("turns", succ), "repeat_pairs_mean": avg("repeat_pairs", succ)},
            "fail": {"n": len(fail), "turns_mean": avg("turns", fail), "repeat_pairs_mean": avg("repeat_pairs", fail),
                     "tool_errors_mean": avg("tool_errors", fail)},
            "rows": rows}


def main():
    out = {a: {s: analyze(a, s) for s in SUBSETS} for a in ARMS}
    paired = {}
    for arm in ARMS:
        if arm == "iq2xxs":
            continue
        paired[arm] = {}
        for sub in SUBSETS:
            a = {r["index"]: r for r in out["iq2xxs"][sub].get("rows", [])}
            b = {r["index"]: r for r in out[arm][sub].get("rows", [])}
            common = sorted(set(a) & set(b))
            f = [i for i in common if a[i]["acc"] >= 1.0 and b[i]["acc"] < 1.0]
            o = [i for i in common if a[i]["acc"] >= 1.0 and b[i]["acc"] >= 1.0]
            paired[arm][sub] = {
                "common": len(common), "iq_ok_t_fail": len(f), "iq_ok_t_ok": len(o), "fail_idx": f,
                "fail_turns": round(sum(b[i]["turns"] for i in f) / len(f), 2) if f else None,
                "fail_rep": round(sum(b[i]["repeat_pairs"] for i in f) / len(f), 2) if f else None,
                "ok_turns": round(sum(b[i]["turns"] for i in o) / len(o), 2) if o else None,
                "ok_rep": round(sum(b[i]["repeat_pairs"] for i in o) / len(o), 2) if o else None,
            }
    with open(os.path.join(RES, "e_a1f_traj_decomp.json"), "w", encoding="utf-8") as fh:
        json.dump({"arms": out, "paired": paired}, fh, ensure_ascii=False, indent=1)
    for arm in ARMS:
        for sub in SUBSETS:
            s = out[arm][sub]
            if not s.get("n"):
                print(arm, sub, "NO DATA", s.get("source"))
                continue
            print("%-14s %-22s acc=%.3f turns=%.1f calls=%.1f multi=%.2f rep=%.1f err=%.1f term=%.2f | fail n=%d turns=%s rep=%s" % (
                arm, sub, s["acc"], s["turns_mean"], s["calls_mean"], s["multi_call_rate"],
                s["repeat_pairs_mean"], s["tool_errors_mean"], s["terminated_rate"],
                s["fail"]["n"], s["fail"]["turns_mean"], s["fail"]["repeat_pairs_mean"]))
    for arm in paired:
        for sub, s in paired[arm].items():
            if s["common"]:
                print("PAIR %-13s %-22s common=%d iq_ok_t_fail=%d fail_turns=%s fail_rep=%s ok_turns=%s ok_rep=%s" % (
                    arm, sub, s["common"], s["iq_ok_t_fail"], s["fail_turns"], s["fail_rep"], s["ok_turns"], s["ok_rep"]))


if __name__ == "__main__":
    main()
