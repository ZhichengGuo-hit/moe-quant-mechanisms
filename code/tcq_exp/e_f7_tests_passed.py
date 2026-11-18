"""E-F7: HumanEval 连续指标 (每题通过 unit test 数) + 配对 t 检验。

做法: 对每个配置每条已完成记录, 用本地重放 (子进程) 把 check() 的每条 assert 单独判通过,
得到连续变量 tests_passed / tests_total; 再以"题"为单位做配对 t 检验 (基线 iq2xxs)。

输入: results/e17_*_humaneval.jsonl (+ hc1024 截断补跑, 按 task_id 合并)
输出: results/e_f7_tests_passed.json  (含每配置均值 / 配对 t / 连续指标 MDE)
      results/e_f7_tests_passed.csv   (逐题明细)
用法: python e_f7_tests_passed.py [--base iq2xxs] [--limit 0]
"""
import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
CASE = os.path.join(HERE, "e_f7_case.py")
TIMEOUT = 15


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def t_p_value(t, df):
    return betai(df / 2.0, 0.5, df / (df + t * t))


def run_case(code, test, entry):
    # ignore_cleanup_errors: Windows 上子进程句柄未释放会撞 WinError 145
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        p = os.path.join(td, "p.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"code": code, "test": test, "entry_point": entry}, f)
        try:
            r = subprocess.run([sys.executable, "-I", CASE, p], cwd=td, timeout=TIMEOUT,
                               capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return {"n": 0, "passed": 0, "flags": [], "error": "Timeout"}
        if r.returncode != 0:
            return {"n": 0, "passed": 0, "flags": [], "error": (r.stderr or "")[-200:]}
        try:
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return {"n": 0, "passed": 0, "flags": [], "error": "bad output"}


def load_rows():
    with open(os.path.join(DATA, "humaneval_164.jsonl"), encoding="utf-8") as f:
        return {r["task_id"]: r for r in (json.loads(l) for l in f if l.strip())}


def load_runs():
    """tag -> task_id -> record; 先读原始全量, 再读 hc1024 覆盖 (截断补跑优先)。"""
    per = {}
    suffix_hc = "_hc1024_humaneval.jsonl"
    suffix = "_humaneval.jsonl"
    files = sorted(glob.glob(os.path.join(RES, "e17_*_humaneval.jsonl")),
                   key=lambda p: ("_hc1024_" in os.path.basename(p), p))
    for p in files:
        name = os.path.basename(p)
        if name.endswith(suffix_hc):
            tag = name[len("e17_"):-len(suffix_hc)]
        elif name.endswith(suffix):
            tag = name[len("e17_"):-len(suffix)]
        else:
            continue
        d = per.setdefault(tag, {})
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                d[r["task_id"]] = r
    return per


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="iq2xxs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tags", default="", help="只看这些配置, 逗号分隔; 空=全部")
    ap.add_argument("--out", default="e_f7_tests_passed", help="输出文件名(无扩展名)")
    a = ap.parse_args(argv)
    rows = load_rows()
    per = load_runs()
    tags = list(per)
    if a.tags:
        want = [t.strip() for t in a.tags.split(",") if t.strip()]
        tags = [t for t in want if t in per]
        miss = [t for t in want if t not in per]
        if miss:
            print(f"[warn] 无结果文件: {miss}")
    if a.limit:
        keep = set(sorted(rows)[: a.limit])
    else:
        keep = set(rows)

    tasks = sorted(keep)
    jobs = [(t, tid, rows[tid]) for t in tags for tid in tasks
            if tid in per[t] and tid in rows]
    print(f"[plan] {len(tags)} configs x {len(tasks)} tasks = {len(jobs)} 例, workers={a.workers}",
          flush=True)

    detail = {}
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {}
        for t, tid, row in jobs:
            rec = per[t][tid]
            code = rec.get("code")
            if not code:
                import e17_task_eval as E17  # noqa: E402  懒导入, 破循环依赖
                code = E17.extract_code(row, rec.get("completion", ""))
            futs[ex.submit(run_case, code, row["test"], row["entry_point"])] = (t, tid, rec)
        for fu in futs:
            t, tid, rec = futs[fu]
            r = fu.result()
            detail[(t, tid)] = {"n": r["n"], "passed": r["passed"],
                                "passed_flag": bool(rec.get("passed")),
                                "err": r.get("error")}
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    # 每题分母 = 各配置观测到的最大 assert 条数
    total = {}
    for (t, tid), v in detail.items():
        total[tid] = max(total.get(tid, 0), v["n"])

    # 逐题明细 -> csv
    out_csv = os.path.join(RES, a.out + ".csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "task_id", "tests_passed", "tests_total", "frac", "acc_passed"])
        for (t, tid) in sorted(detail):
            v = detail[(t, tid)]
            tot = total.get(tid, 0)
            w.writerow([t, tid, v["passed"], tot, round(v["passed"] / tot, 4) if tot else "",
                        v["passed_flag"]])

    # 连续性检验: 逐题 frac, 与基线配对
    base = a.base
    summary = {}
    for t in tags:
        pts = [(detail[(t, tid)]["passed"] / total[tid], total[tid])
               for tid in tasks if tid in total and (t, tid) in detail and total[tid] > 0]
        if not pts:
            continue
        fracs = [x for x, _ in pts]
        m = len(fracs)
        mean_f = sum(fracs) / m
        sd = math.sqrt(sum((x - mean_f) ** 2 for x in fracs) / (m - 1)) if m > 1 else 0.0
        se = sd / math.sqrt(m) if m else 0.0
        mde = (1.959964 + 0.841621) * se
        summary[t] = {"n_tasks": m, "mean_frac": round(mean_f, 4),
                      "mean_tests_passed": round(sum(f * tot for f, tot in pts) / m, 2),
                      "sd_frac": round(sd, 4), "se_frac": round(se, 4),
                      "mde_pt_80pct": round(mde * 100, 2)}

    pairs = {}
    acc_pairs = {}
    if base in per:
        for t in tags:
            if t == base:
                continue
            common = [tid for tid in tasks
                      if (t, tid) in detail and (base, tid) in detail
                      and total.get(tid, 0) > 0]
            if len(common) < 30:
                continue
            d = [(detail[(t, tid)]["passed"] - detail[(base, tid)]["passed"]) / total[tid]
                 for tid in common]
            m = len(d)
            md = sum(d) / m
            sd = math.sqrt(sum((x - md) ** 2 for x in d) / (m - 1))
            se = sd / math.sqrt(m) if m else 0.0
            tv = md / se if se else 0.0
            pairs[f"{t}_vs_{base}"] = {
                "n_tasks": m, "mean_diff_pt": round(md * 100, 2), "sd_diff": round(sd, 4),
                "se_diff_pt": round(se * 100, 3), "t": round(tv, 3), "df": m - 1,
                "p_value": round(t_p_value(tv, m - 1), 5),
                "ci95_pt": [round((md - 1.959964 * se) * 100, 2),
                            round((md + 1.959964 * se) * 100, 2)],
                "mde_pt_80pct": round((1.959964 + 0.841621) * se * 100, 2)}

            # 同一批题、同一口径下的 acc 配对检验, 用于"指标分层"对照
            da = [int(bool(detail[(t, tid)]["passed_flag"]))
                  - int(bool(detail[(base, tid)]["passed_flag"])) for tid in common]
            mad = sum(da) / m
            sda = math.sqrt(sum((x - mad) ** 2 for x in da) / (m - 1))
            sea = sda / math.sqrt(m) if m else 0.0
            ta = mad / sea if sea else 0.0
            b01 = sum(1 for x in da if x == -1)   # 基线对/该配置错
            b10 = sum(1 for x in da if x == 1)    # 基线错/该配置对
            acc_pairs[f"{t}_vs_{base}"] = {
                "n_tasks": m, "acc_diff_pt": round(mad * 100, 2), "se_pt": round(sea * 100, 3),
                "t": round(ta, 3), "p_value": round(t_p_value(ta, m - 1), 5),
                "ci95_pt": [round((mad - 1.959964 * sea) * 100, 2),
                            round((mad + 1.959964 * sea) * 100, 2)],
                "mde_pt_80pct": round((1.959964 + 0.841621) * sea * 100, 2),
                "mcnemar_base_pass_only": b01, "mcnemar_cand_pass_only": b10}


    out = {
        "protocol": {"metric": "tests_passed (0..N asserts of HumanEval check())",
                     "baseline": base, "items": len(tasks), "harness": "e_f7_case.py, "
                     "one subprocess per (config, task), 15s timeout, asserts counted individually",
                     "merge": "hc1024 truncation reruns override by task_id"},
        "note": "连续指标与 acc 在同一批题上的配对检验并列, 用于 §4 功效声明的分层",
        "per_config": summary,
        "paired_t": pairs,
        "paired_t_acc": acc_pairs,
    }
    p_out = os.path.join(RES, a.out + ".json")
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n{'tag':24s} {'tasks':>5s} {'mean_frac':>9s} {'MDE(pt)':>8s}")
    for t, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean_frac"]):
        print(f"{t:24s} {s['n_tasks']:5d} {s['mean_frac']:9.4f} {s['mde_pt_80pct']:8.2f}")
    print("\npaired vs {}:".format(base))
    for k, v in sorted(pairs.items(), key=lambda kv: kv[1]["p_value"]):
        print(f"  {k:34s} diff={v['mean_diff_pt']:+6.2f}pt CI {v['ci95_pt']} "
              f"t={v['t']:+.2f} p={v['p_value']:.4f} MDE={v['mde_pt_80pct']}")
    print("\n同一批题的 acc 对照:")
    for k, v in sorted(acc_pairs.items(), key=lambda kv: kv[1]["p_value"]):
        print(f"  {k:34s} diff={v['acc_diff_pt']:+6.2f}pt CI {v['ci95_pt']} p={v['p_value']:.4f} "
              f"MDE={v['mde_pt_80pct']} b/c={v['mcnemar_base_pass_only']}/{v['mcnemar_cand_pass_only']}")
    print(f"\n[saved] {p_out}\n[saved] {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
