"""E-F10b 判定：imatrix 加权共享码本的 HumanEval 代码侧修复是否成立。

输入：
  results/e17_<new_tag>_humaneval.jsonl            （新跑的两组）
  results/e17_<new_tag>_hc1024_humaneval.jsonl     （若有截断补跑，按 task_id 覆盖）
  results/e17_tcmoe_nl_{code,global}_humaneval.jsonl（对应 nl 基线，同样合并 hc1024）

逐题配对后输出 results/e_f10b_he.json：
  - 每配置 acc、与对应 nl 基线的 diff（pt）
  - McNemar b/c 与精确二项 p；同口径的配对 t 检验（复用 e_f7 的 betai/t_p_value）
  - verdict: fix_confirmed（≥+2pt）/ partial（0~2pt）/ regression（<0）
  - interpretation: 未成立时的归因（码本家族 × 加权交互）

用法：
  python e_f10b_judge.py
  python e_f10b_judge.py --new-code x --new-global y --out _tmp_e_f10b_he.json
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

from e_f7_tests_passed import t_p_value  # noqa: E402  仓库内现成的配对检验实现

PASS_PT = 2.0


def load_tag(tag, res=RES):
    """读 full jsonl，再用 hc1024 覆盖（与 e_f7_tests_passed.load_runs 同口径）。"""
    out = {}
    for suffix in ("_humaneval.jsonl", "_hc1024_humaneval.jsonl"):
        path = os.path.join(res, f"e17_{tag}{suffix}")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                out[rec["task_id"]] = rec
    return out


def mcnemar_exact(b, c):
    """McNemar 精确二项检验（双侧），无需 scipy。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * float(tail) / float(2 ** n))


def verdict_of(diff_pt):
    if diff_pt >= PASS_PT:
        return "fix_confirmed"
    if diff_pt >= 0.0:
        return "partial"
    return "regression"


def compare(new_tag, base_tag, res=RES):
    new = load_tag(new_tag, res)
    base = load_tag(base_tag, res)
    if not new:
        raise SystemExit(f"缺少新结果: e17_{new_tag}_humaneval.jsonl")
    if not base:
        raise SystemExit(f"缺少基线结果: e17_{base_tag}_humaneval.jsonl")

    common = sorted(set(new) & set(base))
    if not common:
        raise SystemExit(f"{new_tag} 与 {base_tag} 无共同题")
    n_new_pass = sum(int(bool(new[t].get("passed"))) for t in new)
    n_base_pass = sum(int(bool(base[t].get("passed"))) for t in base)
    acc_new = n_new_pass / len(new)
    acc_base = n_base_pass / len(base)
    diff_pt = (acc_new - acc_base) * 100

    # 配对：b = 基线对/新错，c = 基线错/新对
    b = c = 0
    deltas = []
    for t in common:
        nb = int(bool(base[t].get("passed")))
        nn = int(bool(new[t].get("passed")))
        if nb and not nn:
            b += 1
        elif nn and not nb:
            c += 1
        deltas.append(nn - nb)
    m = len(deltas)
    md = sum(deltas) / m
    sd = math.sqrt(sum((x - md) ** 2 for x in deltas) / (m - 1)) if m > 1 else 0.0
    se = sd / math.sqrt(m) if m else 0.0
    tv = md / se if se else 0.0
    p_t = t_p_value(tv, m - 1) if m > 1 else 1.0
    p_mcn = mcnemar_exact(b, c)
    verdict = verdict_of(diff_pt)

    return {
        "tag": new_tag,
        "baseline": base_tag,
        "n_new": len(new),
        "n_base": len(base),
        "n_common": m,
        "missing_in_new": sorted(set(base) - set(new)),
        "extra_in_new": sorted(set(new) - set(base)),
        "acc": round(acc_new, 4),
        "baseline_acc": round(acc_base, 4),
        "diff_pt": round(diff_pt, 2),
        "mcnemar": {"b_base_pass_new_fail": b, "c_base_fail_new_pass": c,
                    "p_exact": round(p_mcn, 5)},
        "paired_t": {"mean_diff_pt": round(md * 100, 2),
                     "se_pt": round(se * 100, 3), "t": round(tv, 3),
                     "df": m - 1, "p_value": round(p_t, 5),
                     "ci95_pt": [round((md - 1.959964 * se) * 100, 2),
                                 round((md + 1.959964 * se) * 100, 2)]},
        "verdict": verdict,
    }


INTERPRETATION = (
    "判定未达 fix_confirmed（<+2pt）时，增益不足归因于『码本家族 × 加权交互』："
    "容器只允许每 (layer, proj) 一张 4 级共享表，而 imatrix 加权目标把拟合质量"
    "倾斜给高能量输入通道；per-matrix 的最优码本差异无法被共享表吸收，4 级离散"
    "化又截断了加权收益，两者交互使代码侧 HumanEval 的修复停留在噪声量级。"
)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-code", default="tcmoe_q2nl_im_code")
    ap.add_argument("--new-global", default="tcmoe_q2nl_im_global")
    ap.add_argument("--base-code", default="tcmoe_nl_code")
    ap.add_argument("--base-global", default="tcmoe_nl_global")
    ap.add_argument("--out", default=os.path.join(RES, "e_f10b_he.json"))
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = {}
    for key, new_tag, base_tag in (
            ("code", args.new_code, args.base_code),
            ("global", args.new_global, args.base_global)):
        cfg[key] = compare(new_tag, base_tag)
        print(f"[judge] {new_tag} acc={cfg[key]['acc']:.4f} vs {base_tag} "
              f"{cfg[key]['baseline_acc']:.4f} diff={cfg[key]['diff_pt']:+.2f}pt "
              f"b/c={cfg[key]['mcnemar']['b_base_pass_new_fail']}/"
              f"{cfg[key]['mcnemar']['c_base_fail_new_pass']} "
              f"p_t={cfg[key]['paired_t']['p_value']:.4f} -> {cfg[key]['verdict']}",
              flush=True)

    out = {
        "experiment": "E-F10b",
        "protocol": {
            "task": "humaneval (164)",
            "metric": "pass@1 (unit tests, e17_task_eval.py 判分)",
            "merge": "full jsonl + hc1024 truncation reruns override by task_id",
            "baseline_line": {"tcmoe_nl_code": 0.8902, "tcmoe_nl_global": 0.9085,
                              "iq2xxs_humaneval": 0.9390},
            "verdict_rule": {
                "fix_confirmed": "diff >= +2.0pt vs 对应 nl 基线",
                "partial": "0 <= diff < +2.0pt",
                "regression": "diff < 0",
            },
            "tests": ["McNemar exact binomial (b=base pass/new fail, c=base fail/new pass)",
                      "paired t on per-task 0/1 deltas (e_f7 betai/t_p_value)"],
        },
        "configs": cfg,
        "verdict": cfg["code"]["verdict"],
        "verdict_per_config": {k: v["verdict"] for k, v in cfg.items()},
        "interpretation": INTERPRETATION,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[judge] verdict={out['verdict']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
