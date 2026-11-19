"""E-N1: Nemotron 四档 + Q6 锚 vs Qwen3.6 六档的 acc/CoT-位宽曲线与断崖判定。

只读 tcq_exp/results 的既有数据（C4 = Nemotron 5x100；E17 = Qwen 200 题，取前 100 对齐）。
判定规则见 PREREG.md（分析前冻结）。

用法:
    python n1_analyze.py
产物:
    results/n1_summary.json
    results/n1_curves.csv
    results/n1_curves.svg
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SRC = os.path.abspath(os.path.join(HERE, "..", "tcq_exp", "results"))
DATA = os.path.abspath(os.path.join(HERE, "..", "tcq_exp", "data"))
GGUF_PY = r"E:\CUDA11.8\train\llama.cpp\gguf-py"
MODEL_ROOT = r"E:\CUDA11.8\model"

NEMO_TAGS = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_K_M", "Q6_K"]
# Nemotron: 档位 -> GGUF（bpw 与 Qwen 同口径：文件字节 x 8 / 参数量实算）
NEMO_FILES = {
    "Q2_K":   os.path.join(MODEL_ROOT, "nemotron", "Nemotron-Cascade-2-30B-A3B.i1-Q2_K.gguf"),
    "Q3_K_S": os.path.join(MODEL_ROOT, "nemotron", "Nemotron-Cascade-2-30B-A3B.i1-Q3_K_S.gguf"),
    "Q3_K_M": os.path.join(MODEL_ROOT, "nemotron", "Nemotron-Cascade-2-30B-A3B.i1-Q3_K_M.gguf"),
    "Q4_K_M": os.path.join(MODEL_ROOT, "nemotron", "Nemotron-Cascade-2-30B-A3B.i1-Q4_K_M.gguf"),
    "Q6_K":   os.path.join(MODEL_ROOT, "nemotron", "Nemotron-Cascade-2-30B-A3B.i1-Q6_K.gguf"),
}
N_PAIR = 100
CLIFF_ACC = 0.15      # 相邻档 acc 跌幅 >= 15 pp
CLIFF_PN = 0.70       # 或 CoT 均值跌到 < 70%
SEED = 20261118
NBOOT = 1000

# Qwen3.6: E17 任务文件 + 对应 GGUF（bpw 由文件字节/参数量实算）
QWEN = [
    ("iq2xxs",   os.path.join(MODEL_ROOT, "baselines", "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"), "e17_iq2xxs_gsm8k.jsonl"),
    ("ud_q2kxl", os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),        "e17_ud_q2kxl_gsm8k.jsonl"),
    ("iq3xxs",   os.path.join(MODEL_ROOT, "baselines", "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"), "e17_iq3xxs_gsm8k.jsonl"),
    ("ud_q3kxl", os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"),        "e17_ud_q3kxl_gsm8k.jsonl"),
    ("ud_q4kxl", os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),        "e17_ud_q4kxl_gsm8k.jsonl"),
    ("q4km",     os.path.join(MODEL_ROOT, "Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"), "e17_q4km_gsm8k.jsonl"),
]


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((c - h) / d), float((c + h) / d)]


def mcnemar_exact(b, c):
    """双侧精确 McNemar。b/c 为两臂不一致的配对计数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 2 * P(X <= k), X ~ Bin(n, 0.5)
    from math import comb
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * p))


def boot_mean_ci(x, rng, nboot=NBOOT):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [0.0, 0.0]
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def behavior_stats(rows):
    cnt = {"reasoning": 0, "direct": 0, "truncated": 0, "loop": 0}
    for r in rows:
        b = r.get("behavior") or "truncated"
        cnt[b] = cnt.get(b, 0) + 1
    n = max(len(rows), 1)
    return {k: round(v / n, 4) for k, v in cnt.items()}, cnt


def parse_qtype_bpw(path):
    """bpw = 文件位数 / 参数量；参数量从 GGUF 头累加 n_elements（不含元数据）。"""
    sys.path.insert(0, GGUF_PY)
    from gguf import GGUFReader
    r = GGUFReader(path)
    n_params = int(sum(int(t.n_elements) for t in r.tensors))
    nbytes = os.path.getsize(path)
    return float(nbytes * 8 / n_params), n_params


def nemo_section(rng):
    rows = read_jsonl(os.path.join(SRC, "c4_nemotron_sweep.jsonl"))
    by_tag = {t: [] for t in NEMO_TAGS}
    for r in rows:
        if r.get("tag") in by_tag:
            by_tag[r["tag"]].append(r)
    data = {}
    for t in NEMO_TAGS:
        rs = sorted(by_tag[t], key=lambda x: int(x["task_id"].split("-")[1]))
        ok = [bool(r.get("acc")) for r in rs]
        pn = [int(r.get("predicted_n") or 0) for r in rs]
        bpw, n_params = parse_qtype_bpw(NEMO_FILES[t])
        beh, beh_n = behavior_stats(rs)
        data[t] = {
            "bpw": round(bpw, 3), "n_params": n_params,
            "bpw_source": "measured (bytes*8/params)",
            "n": len(rs), "acc": float(np.mean(ok)) if ok else None,
            "acc_ci95": wilson(int(sum(ok)), len(ok)),
            "acc_ok": ok, "pn": pn,
            "mean_pn": float(np.mean(pn)) if pn else None,
            "median_pn": float(np.median(pn)) if pn else None,
            "mean_pn_ci95": boot_mean_ci(pn, rng) if pn else None,
            "behavior": beh, "behavior_n": beh_n,
            "truncated_rate": beh.get("truncated", 0.0),
            "loop_rate": beh.get("loop", 0.0),
        }
    # 相邻差分
    gaps = []
    for a, b in zip(NEMO_TAGS[:-1], NEMO_TAGS[1:]):
        da = data[b]["acc"] - data[a]["acc"]
        ratio = (data[b]["mean_pn"] / data[a]["mean_pn"]) if data[a]["mean_pn"] else None
        gaps.append({"from": a, "to": b, "acc_delta": round(da, 4),
                     "pn_ratio": round(ratio, 4) if ratio else None,
                     "cliff": bool(da <= -CLIFF_ACC or (ratio is not None and ratio <= CLIFF_PN))})
    # Q2 vs Q3_K_S 配对
    q2 = data["Q2_K"]["acc_ok"]
    q3 = data["Q3_K_S"]["acc_ok"]
    b = sum(1 for x, y in zip(q2, q3) if x and not y)
    c = sum(1 for x, y in zip(q2, q3) if (not x) and y)
    pair = {"q2": q2, "q3": q3, "b": b, "c": c, "p": mcnemar_exact(b, c)}
    four = NEMO_TAGS[:4]
    accs = [data[t]["acc"] for t in four]
    ratios = [g["pn_ratio"] for g in gaps if g["to"] in four[1:]]
    cliff_at_q2 = next((g for g in gaps if g["from"] == "Q2_K"), None)
    if cliff_at_q2 and cliff_at_q2["cliff"] and min(accs[1:]) >= accs[0] * 0.97:
        verdict = "A_cliff"
    elif not any(g["cliff"] for g in gaps[:3]) and (max(accs) - min(accs)) < CLIFF_ACC \
            and all(r is None or r > CLIFF_PN for r in ratios):
        verdict = "B_gradual"
    else:
        verdict = "C_inconclusive"
    return {"quants": data, "gaps": gaps, "mcnemar_q2_q3ks": pair,
            "verdict": verdict, "criteria": {"cliff_acc": CLIFF_ACC, "cliff_pn": CLIFF_PN}}


def qwen_section(rng):
    out = {}
    for tag, gguf, taskfile in QWEN:
        tp = os.path.join(SRC, taskfile)
        if not (os.path.exists(tp) and os.path.exists(gguf)):
            out[tag] = {"skipped": "missing file"}
            continue
        rows = [r for r in read_jsonl(tp) if int(r["task_id"].split("-")[1]) < N_PAIR]
        rows = sorted(rows, key=lambda x: int(x["task_id"].split("-")[1]))
        ok = [bool(r.get("passed", r.get("acc"))) for r in rows]
        pn = [int((r.get("timings") or {}).get("predicted_n") or r.get("predicted_n") or 0)
              for r in rows]
        bpw, n_params = parse_qtype_bpw(gguf)
        out[tag] = {"bpw": round(bpw, 3), "n_params": n_params, "n": len(rows),
                    "acc": float(np.mean(ok)), "acc_ci95": wilson(int(sum(ok)), len(ok)),
                    "mean_pn": float(np.mean(pn)), "median_pn": float(np.median(pn)),
                    "mean_pn_ci95": boot_mean_ci(pn, rng)}
    return out


def svg_plot(nemo, qwen, path):
    # 双面板：左 acc–bpw，右 mean CoT–bpw
    W, H = 960, 420
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 50
    pw, ph = (W - pad_l - pad_r) / 2, H - pad_t - pad_b
    pts_n = [(nemo[t]["bpw"], nemo[t]["acc"]) for t in NEMO_TAGS]
    pts_q = [(v["bpw"], v["acc"]) for v in qwen.values() if "bpw" in v]
    xs = [p[0] for p in pts_n + pts_q]
    ys = [p[1] for p in pts_n + pts_q]
    x0, x1 = min(xs) - 0.3, max(xs) + 0.3
    y0, y1 = min(ys) - 0.02, max(ys) + 0.02
    pn_n = [(nemo[t]["bpw"], nemo[t]["mean_pn"]) for t in NEMO_TAGS]
    pn_q = [(v["bpw"], v["mean_pn"]) for v in qwen.values() if "bpw" in v]
    pys = [p[1] for p in pn_n + pn_q]
    py0, py1 = min(pys) * 0.9, max(pys) * 1.05

    def xy(x, y, ox, y0_, y1_):
        return (ox + (x - x0) / (x1 - x0) * pw,
                pad_t + (1 - (y - y0_) / (y1_ - y0_)) * ph)

    def panel(ox, pts, y0_, y1_, color, title):
        s = ['<text x="%.1f" y="20" font-size="13" text-anchor="middle">%s</text>'
             % (ox + pw / 2, title)]
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333"/>'
                 % (ox, pad_t + ph, ox + pw, pad_t + ph))
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333"/>'
                 % (ox, pad_t, ox, pad_t + ph))
        for i in range(5):
            y = pad_t + ph * i / 4
            v = y1_ - (y1_ - y0_) * i / 4
            s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ddd"/>'
                     % (ox, y, ox + pw, y))
            s.append('<text x="%.1f" y="%.1f" font-size="10" text-anchor="end">%.3g</text>'
                     % (ox - 4, y + 3, v))
        for x in sorted({round(p[0], 2) for p in pts}):
            px, _ = xy(x, y0_, ox, y0_, y1_)
            s.append('<text x="%.1f" y="%.1f" font-size="10" text-anchor="middle">%.1f</text>'
                     % (px, pad_t + ph + 14, x))
        d = " ".join("%.1f,%.1f" % xy(x, y, ox, y0_, y1_) for x, y in sorted(pts))
        s.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (color, d))
        for x, y in sorted(pts):
            px, py = xy(x, y, ox, y0_, y1_)
            s.append('<circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>' % (px, py, color))
        s.append('<text x="%.1f" y="%.1f" font-size="10" text-anchor="middle">bpw</text>'
                 % (ox + pw / 2, H - 12))
        return s

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
           'font-family="sans-serif">' % (W, H),
           '<rect width="100%" height="100%" fill="white"/>']
    svg += panel(pad_l, pts_n, y0, y1, "#c0392b", "Acc (red=Nemotron, blue=Qwen3.6)")
    svg += panel(pad_l, pts_q, y0, y1, "#2c6fbb", "")
    svg += panel(pad_l + pw + pad_r, pn_n, py0, py1, "#c0392b", "Mean CoT (predicted_n)")
    svg += panel(pad_l + pw + pad_r, pn_q, py0, py1, "#2c6fbb", "")
    svg.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def main():
    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(SEED)
    nemo = nemo_section(rng)
    qwen = qwen_section(rng)
    out = {
        "prereg": "PREREG.md",
        "seed": SEED, "nboot": NBOOT,
        "protocol_note": "Nemotron = C4 协议 (GSM8K 前 100, -ngl 20, 1024, 保留 CoT)；"
                         "Qwen = E17 协议 (GSM8K 200 取前 100, 1024)；仅同模型内趋势并置",
        "models": {"nemotron": "Nemotron-Cascade-2-30B-A3B (i1 系列, 31.58B)",
                   "qwen": "Qwen3.6-35B-A3B (34.66B)"},
        "nemotron": nemo,
        "qwen": qwen,
        "verdict": nemo["verdict"],
        "verdict_text": {
            "A_cliff": "Q2_K 处断崖且 Q3+ 恢复：不连续模式切换成立，M6 关闭",
            "B_gradual": "四档渐变：异常缩窄为低比特敏感，机制断言降级，M6 关闭",
            "C_inconclusive": "不连续证据不足：如实报告曲线，M6 以限定语关闭",
        }[nemo["verdict"]],
    }
    with open(os.path.join(RES, "n1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RES, "n1_curves.csv"), "w", encoding="utf-8") as f:
        f.write("model,tag,bpw,acc,acc_lo,acc_hi,mean_pn,median_pn\n")
        for t in NEMO_TAGS:
            d = nemo["quants"][t]
            f.write("nemotron,%s,%.3f,%.4f,%.4f,%.4f,%.1f,%.1f\n" % (
                t, d["bpw"], d["acc"], d["acc_ci95"][0], d["acc_ci95"][1],
                d["mean_pn"], d["median_pn"]))
        for t, d in qwen.items():
            if "bpw" in d:
                f.write("qwen36,%s,%.3f,%.4f,%.4f,%.4f,%.1f,%.1f\n" % (
                    t, d["bpw"], d["acc"], d["acc_ci95"][0], d["acc_ci95"][1],
                    d["mean_pn"], d["median_pn"]))
    svg_plot(nemo["quants"], qwen, os.path.join(RES, "n1_curves.svg"))
    print(json.dumps({"verdict": out["verdict"],
                      "gaps": nemo["gaps"],
                      "mcnemar_q2_q3ks": nemo["mcnemar_q2_q3ks"]},
                     ensure_ascii=False, indent=1))
    print("wrote", RES)


if __name__ == "__main__":
    main()
