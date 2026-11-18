"""E-F6: 双面板 Pareto 汇总 (零机时, 纯读 results)。

左图: E18 keep 网格 (2bit 底座 + INT4 行补丁, 三域 eval64 mean NLL)
      横轴 keep_both GiB (7.97 / 8.72 / 9.47 / 10.22 / 11.72, 另加 base3 15.47 / int4 15.58 参考)
右图: relaxed 任务锚点 (GSM8K 200 + HumanEval 164, -n 1024), 横轴 = 单域激活工作集 GiB
      阴影 = 10-15 GiB / 0.935-0.965 饱和带

用法:
    python e_f6_pareto.py
输出:
    results/e_f6_pareto.png   (双面板图)
    results/e_f6_pareto.csv   (图上每个点的原始数值)
"""
import csv
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

W, H = 1760, 890
BG = (252, 252, 250)
FG = (30, 30, 34)
GRID = (228, 228, 232)
BLUE = (38, 92, 168)
ORANGE = (214, 106, 30)
GREEN = (34, 132, 86)
GREY = (150, 150, 158)
BAND = (255, 236, 210)
BAND_EDGE = (226, 178, 120)

# relaxed 锚点: tag -> (工作集 GiB, 显示名, x 抖动)
ANCHORS = [
    ("iq2xxs", 10.02, "IQ2_XXS", 0.0),
    ("ud_q2kxl", 11.45, "UD-Q2_K_XL", 0.0),
    ("iq3xxs", 12.30, "IQ3_XXS", 0.0),
    ("tcmoe_nl_math", 14.57, "NL-math", -0.55),
    ("tcmoe_nl_global", 14.57, "NL-global", 0.0),
    ("tcmoe_nl_code", 14.57, "NL-code", 0.55),
    ("ud_q3kxl", 15.69, "UD-Q3_K_XL", 0.0),
    ("ud_q4kxl", 20.83, "UD-Q4_K_XL", 0.0),
    ("q4km", 22.13, "Q4_K_M", 0.0),
]


def font(size, bold=False):
    for p in ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
              "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001
                pass
    return ImageFont.load_default()


def acc_from(tag, task, expect=None):
    """合并 hc1024 截断补跑与原始全量结果 (同 task_id 以补跑为准)。

    expect: 完整题量 (gsm8k 200 / humaneval 164); 未跑满的点直接丢弃, 免得出图中途值。
    返回 (acc, n, n_merged, src) 或 None。
    """
    base = os.path.join(RES, f"e17_{tag}_{task}.jsonl")
    hc = os.path.join(RES, f"e17_{tag}_hc1024_{task}.jsonl")
    recs = {}
    if os.path.exists(base):
        with open(base, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                recs[r["task_id"]] = bool(r.get("passed"))
    n_merged = 0
    if os.path.exists(hc):
        with open(hc, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r["task_id"] in recs:
                    recs[r["task_id"]] = bool(r.get("passed"))
                    n_merged += 1
    if not recs:
        return None
    n = len(recs)
    if expect and n < expect:
        return None
    return sum(recs.values()) / n, n, n_merged, ("merged" if n_merged else "base")


def axis(d, box, xr, yr, xlabel, ylabel, title, f_small, f_lab, f_title, yfmt=None,
         nxt=5, nxstep=True, xticks=None):
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=GREY, width=1)
    for i in range(nxt + 1):
        y = y0 + (y1 - y0) * i / nxt
        d.line([(x0, y), (x1, y)], fill=GRID)
        v = yr[0] + (yr[1] - yr[0]) * (nxt - i) / nxt
        t = f"{v:.2f}" if yfmt is None else yfmt(v)
        d.text((x0 - 10, y), t, font=f_small, fill=FG, anchor="rm")
    if xticks:
        for v in xticks:
            x = x0 + (v - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)
            d.line([(x, y0), (x, y1)], fill=GRID)
            d.text((x, y1 + 10), f"{v:.0f}", font=f_small, fill=FG, anchor="ma")
    else:
        nx = 6
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * i / nx
            v = xr[0] + (xr[1] - xr[0]) * i / nx
            d.text((x, y1 + 10), f"{v:.0f}" if xr[1] - xr[0] > 6 else f"{v:.1f}",
                   font=f_small, fill=FG, anchor="ma")
    d.text(((x0 + x1) / 2, y1 + 42), xlabel, font=f_lab, fill=FG, anchor="ma")
    d.text((x0, y0 - 74), f"y: {ylabel}", font=f_lab, fill=FG, anchor="la")
    d.text((x0, y0 - 40), title, font=f_title, fill=FG, anchor="la")


def px(box, xr, yr, x, y):
    x0, y0, x1, y1 = box
    return (x0 + (x - xr[0]) / (xr[1] - xr[0]) * (x1 - x0),
            y0 + (yr[1] - y) / (yr[1] - yr[0]) * (y1 - y0))


def main():
    f_small = font(17)
    f_lab = font(21)
    f_title = font(24, bold=True)
    f_ann = font(16)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    rows = []
    d.text((30, 22), "E-F6  Pareto: compression volume vs quality (relaxed protocol, -n 1024)",
           font=font(30, bold=True), fill=FG)

    # ---------------- 左图: E18 keep 网格 ----------------
    with open(os.path.join(RES, "pareto.json"), encoding="utf-8") as f:
        pj = json.load(f)
    cfgs = pj["configs"]
    frontier = set(pj["frontier"])

    lbox = (150, 165, 800, 640)
    xr = (7.0, 17.0)
    yr = (1.3, 2.35)
    axis(d, lbox, xr, yr,
         "stored volume (GiB, keep_both)", "mean NLL (3 domains x 64 seq)",
         "A. E18 keep grid: 2-bit base + INT4 row patches", f_small, f_lab, f_title)

    # 全域选择器序列 (frontier 主干)
    glob = sorted([c for c in cfgs if c.get("selector") == "global"],
                  key=lambda c: c["volume"]["keep_both"])
    pts = [px(lbox, xr, yr, c["volume"]["keep_both"], c["nll_mean"]) for c in glob]
    d.line(pts, fill=BLUE, width=3)
    for c, p in zip(glob, pts):
        d.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=BLUE)
        lab = f"{c['volume']['keep_both']:.2f}"
        d.text((p[0], p[1] - 14), lab, font=f_ann, fill=BLUE, anchor="mb")

    # task-specific 选择器 (灰点, 说明同体积下的分域分歧)
    for c in cfgs:
        if c.get("selector") in ("task_math", "task_code"):
            p = px(lbox, xr, yr, c["volume"]["keep_both"], c["nll_mean"])
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=GREY)

    # 参考点 base3 / int4
    for c in cfgs:
        if c["name"] in ("base3", "int4"):
            p = px(lbox, xr, yr, c["volume"]["keep_both"], c["nll_mean"])
            d.rectangle([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=GREEN)
            d.text((p[0] + 12, p[1]), f"{c['name']} ({c['base_bits']}-bit)",
                   font=f_ann, fill=GREEN, anchor="lm")

    d.text((150, 724), "blue = global selector (frontier spine); grey = task-specific patch; "
                       "green = 3-bit / INT4 reference", font=f_ann, fill=GREY)
    d.text((150, 752), "keep_both = 2-bit base kept in full + patch rows stored again "
                       "(runtime double-container volume)", font=f_ann, fill=GREY)

    for c in cfgs:
        rows.append({"panel": "A_e18", "name": c["name"],
                     "volume_gib": c["volume"]["keep_both"], "mean_nll": round(c["nll_mean"], 4),
                     "nll_math": round(c["nll"]["math"], 4), "nll_code": round(c["nll"]["code"], 4),
                     "nll_general": round(c["nll"]["general"], 4),
                     "selector": c.get("selector", ""), "base_bits": c["base_bits"]})

    # ---------------- 右图: relaxed 锚点 ----------------
    rbox = (1000, 165, 1650, 640)
    xr2 = (9.5, 23.5)
    yr2 = (0.86, 0.99)
    axis(d, rbox, xr2, yr2, "single-domain working set (GiB)", "accuracy",
         "B. Relaxed task anchors (GSM8K 200 / HumanEval 164)", f_small, f_lab, f_title,
         xticks=[10, 12, 14, 16, 18, 20, 22])

    b0 = px(rbox, xr2, yr2, 10.0, 0.965)
    b1 = px(rbox, xr2, yr2, 15.0, 0.935)
    d.rectangle([b0[0], min(b0[1], b1[1]), b1[0], max(b0[1], b1[1])],
                fill=BAND, outline=BAND_EDGE, width=2)

    best_c, best_h = None, None
    for tag, ws, label, jit in ANCHORS:
        g = acc_from(tag, "gsm8k", 200)
        h = acc_from(tag, "humaneval", 164)
        x = ws + jit
        dy = {"NL-math": -13, "NL-global": 13}.get(label, 0)
        if g:
            p = px(rbox, xr2, yr2, x, g[0])
            d.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=ORANGE,
                      outline=FG if g[0] >= 0.955 else None)
            d.text((p[0] + 10, p[1] + dy), f"{g[0]:.3f}", font=f_ann, fill=ORANGE, anchor="lm")
            rows.append({"panel": "B_anchor", "name": tag, "label": label,
                         "working_set_gib": ws, "task": "gsm8k", "acc": round(g[0], 4),
                         "n": g[1], "merged": g[2], "src": g[3]})
            if best_c is None or g[0] > best_c[0]:
                best_c = (g[0], label, ws)
        if h:
            p = px(rbox, xr2, yr2, x, h[0])
            d.polygon([(p[0], p[1] - 8), (p[0] - 8, p[1] + 6), (p[0] + 8, p[1] + 6)],
                      outline=BLUE, fill=None, width=2)
            d.text((p[0] + 10, p[1] + 2 - dy), f"{h[0]:.3f}", font=f_ann, fill=BLUE, anchor="lm")
            rows.append({"panel": "B_anchor", "name": tag, "label": label,
                         "working_set_gib": ws, "task": "humaneval", "acc": round(h[0], 4),
                         "n": h[1], "merged": h[2], "src": h[3]})
            if best_h is None or h[0] > best_h[0]:
                best_h = (h[0], label, ws)

    # 域内格子的低分锚点 (tcmoe_code HE / tcmoe_math) 只作说明, 不入表
    d.text((1000, 736), "circle = GSM8K (relaxed -n 1024), triangle = HumanEval; "
                       "ring marks >= 0.955", font=f_ann, fill=GREY)
    d.text((1000, 764), "shaded = 10-15 GiB / 0.935-0.965 saturation band; "
                       "NL-* three selectors share 14.57 GiB", font=f_ann, fill=GREY)

    if best_c:
        d.text((1650, 800), f"best GSM8K: {best_c[1]} {best_c[0]:.3f} @ {best_c[2]:.2f} GiB",
               font=f_lab, fill=ORANGE, anchor="ra")
    if best_h:
        d.text((1650, 830), f"best HumanEval: {best_h[1]} {best_h[0]:.3f} @ {best_h[2]:.2f} GiB",
               font=f_lab, fill=BLUE, anchor="ra")

    out_png = os.path.join(RES, "e_f6_pareto.png")
    img.save(out_png)
    out_csv = os.path.join(RES, "e_f6_pareto.csv")
    keys = ["panel", "name", "label", "task", "base_bits", "selector", "volume_gib",
            "working_set_gib", "mean_nll", "nll_math", "nll_code", "nll_general",
            "acc", "n", "merged", "src"]
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[saved] {out_png}")
    print(f"[saved] {out_csv}  ({len(rows)} rows)")
    for r in rows:
        if r["panel"] == "B_anchor":
            print(f"  {r['label']:12s} {r['task']:9s} {r['acc']:.4f} "
                  f"n={r['n']} merged={r['merged']} ws={r['working_set_gib']}")


if __name__ == "__main__":
    main()
