"""B5: "稀疏性-分离度"曲线补点。

原计划: Mixtral-8x7B(top-2, K/N=25%) + DeepSeek-V3 类(256 专家 top-8, 3.1%).
本机不可得(94GB + 671B 级, 且 CPU bf16 前向不可行), 改用**受控替代**:
在 Qwen3-30B-A3B 上重放路由并对每个 k∈{2,4,8,16,32} 重算 H1 (B1 已产出),
得到同一模型内 K/N ∈ {1.6%, 3.1%, 6.25%, 12.5%, 25%} 的 5 个受控点 —— 控制
了模型/数据/层数, 只变 K/N, 因果解释强于跨模型拼接。

同时给出跨模型锚点(B6): 主模型 K/N=3.125%、gemma4 与 Qwen3-30B K/N=6.25%,
用于检查"同 K/N 下不同模型的 lift 量级"。
输出: results/b5_sparsity.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T

PAIRS = ("math-code", "math-general", "code-general")


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float)))
    ry = np.argsort(np.argsort(np.asarray(y, float)))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def main():
    with open(os.path.join(T.RES_DIR, "b1_topk.json"), encoding="utf-8") as f:
        b1 = json.load(f)
    with open(os.path.join(T.RES_DIR, "b6_lift.json"), encoding="utf-8") as f:
        b6 = json.load(f)

    ks = b1["ks"]
    sweep = {}
    for k in ks:
        r = b1["results"][str(k)]
        sweep[str(k)] = {"k_over_N": r["k_over_N"], "h3_acc": r["h3"]["acc"],
                         "h1": {p: {"jaccard": r["h1"][p]["jaccard"],
                                    "lift": r["h1"][p]["lift"],
                                    "spearman": r["h1"][p]["spearman"]} for p in PAIRS},
                         "alloc_j": r["alloc"]["math-code"]["selection_jaccard"]}

    trend = {}
    for p in PAIRS:
        xs = [sweep[str(k)]["k_over_N"] * 100 for k in ks]
        ys = [sweep[str(k)]["h1"][p]["lift"] for k in ks]
        trend[p] = {"spearman_lift_vs_K_over_N": spearman(xs, ys),
                    "lift_at_minK": ys[0], "lift_at_maxK": ys[-1],
                    "delta_lift": ys[-1] - ys[0]}
    trend["h3_acc"] = {"spearman_h3_vs_K_over_N":
                       spearman([sweep[str(k)]["k_over_N"] for k in ks],
                                [sweep[str(k)]["h3_acc"] for k in ks]),
                       "min": min(sweep[str(k)]["h3_acc"] for k in ks),
                       "max": max(sweep[str(k)]["h3_acc"] for k in ks)}

    anchors = {}
    for mname, m in b6["models"].items():
        kn = m["K"] / m["E"]
        anchors[mname] = {"K_over_N": kn,
                          "lift": {p: m["pairs"][p]["lift"] for p in PAIRS},
                          "jaccard": {p: m["pairs"][p]["jaccard_pooled"] for p in PAIRS}}

    out = {"source": "B1 受控 k 扫 (Qwen3-30B-A3B 重放路由) + B6 跨模型锚点",
           "substitution": ("Mixtral-8x7B / DeepSeek-V3 类模型本机不可得(下载体积与 "
                            "CPU bf16 前向不可行), 以同模型 k 扫替代"),
           "ks": ks, "sweep": sweep, "trend": trend, "anchors": anchors,
           "rand_base": b1["rand_base"], "max_len": b1["max_len"],
           "n_calib": b1["n_calib"], "n_eval": b1["n_eval"]}

    p = os.path.join(T.RES_DIR, "b5_sparsity.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[B5] K/N 曲线 (Qwen3-30B 受控 k 扫):")
    for k in ks:
        s = sweep[str(k)]
        print(f"  K/N={s['k_over_N']*100:.2f}%  " + " ".join(
            f"{p2}: J={s['h1'][p2]['jaccard']:.3f}/lift={s['h1'][p2]['lift']:.2f}"
            for p2 in PAIRS) + f"  H3={s['h3_acc']:.3f}  allocJ={s['alloc_j']:.3f}")
    print(f"[B5] trend {json.dumps(trend, ensure_ascii=False)}")
    print(f"[b5] saved {p}")


if __name__ == "__main__":
    main()
