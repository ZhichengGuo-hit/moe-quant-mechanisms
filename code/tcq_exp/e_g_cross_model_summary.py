"""E-G: 跨模型总表 (G1/G2/G3 各配置 x 三任务 acc + 体积 + 生成耗时)。

只聚合已落盘的 summary json 与 GGUF 实际体积, 不重跑任何评测。
输出 results/e_g_cross_model_summary.json, 供论文 §5.5 直接引用。

    python e_g_cross_model_summary.py
"""
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

import e_g1_eval_matrix as G1M  # noqa: E402
import e_g_eval_matrix as GGM  # noqa: E402

OUT = os.path.join(RES, "e_g_cross_model_summary.json")

TASKS = ("gsm8k", "humaneval", "gpqa")

ROLES = {
    "g1": {
        "A_qat": "官方 QAT (UD-Q4_K_XL)",
        "B_ud_q4kxl": "社区 UD-Q4_K_XL (基线)",
        "C_roleq4": "角色对齐: 专家 Q4_K + 非专家 Q5_K",
        "Cprime_roleq4_nx6": "角色对齐 + 非专家 Q6_K (比特梯度)",
        "D_roleq4_im": "C + imatrix",
        "R_roleq3_im": "专家 Q3_K + imatrix (低比特档)",
    },
    "g2": {
        "A_off_q4km": "官方 Q4_K_M (基线)",
        "Rofficial_mixq3im": "官方 mix_q3gud_im (官方 imatrix 混档)",
        "C_roleq4": "角色对齐: 专家 Q4_K + 非专家 Q5_K",
        "Clayer_route": "角色对齐 + 逐层路由分档",
        "D_roleq4_im": "C + imatrix",
        "R_roleq3_im": "专家 Q3_K + imatrix (低比特档)",
    },
    "g3": {
        "A_off_iq4xs": "官方 IQ4_XS (基线)",
        "B1_venastine_q4km": "社区 Venastine Q4_K_M",
        "B2_dealign_q4km": "社区 dealignai CRACK Q4_K_M",
        "C_roleq4": "角色对齐: 专家 Q4_K + 非专家 Q5_K",
        "D_roleq4_im": "C + imatrix",
        "R_roleq3_im": "专家 Q3_K + imatrix (低比特档)",
    },
}

GROUPS = [
    {
        "suite": "g1", "label": "G1", "model": "gemma-4-26B-A4B",
        "params": "26B / A4B", "summary": "e_g1_gemma4_summary.json",
        "vs_summary": None, "baseline": "B_ud_q4kxl",
        "configs": G1M.CONFIGS,
        "note": "融合三维专家张量, 容器不支持逐专家规则; C/C'/D/R 为角色对齐整张量分档",
    },
    {
        "suite": "g2", "label": "G2", "model": "MiniCPM-V-4.7-35B-A3B",
        "params": "35B / A3B", "summary": "e_g2_minicpm_summary.json",
        "vs_summary": "e_g2_minicpm_summary_vs_Rofficial_mixq3im.json",
        "baseline": "A_off_q4km", "configs": GGM.SUITES["g2"]["configs"],
        "note": "双基线: 官方 Q4_K_M 与官方 imatrix 混档; 另含逐层路由分档 C_layer",
    },
    {
        "suite": "g3", "label": "G3", "model": "Xing4.0-29B-A4B",
        "params": "29B / A4B", "summary": "e_g3_xing_summary.json",
        "vs_summary": "e_g3_xing_summary_vs_B1_venastine_q4km.json",
        "baseline": "A_off_iq4xs", "configs": GGM.SUITES["g3"]["configs"],
        "note": "含两个社区基线 (Venastine / dealignai) 与官方 IQ4_XS",
    },
]


def size_of(path):
    if not os.path.exists(path):
        return None
    return {"bytes": os.path.getsize(path),
            "gib": round(os.path.getsize(path) / float(1 << 30), 3)}


def load(p):
    with open(os.path.join(RES, p), encoding="utf-8") as f:
        return json.load(f)


def main():
    models, missing = [], []
    for g in GROUPS:
        s = load(g["summary"])
        per = {}
        for r in s["per_tag_task"]:
            per.setdefault(r["tag"], {})[r["task"]] = r
        diffs = {}
        for r in s["vs_baseline"]:
            diffs.setdefault(r["tag"], {})[r["task"]] = r
        rows = []
        for tag in sorted(per, key=lambda t: list(g["configs"]).index(t)):
            cfg = g["configs"][tag]
            path = cfg[0] if isinstance(cfg, (tuple, list)) else cfg
            ngl = cfg[1] if isinstance(cfg, (tuple, list)) else 99
            size = size_of(path)
            if size is None:
                missing.append("%s/%s" % (g["label"], tag))
            row = {
                "tag": tag,
                "role": ROLES[g["suite"]].get(tag, ""),
                "is_baseline": tag == g["baseline"],
                "file": os.path.basename(path),
                "ngl": ngl,
                "size_gib": size["gib"] if size else None,
                "size_bytes": size["bytes"] if size else None,
                "acc": {}, "n": {}, "passed": {}, "avg_gen_s": {},
                "avg_predicted_n": {},
                "vs_baseline": {},
            }
            for t in TASKS:
                r = per[tag].get(t)
                if not r:
                    missing.append("%s/%s/%s" % (g["label"], tag, t))
                    continue
                row["acc"][t] = r["acc"]
                row["n"][t] = r["n"]
                row["passed"][t] = r["passed"]
                row["avg_gen_s"][t] = r["avg_gen_s"]
                row["avg_predicted_n"][t] = r["avg_predicted_n"]
                d = diffs.get(tag, {}).get(t)
                if d:
                    row["vs_baseline"][t] = {
                        "acc_diff": round(d["acc_diff"], 4),
                        "boot_ci_lo": d["boot_ci_lo"],
                        "boot_ci_hi": d["boot_ci_hi"],
                        "mcnemar_p": round(d["mcnemar_p_exact"], 6),
                    }
            rows.append(row)
        models.append({
            "suite": g["label"], "model": g["model"], "params": g["params"],
            "baseline": g["baseline"], "note": g["note"],
            "summary_file": g["summary"], "vs_summary_file": g["vs_summary"],
            "rows": rows,
        })

    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "bin": "llama-upstream/build-cuda/bin",
            "ctx": 4096, "threads": 16, "n_predict": 1024, "block": 50,
            "cpus": "0-31", "temperature": 0.0, "top_k": 1, "seed": 1234,
            "thinking": "disabled (chat_template_kwargs.enable_thinking=false)",
            "tasks": {"gsm8k": 200, "humaneval": 164, "gpqa": 100},
            "scoring": "gsm8k 精确答案 / humaneval 抽取后单测隔离执行 / gpqa 单字母抽取",
            "gpu": "Quadro RTX 5000 16 GB, 混合推理 -ngl 按配置固定 (见各 suite 脚本)",
        },
        "task_names": {"gsm8k": "GSM8K (200)", "humaneval": "HumanEval (164)",
                       "gpqa": "GPQA-Diamond (100)"},
        "models": models,
        "g4_reference": {
            "status": "degraded_to_c (不做质量对比, 仅生态观察)",
            "source": "results/e_g4_smoke.json",
            "reason": "llama.cpp diffusion 白名单不含 clover_moe; AR 模式输出乱码; "
                      "transformers 路线需 >16 GB 显存",
            "retry_entry": "SGLang PR #46927 或 >=48 GB 显存硬件, 协议 block=32 / "
                           "low_confidence=0.95 / thinking 开",
        },
        "missing": missing,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[write]", OUT)
    for m in models:
        print("[%s] %-24s %d 配置" % (m["suite"], m["model"], len(m["rows"])))
        for r in m["rows"]:
            print("   %-22s %6.2f GiB  gsm8k=%.3f  humaneval=%.3f  gpqa=%.2f"
                  % (r["tag"], r["size_gib"] or 0, r["acc"].get("gsm8k", 0),
                     r["acc"].get("humaneval", 0), r["acc"].get("gpqa", 0)))
    if missing:
        print("[warn] 缺失:", missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
