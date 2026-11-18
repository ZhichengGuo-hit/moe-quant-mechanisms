"""E-F4 汇总: 3 轮 llama-bench 的 pp512 / tg1 中位数 + 极差。

输入: results/_f4_round{1,2,3}.json (llama-bench -o json)
输出: results/e_f4_prefill.json (含 protocol / llama.cpp commit / 模型体积与 oid)
"""
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LLAMA = os.path.abspath(os.path.join(HERE, "..", "llama.cpp"))

MODELS = [
    ("tcmoe_q4k", os.path.join(HERE, "..", "tcmoe", "out", "model-tcmoe-q4k.gguf"), None),
    ("ud_q4km", r"E:\CUDA11.8\model\Qwen3.6-35B-A3B\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", None),
    ("iq2xxs", r"E:\CUDA11.8\model\baselines\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf", None),
    ("ud_q3kxl", r"E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
     "a832b9689925f1bd335bbe985cdfb06c36bf2cf268f4f8f6eceafa3ceb515617"),
]


def git_commit():
    try:
        h = subprocess.run(["git", "-C", LLAMA, "log", "-1", "--format=%H %ci"],
                           capture_output=True, text=True, timeout=20)
        return h.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def load_rounds():
    rounds = []
    for i in (1, 2, 3):
        p = os.path.join(RES, f"_f4_round{i}.json")
        if not os.path.exists(p):
            print(f"[warn] missing {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            txt = f.read().strip()
        data = json.loads(txt)
        rounds.append(data)
    return rounds


def key_of(rec):
    return (os.path.basename(rec["model_filename"]),
            int(rec.get("n_prompt", 0)), int(rec.get("n_gen", 0)))


def main():
    rounds = load_rounds()
    if not rounds:
        print("no rounds", file=sys.stderr)
        sys.exit(1)

    acc = {}
    for rd in rounds:
        for rec in rd:
            acc.setdefault(key_of(rec), []).append(float(rec["avg_ts"]))

    out = {"protocol": {
        "tool": "llama-bench (build-tcmoe fork, type45 support)",
        "n_prompt": 512, "n_gen": 1, "reps_per_round": 1, "rounds": len(rounds),
        "threads": 16, "cpu_affinity": "0xff",
        "backend": "CPU only (llama-bench --list-devices = none; no CUDA device)",
        "interleave": "one process per round, 4 models back-to-back in that process",
    },
        "llama_cpp": {"commit": git_commit(),
                      "note": "local tcmoe patches on top (see git status)"},
        "models": [],
        "results": []}

    for tag, path, oid in MODELS:
        size = os.path.getsize(os.path.abspath(path)) if os.path.exists(path) else None
        out["models"].append({"tag": tag, "file": os.path.basename(path),
                              "size_bytes": size,
                              "size_gib": round(size / 2 ** 30, 3) if size else None,
                              "hf_lfs_oid": oid})

    base = {os.path.basename(p): t for t, p, _ in MODELS}
    for tag, path, oid in MODELS:
        name = os.path.basename(path)
        rec = {"tag": tag, "file": name}
        for (fn, np_, ng), vals in acc.items():
            if fn != name:
                continue
            lab = "pp512" if ng == 0 else ("tg1" if np_ == 0 else f"pp{np_}_tg{ng}")
            rec[lab] = {"median_tok_s": round(statistics.median(vals), 3),
                        "min": round(min(vals), 3), "max": round(max(vals), 3),
                        "range": round(max(vals) - min(vals), 3),
                        "n": len(vals), "raw": [round(v, 3) for v in vals]}
        out["results"].append(rec)

    p = os.path.join(RES, "e_f4_prefill.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"{'tag':12s} {'pp512 median':>14s} {'range':>8s} {'tg1 median':>12s} {'range':>8s}")
    for r in out["results"]:
        pp = r.get("pp512", {})
        tg = r.get("tg1", {})
        print(f"{r['tag']:12s} {pp.get('median_tok_s', 0):14.2f} {pp.get('range', 0):8.2f} "
              f"{tg.get('median_tok_s', 0):12.2f} {tg.get('range', 0):8.2f}")
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
