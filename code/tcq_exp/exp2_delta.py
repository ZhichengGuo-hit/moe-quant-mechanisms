"""实验2: 抽样专家的直接 ΔL 测量 (任务条件敏感度的金标准验证)。
对每个抽样专家: 用 BF16 权重重量化为 INT2 写入 COW 映射 -> 三域 NLL -> 恢复。
ΔL(e,D) = NLL_D(e被INT2化) - NLL_D(基线INT4)
输出: results/delta_matrix.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import tcq_lib as T

N_EVAL_SEQ = 8
TOP_PER_DOMAIN = 8      # 每域全局 top-8 敏感专家
N_RANDOM = 3            # 另取 3 个"被使用过"的随机专家作对照


def pick_experts(sens, usage):
    """每域全局 top-k 敏感专家 + 随机专家(限被路由过), 去重 -> [(layer,e,tag)]"""
    s = np.array(sens)      # [L,E,3]
    u = np.array(usage)
    E = s.shape[1]
    picked, seen = [], set()
    for di, dom in enumerate(T.DOMAINS):
        score = np.where(u[:, :, di] > 0, s[:, :, di], -1.0)
        flat = np.argsort(-score.ravel())
        cnt = 0
        for fi in flat:
            layer, e = divmod(int(fi), E)
            if score[layer, e] <= 0:
                break
            if (layer, e) in seen:
                continue
            seen.add((layer, e))
            picked.append((layer, e, f"top-{dom}"))
            cnt += 1
            if cnt >= TOP_PER_DOMAIN:
                break
    rng = np.random.default_rng(42)
    used = np.nonzero(u.sum(-1) > 0)
    cand = [(int(a), int(b)) for a, b in zip(*used) if (int(a), int(b)) not in seen]
    while len([p for p in picked if p[2].startswith("rand")]) < N_RANDOM and cand:
        i = int(rng.integers(0, len(cand)))
        layer, e = cand.pop(i)
        seen.add((layer, e))
        picked.append((layer, e, "rand"))
    return picked


def main():
    with open(os.path.join(T.RES_DIR, "sens_matrix.json"), encoding="utf-8") as f:
        sens = json.load(f)
    picked = pick_experts(sens["s"], sens["usage"])
    print(f"[pick] {len(picked)} experts: " +
          ", ".join(f"L{l}E{e}({tag})" for l, e, tag in picked), flush=True)

    eng = T.build_engine()
    tok = eng["tok"]
    patcher = T.ExpertPatcher(eng["store"])
    wmap = T.load_weight_map()

    evals = {}
    for di, dom in enumerate(T.DOMAINS):
        rows = T.load_domain(dom, "eval")[:N_EVAL_SEQ]
        evals[dom] = [r["text"] for r in rows]

    base = {}
    for dom in T.DOMAINS:
        t0 = time.time()
        tot, n, per = T.eval_nll(eng["model"], tok, evals[dom], max_len=64, batch_size=8)
        base[dom] = per
        print(f"[base] {dom} nll={per:.4f} ({time.time()-t0:.0f}s, {n} tok)", flush=True)

    rows_out = []
    for li, (layer, e, tag) in enumerate(picked):
        w_gu, w_dn = T.read_expert_bf16(wmap, T.MOE_MODEL_PATH, layer, e)
        pq, sq = T.quantize_bits(w_gu, 2)
        pd, sd = T.quantize_bits(w_dn, 2)
        patcher.set_expert(layer, e, {"gu_q": pq, "gu_s": sq, "dn_q": pd, "dn_s": sd})
        try:
            res = {"layer": layer, "expert": e, "tag": tag, "nll": {}, "delta": {}}
            for dom in T.DOMAINS:
                tot, n, per = T.eval_nll(eng["model"], tok, evals[dom],
                                         max_len=64, batch_size=8)
                res["nll"][dom] = per
                res["delta"][dom] = per - base[dom]
            rows_out.append(res)
            print(f"[{li+1}/{len(picked)}] L{layer}E{e} {tag} "
                  f"dNLL=" + " ".join(f"{d}:{res['delta'][d]:+.4f}" for d in T.DOMAINS),
                  flush=True)
        finally:
            patcher.restore_expert(layer, e)

    out = {"n_evals": N_EVAL_SEQ, "base_nll": base, "experts": rows_out,
           "picked": [{"layer": l, "expert": e, "tag": t} for l, e, t in picked]}
    p = os.path.join(T.RES_DIR, "delta_matrix.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[delta] saved {p}", flush=True)


if __name__ == "__main__":
    main()
