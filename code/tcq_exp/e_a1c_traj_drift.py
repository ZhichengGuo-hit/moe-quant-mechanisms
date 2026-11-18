"""E-A1c: 轨迹重放保真 (零环境依赖).

FP16 参考模型对一条手写多轮工具调用轨迹逐轮打分, 各量化配置在相同上下文上
对参考模型的助手输出逐 token 比较 next-token 分布, 得到:
  每轮 top-1 一致率 / KLD / JSD 曲线 (vs BF16 参考), 以及首次偏离轮次。

轨迹不用生成, 直接手写 (用户方案允许 "手写工具调用序列"), 只做前向打分;
打分协议与 E-F9 完全一致: /completion + n_probs=k + post_sampling_probs=false。

    python e_a1c_traj_drift.py [--tokens 24] [--configs iq2xxs,ud_q2kxl,...]
    python e_a1c_traj_drift.py --smoke
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e_f9_kld as F9  # noqa: E402  复用 server/打分/指标实现

RES = os.path.join(HERE, "results")
MODEL_ROOT = r"E:\CUDA11.8\model"
REF = os.path.join(MODEL_ROOT, "e20", "qwen36_f16.gguf")
TCMOE = os.path.join(HERE, "..", "tcmoe", "out", "model-tcmoe-q4k-nl.gguf")

CONFIGS = {
    "iq2xxs": {"path": os.path.join(MODEL_ROOT, "baselines", "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"),
               "domain": None},
    "ud_q2kxl": {"path": os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
                 "domain": None},
    "tcq_nl_global": {"path": os.path.abspath(TCMOE), "domain": "global"},
    "tcq_nl_math": {"path": os.path.abspath(TCMOE), "domain": "math"},
    "base2": {"path": os.path.abspath(TCMOE), "domain": "base"},
    "ud_q4km": {"path": os.path.join(MODEL_ROOT, "Qwen3.6-35B-A3B",
                                     "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"), "domain": None},
}

SYSTEM = ("你是行程助手。可用工具: get_weather(city) / search_trains(from,to,date) / "
          "book_seat(train_no,seat)。需要外部信息时, 先只输出一行 JSON 工具调用; "
          "收到工具结果后再继续。")
# (role, content) —— 助手轮为固定文本, 即"手写工具调用序列"
TRACE = [
    ("user", "帮我查北京今天天气, 再找明天上午北京到上海最早的高铁, 并订一个靠窗座位。"),
    ("assistant", '{"tool":"get_weather","city":"北京"}'),
    ("tool", '{"temp_c":18,"condition":"晴","wind":"西北风3级"}'),
    ("assistant", '{"tool":"search_trains","from":"北京","to":"上海","date":"2026-11-12"}'),
    ("tool", '{"trains":[{"no":"G1","depart":"07:00","arrive":"11:28"},'
             '{"no":"G5","depart":"08:00","arrive":"12:30"}]}'),
    ("assistant", '{"tool":"book_seat","train_no":"G1","seat":"window"}'),
    ("tool", '{"status":"ok","train_no":"G1","seat":"01A","coach":"3"}'),
    ("assistant", "已订好明天 07:00 的 G1 次高铁, 3 车厢靠窗 01A; 北京今天晴, 18℃, "
                  "建议 06:20 前到北京南站。"),
    ("user", "如果出发当天下雨, 提醒我带伞。"),
    ("assistant", "已记录: 出发当天若降水概率超过 50%, 我会在 06:00 提醒你带伞。"),
]


def render(port, msgs):
    """用 server 自己的 chat 模板把 messages 渲染成 prompt 字符串。"""
    try:
        r = F9.post_json("http://127.0.0.1:%d/apply-template" % port,
                         {"messages": [{"role": r, "content": c} for r, c in msgs]}, timeout=60)
        p = r.get("prompt")
        if p:
            return p
    except Exception:  # noqa: BLE001
        pass
    out = []
    for role, content in msgs:
        out.append("<|im_start|>%s\n%s<|im_end|>\n" % (role, content))
    out.append("<|im_start|>assistant\n")
    return "".join(out)


def assistant_turns():
    """返回 [(轮号, 该轮之前的 messages, 助手文本)]"""
    turns, msgs = [], [("system", SYSTEM)]
    n = 0
    for role, content in TRACE:
        if role == "assistant":
            n += 1
            turns.append((n, list(msgs), content))
        msgs.append((role, content))
    return turns


def load_model(path, port, log_name):
    log = os.path.join(RES, log_name)
    t0 = time.time()
    proc, fh = F9.start_server(path, port, 16, 8192, log, 2400)
    print("[load] %s %.1f min" % (os.path.basename(path), (time.time() - t0) / 60), flush=True)
    return proc, fh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--tokens", type=int, default=24, help="每轮打分 token 数")
    ap.add_argument("--k", type=int, default=64, help="n_probs top-k")
    ap.add_argument("--out", default="e_a1c_traj_drift.json")
    ap.add_argument("--smoke", action="store_true", help="只跑 1 配置 8 token")
    args = ap.parse_args()
    names = args.configs.split(",")
    if args.smoke:
        names, args.tokens = ["iq2xxs"], 8
    os.makedirs(RES, exist_ok=True)

    ref_proc = ref_fh = None
    tgt_proc = tgt_fh = None
    out = {"protocol": {"ref": os.path.basename(REF), "k": args.k, "tokens_per_turn": args.tokens,
                        "metrics": "KLD/JSD/top1 on top-k partition (same as E-F9)"},
           "turns": [], "configs": {}}
    try:
        ref_proc, ref_fh = load_model(REF, 18131, "e_a1c_ref.server.log")
        info = F9.server_info(18131)
        n_vocab = int(info.get("n_vocab") or F9.DEFAULT_N_VOCAB)
        print("[ref] n_vocab=%d ftype=%s" % (n_vocab, info.get("ftype")), flush=True)

        # 参考侧每轮每个位置的 top-k 只算一次
        ref_pos = {}
        for turn, ctx, text in assistant_turns():
            prompt = render(18131, ctx)
            ids = F9.tokenize(18131, prompt)
            suffix = F9.tokenize(18131, text)
            n = min(args.tokens, len(suffix))
            rows = []
            for i in range(n):
                idx = len(ids) + i
                p_top = F9.score_position(18131, ids + suffix, idx, args.k, True, None)
                rows.append({"target": suffix[i], "p": p_top})
            ref_pos[turn] = {"ctx_ids": ids, "suffix": suffix, "n": n, "rows": rows}
            print("[ref] turn %d scored %d pos (ctx=%d)" % (turn, n, len(ids)), flush=True)
            out["turns"].append({"turn": turn, "ctx_tokens": len(ids),
                                 "assistant_tokens": len(suffix),
                                 "scored_tokens": n})

        for name in names:
            cfg = CONFIGS[name]
            tgt_proc, tgt_fh = load_model(cfg["path"], 18132, "e_a1c_%s.server.log" % name)
            per_turn, first_drift = [], None
            for turn, _ctx, _text in assistant_turns():
                r = ref_pos[turn]
                ids = r["ctx_ids"] + r["suffix"]
                klds, jsds, tops, cov_r, cov_m = [], [], [], [], []
                for i, row in enumerate(r["rows"]):
                    idx = len(r["ctx_ids"]) + i
                    q_top = F9.score_position(18132, ids, idx, args.k, True, cfg["domain"])
                    m = F9.position_metrics(row["p"], q_top, row["target"], n_vocab)
                    klds.append(m["kld"]); jsds.append(m["jsd"]); tops.append(m["top1_agree"])
                    cov_r.append(m["cov_ref"]); cov_m.append(m["cov_model"])
                rec = {"turn": turn, "n_pos": len(tops),
                       "kld": sum(klds) / len(klds), "jsd": sum(jsds) / len(jsds),
                       "top1": sum(tops) / len(tops),
                       "cov_ref": sum(cov_r) / len(cov_r), "cov_model": sum(cov_m) / len(cov_m)}
                per_turn.append(rec)
                print("[%s] turn %d kld=%.4f jsd=%.4f top1=%.3f" %
                      (name, turn, rec["kld"], rec["jsd"], rec["top1"]), flush=True)
                if first_drift is None and (rec["top1"] < 0.95 or rec["kld"] > 0.05):
                    first_drift = turn
            out["configs"][name] = {"path": os.path.basename(cfg["path"]),
                                    "domain": cfg["domain"], "per_turn": per_turn,
                                    "first_drift_turn": first_drift}
            F9.stop_server(tgt_proc, tgt_fh)
            tgt_proc = tgt_fh = None
            with open(os.path.join(RES, args.out), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    finally:
        F9.stop_server(tgt_proc, tgt_fh)
        F9.stop_server(ref_proc, ref_fh)

    path = os.path.join(RES, args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if not args.smoke:
        open(os.path.join(RES, ".a1c_done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    print("[done] %s" % path, flush=True)


if __name__ == "__main__":
    main()
