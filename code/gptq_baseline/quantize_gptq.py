"""E-GPTQ 量化脚本（在具备 gptqmodel 的环境运行）。

    python quantize_gptq.py --model E:\\CUDA11.8\\model\\Qwen3.6-35B-A3B \
        --out out/qwen36-3bit-gptq --bits 3 --group 128 --damp 0.01 --calib 128

校准集：tcq_exp/data/{math,code,general}_calib.jsonl 按域轮转取 128 条，
seqlen 2048。16 GB 显存下需逐层 offload（GPTQModel 自动处理 device_map）。
"""
import argparse
import json
import os
import sys
import time


def log(m):
    print("%s %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def load_calib(n):
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.abspath(os.path.join(here, "..", "tcq_exp", "data"))
    pools = []
    for d in ["math", "code", "general"]:
        p = os.path.join(data, "%s_calib.jsonl" % d)
        with open(p, encoding="utf-8") as f:
            pools.append([json.loads(l)["text"] for l in f if l.strip()])
    out = []
    i = 0
    while len(out) < n:
        out.append(pools[i % 3][(i // 3) % len(pools[i % 3])])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--desc-act", action="store_true")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except Exception as e:  # noqa: BLE001
        log("gptqmodel 不可用: %r" % (e,))
        log("见 PREREG.md §4：本机因 pypcre/PCRE2 构建失败而阻塞")
        return 2

    qc = QuantizeConfig(bits=args.bits, group_size=args.group,
                        damp_percent=args.damp, desc_act=args.desc_act, sym=True)
    log("load model %s" % args.model)
    model = GPTQModel.load(args.model, qc)
    texts = load_calib(args.calib)
    log("quantize: bits=%d group=%d damp=%.3f calib=%d seqlen=%d"
        % (args.bits, args.group, args.damp, len(texts), args.seqlen))
    t0 = time.time()
    model.quantize(texts, batch_size=args.batch, seqlen=args.seqlen)
    log("quantize done %.1f min" % ((time.time() - t0) / 60))
    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)
    log("saved %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
