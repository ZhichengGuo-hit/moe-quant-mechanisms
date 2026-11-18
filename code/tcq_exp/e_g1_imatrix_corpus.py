"""E-G1: 把 math / code / general 三域校准集合并成 llama-imatrix 的纯文本语料。

llama-imatrix 通过 -f/--file 读入文本, 内部整文件读成一条字符串后按 token 切 chunk,
所以每行一段 text 即可。这里按域轮转交错排列 (避免某个域整体排在前面造成偏差),
并把每条内的换行/连续空白压成单空格。

输出: data/g1_calib.txt
用法: python e_g1_imatrix_corpus.py
      python e_g1_imatrix_corpus.py --per-domain 8 --dry-run
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DEFAULT_OUT = os.path.join(DATA, "g1_calib.txt")


def read_texts(path, limit):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(" ".join(json.loads(line)["text"].split()))
            if limit and len(texts) >= limit:
                break
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--domains", default="math,code,general")
    ap.add_argument("--per-domain", type=int, default=0, help="每域最多取多少条; 0=全部")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="只打印, 不写文件")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    per_dom = {}
    for d in domains:
        path = os.path.join(args.data_dir, f"{d}_calib.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {d}: 文件不存在 {path}")
            continue
        per_dom[d] = read_texts(path, args.per_domain)

    # 轮转交错: 依次取每个域的下一条, 直到取完
    merged = []
    idx = 0
    while True:
        added = False
        for d in per_dom:
            if idx < len(per_dom[d]):
                merged.append(per_dom[d][idx])
                added = True
        if not added:
            break
        idx += 1

    chars = sum(len(t) for t in merged)
    total_bytes = sum(len(t.encode("utf-8")) for t in merged)
    for d, texts in per_dom.items():
        print(f"[read] {d:8s}: {len(texts)} 条, {sum(len(t) for t in texts):,} 字符")
    print(f"[merge] 共 {len(merged)} 条, {chars:,} 字符, {total_bytes:,} UTF-8 字节")
    print(f"[out] {args.out}")

    if args.dry_run:
        print("[dry-run] 前 3 条预览:")
        for t in merged[:3]:
            print("  " + t[:100])
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in merged:
            f.write(t + "\n")
    print(f"[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
