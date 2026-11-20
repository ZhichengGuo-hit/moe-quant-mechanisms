#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_data_table.py —— 校验 DATA_TABLE.csv 中每条 result_files 路径是否真实存在于仓库。

用法:
    python check_data_table.py            # 校验结果文件
    python check_data_table.py --code     # 同时校验 code 列的脚本路径（相对 code/）
    python check_data_table.py --strict   # 存在缺失时以非零退出码结束（供 CI 使用）

约定:
  - result_files 为逗号分隔的仓库相对路径；
  - 以 "（" / "(" 开头的片段是「未随包」说明，跳过；
  - "—" / "-" / 空 表示该实验无独立结果文件，不算缺失；
  - 行内出现 N/A 同样跳过。
"""
import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "DATA_TABLE.csv")
CODE_DIR = os.path.join(ROOT, "code")


def split_paths(cell):
    """把 result_files 单元格拆成待校验路径列表。"""
    out = []
    if not cell:
        return out
    s = cell.strip()
    if s in ("—", "-", "N/A", ""):
        return out
    # 中文/英文括号内的说明文字不是路径
    s = re.sub(r"[（(][^）)]*[）)]", "", s)
    for piece in s.split(","):
        p = piece.strip().strip("`").strip()
        if not p or p in ("—", "-", "N/A"):
            continue
        # 过滤误留的说明尾巴（如 "8 MB"）
        if re.match(r"^\d", p):
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="store_true", help="同时校验 code 列的脚本")
    ap.add_argument("--strict", action="store_true", help="有缺失时返回非零退出码")
    args = ap.parse_args()

    if not os.path.exists(CSV_PATH):
        print("找不到 DATA_TABLE.csv")
        return 2

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    total = ok = 0
    missing = []
    dash_rows = []
    code_total = code_ok = 0
    code_missing = []

    for r in rows:
        rid = (r.get("id") or "").strip()
        cell = (r.get("result_files") or "").strip()
        paths = split_paths(cell)
        if not paths:
            if cell in ("—", "-", ""):
                dash_rows.append(rid)
            continue
        for p in paths:
            total += 1
            full = os.path.join(ROOT, p.replace("/", os.sep))
            if os.path.exists(full):
                ok += 1
            else:
                missing.append((rid, p))

        if args.code:
            for c in split_paths(r.get("code") or ""):
                if not c.endswith(".py") and not c.endswith(".md"):
                    continue
                code_total += 1
                cand = c if c.endswith(".md") else os.path.join("code", c)
                if os.path.exists(os.path.join(ROOT, cand.replace("/", os.sep))):
                    code_ok += 1
                else:
                    code_missing.append((rid, c))

    print("DATA_TABLE.csv 共 %d 行" % len(rows))
    print("结果文件路径：%d 条，命中 %d，缺失 %d" % (total, ok, len(missing)))
    if missing:
        for rid, p in missing:
            print("  MISSING  %-8s %s" % (rid, p))

    if args.code:
        print("代码路径：%d 条，命中 %d，缺失 %d" % (code_total, code_ok, len(code_missing)))
        for rid, c in code_missing:
            print("  MISSING  %-8s %s" % (rid, c))

    print("无结果文件（—）的行：%d%s"
          % (len(dash_rows), ("  " + ", ".join(dash_rows)) if dash_rows else ""))

    bad = len(missing) + (len(code_missing) if args.code else 0)
    if bad:
        print("\n结论：存在 %d 条缺失路径。" % bad)
        return 1 if args.strict else 0
    print("\n结论：全部路径有效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
