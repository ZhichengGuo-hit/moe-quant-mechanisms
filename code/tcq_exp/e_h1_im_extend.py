#!/usr/bin/env python
"""E-H1: 扩展 imatrix 覆盖到 blk.40（MTP）与 blk.40.nextn.*

背景: im_global.gguf 只含 blk.0-39（2040 条 = 40 层 × 51 张量 × 4 后缀），
b8746 的 llama-quantize 对 IQ2/IQ3 族要求每张量都有 imatrix，缺 blk.40 即 bail out
（rc=1、产物截断、GGUF magic 无效）。

做法: 不改动 blk.0-39 的任何数值，只为缺失的 blk.40/nextn 张量补条目：
  - 角色匹配时复用 blk.39 同名后缀条目（MTP 层结构同末层）
  - nextn.eh_proj (4096×2048) 复用 blk.39.attn_output；(enorm/hnorm/shared_head_norm) 复用 blk.39.attn_norm
  - 兜底：按目标张量 ne0*ne2*ne3 生成全 1
产物: model\e20\im_global2.gguf（im-on）、model\e20\im_ones2.gguf（im-off，全 1）

注意: blk.40/nextn 只影响 MTP 投机头，不参与本论文所有评测的前向计算；
其 imatrix 来源在 results\e_h1_im_extend.json 中如实记录。
"""
import json
import os
import sys

import numpy as np

GGUF_PY = r"E:\CUDA11.8\train\llama.cpp\gguf-py"
sys.path.insert(0, GGUF_PY)
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

SRC = r"E:\CUDA11.8\model\e20\qwen36_f16.gguf"
IM_IN = r"E:\CUDA11.8\model\e20\im_global.gguf"
OUT_GLOBAL = r"E:\CUDA11.8\model\e20\im_global2.gguf"
OUT_ONES = r"E:\CUDA11.8\model\e20\im_ones2.gguf"
META = r"E:\CUDA11.8\train\tcq_exp\results\e_h1_im_extend.json"
SUFFIXES = [".in_sum", ".in_sum2", ".counts", ".stats"]

# nextn 子张量 -> blk.39 同名角色
NEXTN_DONOR = {
    "nextn.eh_proj.weight": "attn_output.weight",
    "nextn.enorm.weight": "attn_norm.weight",
    "nextn.hnorm.weight": "attn_norm.weight",
    "nextn.shared_head_norm.weight": "attn_norm.weight",
}


def donor_for(name):
    """blk.40.xxx -> blk.39.xxx；blk.40.nextn.yyy -> blk.39.<NEXTN_DONOR>；其它 -> None"""
    if not name.startswith("blk.40."):
        return None
    rest = name[len("blk.40."):]
    if rest.startswith("nextn."):
        k = rest[len("nextn."):]
        if k in NEXTN_DONOR:
            return "blk.39." + NEXTN_DONOR[k]
        return None
    return "blk.39." + rest


def main():
    src = GGUFReader(SRC)
    im = GGUFReader(IM_IN)

    fields = im.fields
    chunk_count = int(fields["imatrix.chunk_count"].contents())
    chunk_size = int(fields["imatrix.chunk_size"].contents())
    datasets = [str(x) for x in fields["imatrix.datasets"].contents()]

    entries = {}   # (base, suffix) -> (shape, ndarray)
    for t in im.tensors:
        for suf in SUFFIXES:
            if t.name.endswith(suf):
                base = t.name[: -len(suf)]
                entries[(base, suf)] = (tuple(int(x) for x in t.shape),
                                        np.array(t.data, dtype=np.float32))
                break

    have = {base for (base, _s) in entries}
    report = {"src": SRC, "im_in": IM_IN, "chunk_count": chunk_count,
              "chunk_size": chunk_size, "datasets": datasets,
              "n_entries_in": len(entries), "added": {}, "ones_fallback": []}

    # 需要补的张量: 源模型中存在但 im 缺失，且 2D+ / *.weight
    need = []
    for t in src.tensors:
        if t.name in have:
            continue
        if not t.name.endswith("weight") or len(t.shape) < 2:
            continue
        need.append(t.name)

    added_shapes = {}
    for name in need:
        dname = donor_for(name)
        d_ok = dname is not None and dname in have
        for suf in SUFFIXES:
            key = (name, suf)
            if d_ok and (dname, suf) in entries:
                shape, data = entries[(dname, suf)]
                entries[key] = (shape, data.copy())
                report["added"][name + suf] = "copy:" + dname + suf
            else:
                # 兜底全 1；形状与 llama-quant 期望一致: ne0 x (ne2*ne3)
                t = next(x for x in src.tensors if x.name == name)
                dims = [int(v) for v in t.shape] + [1, 1, 1, 1]
                nmat = dims[2] * dims[3]
                shape = (1, nmat) if suf == ".counts" else (dims[0], nmat)
                entries[key] = (shape, np.ones(shape, dtype=np.float32))
                report["ones_fallback"].append(name + suf)
        added_shapes[name] = [entries[(name, s)][0] for s in SUFFIXES]

    # 写 im_global2
    def write(path, force_ones, note):
        w = GGUFWriter(path, "imatrix")
        w.add_uint32("imatrix.chunk_count", chunk_count if not force_ones else 1)
        w.add_uint32("imatrix.chunk_size", chunk_size)
        w.add_array("imatrix.datasets", datasets if not force_ones else [note])
        for (base, suf), (shape, data) in entries.items():
            arr = np.ones(shape, dtype=np.float32) if force_ones else data
            w.add_tensor(base + suf, np.ascontiguousarray(arr),
                         raw_dtype=GGMLQuantizationType.F32)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print("wrote %s  tensors=%d  %.1f MB" % (path, len(entries),
                                                 os.path.getsize(path) / 1e6))

    # 校验覆盖：量化候选张量是否全部有 in_sum2+counts，且 in_sum2 元素数 == ne0*ne2*ne3
    miss = [n for n in need if (n, ".in_sum2") not in entries or (n, ".counts") not in entries]
    size_bad = []
    for n in need:
        t = next(x for x in src.tensors if x.name == n)
        want = int(t.shape[0]) * (int(t.shape[2]) if len(t.shape) > 2 else 1) * (int(t.shape[3]) if len(t.shape) > 3 else 1)
        got = int(np.prod(entries[(n, ".in_sum2")][0]))
        if got != want:
            size_bad.append((n, got, want))
    report["missing_after"] = miss
    report["size_mismatch"] = size_bad
    report["n_entries_out"] = len(entries)
    os.makedirs(os.path.dirname(META), exist_ok=True)
    json.dump(report, open(META, "w"), indent=1, ensure_ascii=False)
    print("added tensors=%d  missing_after=%d  size_mismatch=%d" % (
        len(need), len(miss), len(size_bad)))
    for n, g, w in size_bad:
        print("  SIZE BAD %-42s got=%d want=%d" % (n, g, w))
    if miss or size_bad:
        print("ABORT: imatrix 覆盖不完整，未写文件")
        return 1

    write(OUT_GLOBAL, False, None)
    write(OUT_ONES, True, "ones-imatrix (E-H1 im-off, extended to blk.40)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
