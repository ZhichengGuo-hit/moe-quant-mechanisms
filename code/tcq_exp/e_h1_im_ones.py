#!/usr/bin/env python
"""E-H1: 生成“全 1” imatrix。

b8746 的 llama-quant 对 IQ2/IQ3 族硬性要求 --imatrix（无则 throw），
因此 im-off 臂用全 1 imatrix 实现：权重=1 与“不提供 imatrix”的默认权重在数值上等价，
只绕过硬校验。文件结构与 im_global.gguf 完全一致（同名同 shape，in_sum2/counts 均为 1）。
"""
import os
import sys

import numpy as np

GGUF_PY = r"E:\CUDA11.8\train\llama.cpp\gguf-py"
sys.path.insert(0, GGUF_PY)
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

IM_GLOBAL = r"E:\CUDA11.8\model\e20\im_global.gguf"
OUT = r"E:\CUDA11.8\model\e20\im_ones.gguf"


def main():
    src = GGUFReader(IM_GLOBAL)
    w = GGUFWriter(OUT, "imatrix")
    w.add_uint32("imatrix.chunk_count", 1)
    w.add_uint32("imatrix.chunk_size", 512)
    w.add_array("imatrix.datasets", ["ones-imatrix (E-H1 im-off)"])
    n = 0
    for t in src.tensors:
        name = t.name
        if name.endswith(".in_sum2") or name.endswith(".counts"):
            data = np.ones(tuple(t.shape), dtype=np.float32)
            w.add_tensor(name, data, raw_dtype=GGMLQuantizationType.F32)
            n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s  tensors=%d  %.1f MB" % (OUT, n, os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
