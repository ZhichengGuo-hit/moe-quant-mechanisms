"""E20: 删除 --prune-layers 遗留的 MTP 元数据并重写量化文件。

prune 只改了 block_count, 保留了 41 项的 recurrent_layers 和 nextn_predict_layers,
导致 llama.cpp 加载报 "recurrent_layers has wrong array length". 这里按官方 UD 基线的
口径 (block_count=40, 无 nextn/recurrent 键) 重写 KV; 张量按原顺序、原类型搬运。

量化张量的 data 是字节形状 (如 (256,512,528) uint8), 不能走 add_tensor 的
quant_shape_from_byte_shape 路径, 故手工 add_tensor_info + write_tensor_data。
"""
import os
import sys

sys.path.insert(0, r"E:\CUDA11.8\train\llama.cpp\gguf-py")
import gguf  # noqa: E402
import numpy as np  # noqa: E402

DROP_KEYS = {
    "general.architecture",  # GGUFWriter 构造时已写入
    "qwen35moe.attention.recurrent_layers",
    "qwen35moe.nextn_predict_layers",
}


def patch(src: str, dst: str, block_count: int = 40) -> None:
    r = gguf.GGUFReader(src)
    arch = str(r.get_field("general.architecture").contents())
    w = gguf.GGUFWriter(dst, arch, use_temp_file=False)
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key in DROP_KEYS:
            continue
        val = field.contents()
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        if key == "qwen35moe.block_count":
            val = block_count
            vtype = gguf.GGUFValueType.UINT32
            sub = None
        w.add_key_value(key, val, vtype, sub_type=sub)
    for t in r.tensors:
        w.add_tensor_info(t.name, tuple(reversed(t.shape)), np.dtype("float32"), t.n_bytes,
                          raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(np.asarray(t.data))
    w.close()
    print(f"patched {os.path.basename(src)} -> {os.path.basename(dst)} "
          f"({os.path.getsize(dst)/2**30:.2f} GiB, {len(r.tensors)} tensors)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        patch(sys.argv[1], sys.argv[2])
    else:
        d = r"E:\CUDA11.8\model\e20"
        for dom in ("math", "code", "general", "global"):
            src = os.path.join(d, f"iq2xxs_{dom}.gguf")
            dst = os.path.join(d, f"iq2xxs_{dom}_fix.gguf")
            if not os.path.exists(src) or os.path.exists(dst):
                continue
            patch(src, dst)
