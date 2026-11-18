"""int2_ext 构建/加载 helper: 2bit 密排底座 + INT4 补丁 混合 FFN 扩展 (A1)。

与 int3_ext 共用同一份 C++ 源码 (coserving/int3_ext.cpp), 仅以不同模块名编译,
避免覆盖既有 int3_ext.pyd (E15 复现依赖它)。首次编译 30-90s。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "coserving", "int3_ext.cpp")
_BUILD = os.path.join(_HERE, "int2_ext_build")
_EXT = None


def load_ext(threads: int = 16):
    global _EXT
    if _EXT is not None:
        return _EXT
    os.makedirs(_BUILD, exist_ok=True)
    import torch
    torch.set_num_threads(threads)
    import importlib.util
    pyd = os.path.join(_BUILD, "int2_ext.pyd")
    if os.path.exists(pyd):
        spec = importlib.util.spec_from_file_location("int2_ext", pyd)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT
    import sys
    exe_dir = os.path.dirname(sys.executable)
    os.environ["PATH"] = exe_dir + os.pathsep + os.path.join(exe_dir, "Scripts") + \
        os.pathsep + os.environ.get("PATH", "")
    from torch.utils.cpp_extension import load
    _EXT = load(name="int2_ext", sources=[_SRC],
                extra_cflags=["/O2", "/arch:AVX2", "/utf-8"],
                build_directory=_BUILD, verbose=True)
    return _EXT


if __name__ == "__main__":
    ext = load_ext()
    print("[int2_ext] loaded:", ext.moe_ffn_mixed2_batch.__doc__)
