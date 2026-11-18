"""G2/G3 第0步: 只拉 GGUF 头 (Range 前 N MB), 读 arch / 张量命名, 判断专家布局与分档可行性。

    python e_g_gguf_header.py openbmb/MiniCPM-V-4.7-35B-A3B-gguf MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
    python e_g_gguf_header.py XingChen-AGI/Xing4.0-29B-A4B-GGUF xing4_0-29b-iq4_xs.gguf

第一参 = HF repo, 第二参 = 仓库内文件名, 第三参(可选) = 拉取字节数 (默认 16 MB)。
"""
import os
import struct
import subprocess
import sys
import tempfile

MS = "https://www.modelscope.cn/models/%s/resolve/master/%s"
HF = "https://hf-mirror.com/%s/resolve/main/%s"

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, \
    T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(13)
SCALAR = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
          T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
          T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d"}


class Cap(Exception):
    pass


class Buf:
    def __init__(self, b):
        self.b = b
        self.i = 0

    def take(self, n):
        if self.i + n > len(self.b):
            raise Cap()
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        n = self.u64()
        if n > 1 << 26:
            raise Cap()
        return self.take(n).decode("utf-8", "replace")


def skip_value(buf, vtype, keep=False):
    if vtype == T_STRING:
        return buf.string() if keep else (buf.string() and None)
    if vtype == T_ARRAY:
        et = buf.u32()
        n = buf.u64()
        if et == T_STRING:
            if n > 5_000_000:
                raise Cap()
            # 只保留极短的数组 (如 tokenizer 特殊标记), 长数组直接跳过
            if keep and n <= 64:
                return [buf.string() for _ in range(n)]
            for _ in range(n):
                buf.string()
            return None
        fmt = SCALAR.get(et)
        if fmt is None:
            raise Cap()
        size = struct.calcsize(fmt) * n
        if buf.i + size > len(buf.b):
            raise Cap()
        buf.i += size
        return None
    fmt = SCALAR.get(vtype)
    if fmt is None:
        raise Cap()
    return struct.unpack(fmt, buf.take(struct.calcsize(fmt)))[0]


KEEP_KV = {"general.architecture", "general.name", "general.file_type",
           "general.basename", "split.count", "split.no",
           "gemma4.expert_count", "minicpm.expert_count", "xing4.expert_count",
           "qwen35moe.expert_count", "minicpm3.expert_count",
           "tokenizer.chat_template", "qwen35moe.expert_used_count",
           "qwen35moe.embedding_length", "qwen35moe.block_count"}


def parse(raw):
    buf = Buf(raw)
    if buf.take(4) != b"GGUF":
        raise SystemExit("不是 GGUF")
    ver = buf.u32()
    n_tensors = buf.u64()
    n_kv = buf.u64()
    kv = {}
    for _ in range(n_kv):
        k = buf.string()
        vt = buf.u32()
        keep = k in KEEP_KV
        v = skip_value(buf, vt, keep=keep)
        if keep:
            kv[k] = v
    tensors = []
    for _ in range(n_tensors):
        name = buf.string()
        nd = buf.u32()
        dims = [buf.u64() for _ in range(nd)]   # GGUF v3 里张量维度是 uint64
        tt = buf.u32()
        off = buf.u64()
        tensors.append((name, dims, tt, off))
    return ver, n_tensors, kv, tensors


def fetch(repo, fname, nbytes):
    fd, path = tempfile.mkstemp(suffix=".gguf.part")
    os.close(fd)
    for url in (MS % (repo, fname), HF % (repo, fname)):
        cmd = ["curl.exe", "-sL", "--max-time", "600", "-r", f"0-{nbytes - 1}", "-o", path, url]
        if subprocess.run(cmd).returncode == 0 and os.path.getsize(path) >= min(nbytes, 4096):
            with open(path, "rb") as f:
                raw = f.read()
            os.unlink(path)
            return raw
    raise SystemExit("拉取头部失败")


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    dump = ""
    if "--dump" in argv:
        dump = argv[argv.index("--dump") + 1]
    if argv[1].lower().endswith(".gguf"):
        # 本地模式: 只读前 N MB 头部 (F16 权重文件太大, 不能整读)
        path = argv[1]
        nbytes = int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else 256 << 20
        with open(path, "rb") as f:
            raw = f.read(nbytes)
        print("local %s 读 %.1f MB" % (os.path.basename(path), len(raw) / 1e6))
    else:
        repo, fname = argv[1], argv[2]
        nbytes = int(argv[3]) if len(argv) > 3 else 16 << 20
        raw = fetch(repo, fname, nbytes)
        print("fetched %.1f MB" % (len(raw) / 1e6))
    try:
        ver, n_tensors, kv, tensors = parse(raw)
    except Cap:
        print("头部超出 %d MB, 请加大第三参" % (nbytes >> 20))
        return 1
    print("gguf v%d  n_tensors=%d  n_kv=%d" % (ver, n_tensors, len(kv)))
    for k in sorted(kv):
        print("  %-28s %s" % (k, kv[k]))
    names = [t[0] for t in tensors]
    print("tensor infos parsed: %d" % len(names))
    print("  前 8 个:")
    for n in names[:8]:
        print("    " + n)

    # 专家张量布局判定
    fused = [n for n in names if "gate_up_exps" in n]
    per_expert = [n for n in names if "exps." in n or "expert" in n.lower()]
    ffn = [n for n in names if "ffn" in n]
    print("  ffn 张量: %d 个" % len(ffn))
    for n in ffn[:12]:
        print("    " + n)
    print("  gate_up_exps(融合): %d 个 -> %s" % (len(fused), fused[:4]))
    print("  exps.<idx>(逐专家): %d 个 -> %s" % (len(per_expert), per_expert[:6]))
    if dump:
        with open(dump, "w", encoding="utf-8") as f:
            f.write("gguf v%d n_tensors=%d\n" % (ver, n_tensors))
            f.write("kv " + repr({k: kv[k] for k in kv if not k.startswith("tokenizer.chat")}) + "\n")
            for name, dims, tt, _off in tensors:
                f.write("%s\t%s\t%d\n" % (name, dims, tt))
        print("  dump -> %s" % dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
