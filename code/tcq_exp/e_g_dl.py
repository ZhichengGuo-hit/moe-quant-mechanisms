"""E-G2/G3 权重下载 (ModelScope 主源, hf-mirror 兜底; 分块 + 断点续传 + 单实例锁)。

每个文件按 CHUNK 切块, 块完成后编号追加到 <out>.parts 并 fsync;
块齐了才写 <out>.done, 重启只补缺失块。

    python e_g_dl.py --only g2
    python e_g_dl.py --only g3
    python e_g_dl.py --list
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.request

MS = "https://www.modelscope.cn/models/%s/resolve/master/%s"
HF = "https://hf-mirror.com/%s/resolve/main/%s"

LOG = r"E:\CUDA11.8\train\_tmp_g_dl.log"
CHUNK = 16 << 20
WORKERS = int(os.environ.get("E_G_WORKERS", "12"))
_lk = threading.Lock()


def say(m):
    line = "%s %s" % (time.strftime("%H:%M:%S"), m)
    with _lk:
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def size_of(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            if n:
                return int(n)
    except Exception:
        pass
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=30) as r:
        cr = r.headers.get("Content-Range") or ""
        if "/" in cr:
            return int(cr.split("/")[-1])
        n = r.headers.get("Content-Length")
        if n:
            return int(n)
    raise IOError("cannot determine size")


def pick_lock(dst):
    os.makedirs(dst, exist_ok=True)
    lock = os.path.join(dst, ".dl.lock")
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            os.kill(pid, 0)
            say("another downloader alive (pid=%d) in %s, exit" % (pid, dst))
            return False
        except Exception:
            pass
    open(lock, "w").write(str(os.getpid()))
    return True


def fetch_chunk(url, path, idx, n, prog):
    a = idx * CHUNK
    b = min(n - 1, a + CHUNK - 1)
    for attempt in range(6):
        got = 0
        h = None
        try:
            h = open(path, "r+b")
            req = urllib.request.Request(url, headers={
                "Range": "bytes=%d-%d" % (a, b), "User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=90) as r:
                h.seek(a)
                while got < b - a + 1:
                    buf = r.read(min(1 << 20, b - a + 1 - got))
                    if not buf:
                        raise IOError("eof %d/%d" % (got, b - a + 1))
                    h.write(buf)
                    got += len(buf)
                    with _lk:
                        prog[0] += len(buf)
            h.flush()
            os.fsync(h.fileno())
            h.close()
            return True
        except Exception:
            try:
                h.close()
            except Exception:
                pass
            if got:
                a += got
            time.sleep(2 + attempt * 3)
    return False


def get_file(repo, fname, dst, url_fmt=MS):
    url = url_fmt % (repo, fname)
    out = os.path.join(dst, fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out + ".done"):
        say("  skip %s (done)" % fname)
        return True
    n = size_of(url)
    if not (os.path.exists(out) and os.path.getsize(out) == n):
        with open(out, "wb") as f:
            f.seek(n - 1)
            f.write(b"\0")
    side = out + ".parts"
    done = set()
    if os.path.exists(side):
        with open(side) as f:
            done = set(int(x) for x in f.read().split() if x.strip().isdigit())
    total = (n + CHUNK - 1) // CHUNK
    todo = [i for i in range(total) if i not in done]
    need = sum(min(n, (i + 1) * CHUNK) - i * CHUNK for i in todo)
    say("  %s %.2f GiB, 块 %d/%d 已有, 待下 %.2f GiB"
        % (os.path.basename(fname), n / 2**30, total - len(todo), total, need / 2**30))
    if not todo:
        open(out + ".done", "w").write("ok")
        say("  OK %s (already complete)" % fname)
        return True

    q = queue.Queue()
    for i in todo:
        q.put(i)
    prog = [0]
    failed = []
    t0 = time.time()

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            if fetch_chunk(url, out, i, n, prog):
                with _lk, open(side, "a") as sf:
                    sf.write("%d\n" % i)
                    sf.flush()
                    os.fsync(sf.fileno())
            else:
                with _lk:
                    failed.append(i)

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for t in ths:
        t.start()
    last = 0
    while any(t.is_alive() for t in ths):
        time.sleep(20)
        cur = prog[0]
        sp = (cur - last) / 20 / 1e6
        last = cur
        left = (need - cur) / 2**30 / max(sp / 1024, 0.1)
        say("  %s 本次 %.2f/%.2f GiB  %.1f MB/s  %.0f min left"
            % (os.path.basename(fname), cur / 2**30, need / 2**30, sp, left))
    for t in ths:
        t.join()

    with open(side) as f:
        done = set(int(x) for x in f.read().split() if x.strip().isdigit())
    missing = [i for i in range(total) if i not in done]
    if missing:
        say("  WARN %s 缺 %d 块 (例: %s), 下次续跑" % (fname, len(missing), missing[:5]))
        return False
    open(out + ".done", "w").write("ok")
    say("  OK %s %.2f GiB in %.1f min" % (fname, n / 2**30, (time.time() - t0) / 60))
    return True


def get_small(repo, dst, names):
    import shutil
    d = os.path.join(dst, "safetensors")
    os.makedirs(d, exist_ok=True)
    for f in names:
        out = os.path.join(d, f)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        for u in (MS % (repo, f), HF % (repo, f)):
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(u, headers={"User-Agent": "curl/8"}), timeout=60) as r:
                    with open(out, "wb") as w:
                        shutil.copyfileobj(r, w)
                if os.path.getsize(out) > 0:
                    say("  small OK %s (%d B)" % (f, os.path.getsize(out)))
                    break
            except Exception as e:
                say("  small FAIL %s %r" % (f, e))


SMALL = ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
         "chat_template.jinja", "model.safetensors.index.json", "preprocessor_config.json",
         "processor_config.json"]

# (group, repo, fname, 角色)
JOBS = [
    # ---- G2 MiniCPM-V 4.7 35B-A3B: F16 建 C/C_layer/D/R; A/R_official 直接评测
    ("g2", "openbmb/MiniCPM-V-4.7-35B-A3B-gguf", "MiniCPM-V-4_7-35B-A3B-F16.gguf", "f16"),
    ("g2", "openbmb/MiniCPM-V-4.7-35B-A3B-gguf", "MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf", "A"),
    ("g2", "openbmb/MiniCPM-V-4.7-35B-A3B-gguf", "MiniCPM-V-4_7-35B-A3B-mix_q3gud_im.gguf", "R_official"),
    # ---- G3 Xing4.0 29B-A4B: f16 建 C/D/R; 官方 iq4_xs; 两个社区 Q4_K_M
    ("g3", "Venastine-Research/Xing4.0-29B-A4B-GGUF", "Xing4.0-29B-A4B-f16.gguf", "f16"),
    ("g3", "Venastine-Research/Xing4.0-29B-A4B-GGUF", "Xing4.0-29B-A4B-Q4_K_M.gguf", "B1"),
    ("g3", "dealignai/Xing4.0-29B-A4B-CRACK-GGUF", "Xing4.0-29B-A4B-CRACK-Q4_K_M.gguf", "B2"),
    ("g3", "XingChen-AGI/Xing4.0-29B-A4B-GGUF", "xing4_0-29b-iq4_xs.gguf", "A"),
    ("g3", "XingChen-AGI/Xing4.0-29B-A4B-GGUF", "xing4_0-29b-imatrix.gguf", "imatrix_official"),
    # ---- G4 Clover 35B-A3B (dLLM): A 官方 GGUF (HF); B 社区 AtomicChat AD 档 (ModelScope)
    ("g4", "Diffuspace/Clover-35B-A3B-Preview-GGUF", "Clover-35B-A3B-Preview-Q4_K_M.gguf", "A"),
    ("g4", "AtomicChat/Clover-35B-A3B-Preview-GGUF", "Clover-35B-A3B-Preview-AD-Q4_K-IQ4_XS.gguf", "B"),
    # B_low 不在本次 A/B 实验范围, 跳过以免占用带宽与磁盘
    # ("g4", "AtomicChat/Clover-35B-A3B-Preview-GGUF", "Clover-35B-A3B-Preview-AD-IQ3_XXS-IQ2_S.gguf", "B_low"),
    # ---- G5 GLM-4.7-Flash 30B-A3B (GGUF arch=deepseek2, GLM4.7 Lite 分支): 两档体积
    ("g5", "unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-UD-Q2_K_XL.gguf", "A"),
    ("g5", "unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-UD-Q3_K_XL.gguf", "B"),
    # ---- G6 Nemotron-Cascade-2-30B-A3B (arch=nemotron_h_moe, Mamba2+Attention 混合):
    #      非量化张量占比高, 连 IQ2 都 ~16.6 GiB, 只能部分卸载
    ("g6", "mradermacher/Nemotron-Cascade-2-30B-A3B-i1-GGUF",
     "Nemotron-Cascade-2-30B-A3B.i1-Q2_K.gguf", "A"),
    ("g6", "mradermacher/Nemotron-Cascade-2-30B-A3B-i1-GGUF",
     "Nemotron-Cascade-2-30B-A3B.i1-Q3_K_M.gguf", "B"),
    # ---- G7 Raon-1-mini-preview (arch=raon1, 本机两套 llama.cpp 均无该 arch, 先下最小档做冒烟取证)
    ("g7", "KRAFTON/Raon-1-mini-preview-gguf", "raon-1-mini-preview-IQ4_XS.gguf", "A"),
]

# TCQ 臂 (C/D/R) 需要 F16/BF16 源: G5 直接下官方 BF16 GGUF 分片; G6/G7 下原始 safetensors 再转换
# G6/G7 的 HF 仓库把权重放在根目录 (无 src/ 前缀), 但 convert 需要独立的 src/ 目录, 下载时落到 <dst>/src
BASE = {
    "g5": [("unsloth/GLM-4.7-Flash-GGUF", "BF16/GLM-4.7-Flash-BF16-00001-of-00002.gguf"),
           ("unsloth/GLM-4.7-Flash-GGUF", "BF16/GLM-4.7-Flash-BF16-00002-of-00002.gguf")],
    "g6": [("nvidia/Nemotron-Cascade-2-30B-A3B", "model-%05d-of-00013.safetensors" % i)
           for i in range(1, 14)] + [
        ("nvidia/Nemotron-Cascade-2-30B-A3B", f) for f in
        ("config.json", "generation_config.json", "model.safetensors.index.json",
         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "chat_template.jinja")],
    "g7": [("KRAFTON/Raon-1-mini-preview", "model-%05d-of-00007.safetensors" % i)
           for i in range(1, 8)] + [
        ("KRAFTON/Raon-1-mini-preview", f) for f in
        ("config.json", "generation_config.json", "model.safetensors.index.json",
         "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")],
}

HF_SHARDS = ["model-%05d-of-00016.safetensors" % i for i in range(1, 17)]

DST = {"g2": r"E:\CUDA11.8\model\minicpm", "g3": r"E:\CUDA11.8\model\xing",
       "g4": r"E:\CUDA11.8\model\clover", "g5": r"E:\CUDA11.8\model\glm47",
       "g6": r"E:\CUDA11.8\model\nemotron", "g7": r"E:\CUDA11.8\model\raon"}
# 只发在 HF (或 ModelScope 未镜像) 的组, 一律走 hf-mirror
HF_GROUPS = {"g6", "g7"}


def main():
    global WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="g2,g3")
    ap.add_argument("--with-hf", action="store_true", help="同时下 MiniCPM 的 HF safetensors (路由统计用)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    WORKERS = args.workers
    if args.list:
        for g, r, f, role in JOBS:
            print("%-3s %-10s %-45s %s" % (g, role, f, r))
        return 0

    groups = {x.strip() for x in args.only.split(",") if x.strip()}
    say("download start (modelscope, chunk=%d MiB workers=%d groups=%s)"
        % (CHUNK >> 20, WORKERS, sorted(groups)))
    for g in sorted(groups):
        dst = DST[g]
        if not pick_lock(dst):
            continue
        try:
            repo_hf = {"g2": "openbmb/MiniCPM-V-4.7-35B-A3B",
                       "g3": "XingChen-AGI/Xing4.0-29B-A4B",
                       "g4": "Diffuspace/Clover-35B-A3B-Preview",
                       "g5": "unsloth/GLM-4.7-Flash-GGUF",
                       "g6": "mradermacher/Nemotron-Cascade-2-30B-A3B-i1-GGUF",
                       "g7": "KRAFTON/Raon-1-mini-preview-gguf"}[g]
            get_small(repo_hf, dst, SMALL)
            for jg, repo, f, role in JOBS:
                if jg != g:
                    continue
                # 官方 Clover GGUF 与 G5-G7 只发在 HF/未镜像, 走 hf-mirror; 其余走 ModelScope
                fmt = HF if (g in HF_GROUPS or (g == "g4" and role == "A")) else MS
                try:
                    get_file(repo, f, dst, url_fmt=fmt)
                except Exception as e:
                    say("FAIL %s %r" % (f, e))
            # F16/BF16 源 (TCQ 臂用): G5 的 BF16 在 ModelScope 有镜像 (35 MB/s), G6/G7 只发 HF
            base_dst = os.path.join(dst, "src") if g in ("g6", "g7") else dst
            for repo, f in BASE.get(g, []):
                try:
                    get_file(repo, f, base_dst, url_fmt=MS if g == "g5" else HF)
                except Exception as e:
                    say("FAIL %s %r" % (f, e))
            if g == "g2" and args.with_hf:
                for f in HF_SHARDS:
                    try:
                        get_file(repo_hf, f, dst)
                    except Exception as e:
                        say("FAIL %s %r" % (f, e))
        except BaseException:
            import traceback
            say("CRASH " + traceback.format_exc()[-900:])
        finally:
            try:
                os.remove(os.path.join(dst, ".dl.lock"))
            except Exception:
                pass
    say("download done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
