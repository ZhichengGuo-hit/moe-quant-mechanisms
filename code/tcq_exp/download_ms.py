"""ModelScope 并行分段下载 (HF 镜像不稳时的替代方案)。"""
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor

BASE = "https://modelscope.cn/models/{repo}/resolve/master/{name}"
CHUNK = 16 * 1024 * 1024
NTHREAD = 16
RETRY = 4


def url(repo, name):
    return BASE.format(repo=repo, name=name)


def size_of(repo, name):
    for _ in range(RETRY):
        try:
            r = requests.get(url(repo, name), headers={"Range": "bytes=0-0"},
                             timeout=30, stream=True)
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                return int(cr.split("/")[-1])
            cl = r.headers.get("Content-Length")
            if cl:
                return int(cl)
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"size fail {name}")


def get_range(repo, name, start, end, dst):
    for _ in range(RETRY):
        try:
            r = requests.get(url(repo, name),
                             headers={"Range": f"bytes={start}-{end}"},
                             timeout=180, stream=True)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"http {r.status_code}")
            with open(dst, "r+b") as f:
                f.seek(start)
                for c in r.iter_content(1 << 20):
                    f.write(c)
            return end - start + 1
        except Exception as e:
            err = e
            time.sleep(2)
    raise RuntimeError(f"chunk fail {start}: {err}")


def download(repo, name, dst_dir):
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {name}", flush=True)
        return
    total = size_of(repo, name)
    print(f"[get] {name} {total/2**30:.2f} GiB", flush=True)
    if total < 4 * CHUNK:
        r = requests.get(url(repo, name), timeout=300)
        r.raise_for_status()
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "wb") as f:
            f.write(r.content)
        return
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.truncate(total)
    offs = list(range(0, total, CHUNK))
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(NTHREAD) as ex:
        futs = []
        for o in offs:
            e = min(o + CHUNK - 1, total - 1)
            futs.append(ex.submit(get_range, repo, name, o, e, tmp))
        for fu in futs:
            done += fu.result()
    os.replace(tmp, dst)
    print(f"[done] {name} {total/2**30:.2f} GiB {time.time()-t0:.0f}s "
          f"({total/2**20/max(time.time()-t0,1e-9):.1f} MB/s)", flush=True)


def main():
    repos = {
        "Qwen/Qwen3-30B-A3B": (r"E:\CUDA11.8\model\Qwen3-30B-A3B",
                               [f"model-{i:05d}-of-00016.safetensors" for i in range(1, 17)] +
                               ["config.json", "generation_config.json",
                                "model.safetensors.index.json", "tokenizer.json",
                                "tokenizer_config.json", "vocab.json", "merges.txt",
                                "special_tokens_map.json", "chat_template.jinja"]),
        "google/gemma-4-26B-A4B-it": (r"E:\CUDA11.8\model\gemma-4-26B-A4B",
                                      [f"model-{i:05d}-of-00002.safetensors" for i in range(1, 3)] +
                                      ["config.json", "generation_config.json",
                                       "model.safetensors.index.json", "tokenizer.json",
                                       "tokenizer_config.json", "tokenizer.model",
                                       "special_tokens_map.json", "chat_template.jinja"]),
    }
    for repo, (dst, names) in repos.items():
        os.makedirs(dst, exist_ok=True)
        print(f"=== {repo} -> {dst}", flush=True)
        for n in names:
            try:
                download(repo, n, dst)
            except Exception as e:
                print(f"[fail] {n}: {repr(e)[:120]}", flush=True)
        print(f"=== {repo} finished", flush=True)


if __name__ == "__main__":
    main()
