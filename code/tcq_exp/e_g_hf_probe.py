"""G2/G3 第0步: HF 仓库盘点 (走 hf-mirror, 本机只有镜像可解析)。"""
import json
import sys
import urllib.request

API = "https://hf-mirror.com/api"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def show(repo):
    try:
        d = get(f"{API}/models/{repo}")
    except Exception as e:  # noqa: BLE001
        print(f"\n=== {repo} ERROR {e!r}")
        return
    print(f"\n=== {repo}")
    print("  created=%s modified=%s downloads=%s likes=%s"
          % (d.get("createdAt"), d.get("lastModified"), d.get("downloads"), d.get("likes")))
    gg = [s["rfilename"] for s in d.get("siblings", []) if s["rfilename"].lower().endswith(".gguf")]
    print("  gguf files: %d" % len(gg))
    for f in gg:
        print("    " + f)


def search(q, limit=30):
    d = get(f"{API}/models?search={q}&limit={limit}")
    print(f"\n--- search '{q}': {len(d)} hits")
    for m in d:
        print("  %-64s dl=%-8s cr=%s" % (m["modelId"], m.get("downloads"),
                                          (m.get("createdAt") or "")[:10]))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a.startswith("q:"):
            search(a[2:])
        else:
            show(a)
