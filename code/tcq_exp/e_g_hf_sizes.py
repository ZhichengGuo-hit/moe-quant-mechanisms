"""列指定仓库的 gguf 文件大小 (MB) —— 用于选配置。"""
import json
import sys
import urllib.request

API = "https://hf-mirror.com/api"


def tree(repo):
    url = f"{API}/models/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


for repo in sys.argv[1:]:
    print(f"\n=== {repo}")
    for e in tree(repo):
        p = e.get("path", "")
        if p.lower().endswith(".gguf"):
            sz = e.get("size") or (e.get("lfs") or {}).get("size") or 0
            print("  %-56s %8.2f GB" % (p, sz / 1e9))
