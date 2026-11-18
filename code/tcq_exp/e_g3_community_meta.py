"""E-G3: 社区构建元数据补录 (作者 / 发布日期 / 下载量 / 是否声明 imatrix / 滞后天数)。

§5.5 的"社区方差"一节需要这些列; 元数据与评测解耦, 评测跑完再补录即可。
数据源走 hf-mirror 的 API (huggingface.co 在本机 DNS 不通)。

    python e_g3_community_meta.py            # 拉取并写 results/e_g3_community_meta.json
    python e_g3_community_meta.py --attach   # 把元数据写进 B1/B2 的 jsonl 每条记录
"""
import argparse
import datetime as dt
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
API = "https://hf-mirror.com/api/models/%s"
RAW = "https://hf-mirror.com/%s/resolve/main/README.md"
OUT = os.path.join(RES, "e_g3_community_meta.json")

REPOS = {
    "official": "XingChen-AGI/Xing4.0-29B-A4B-GGUF",
    "B1_venastine_q4km": "Venastine-Research/Xing4.0-29B-A4B-GGUF",
    "B2_dealign_q4km": "dealignai/Xing4.0-29B-A4B-CRACK-GGUF",
}


def get(url, timeout=60):
    last = None
    for i in range(4):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "curl/8"},
                                       ), timeout=timeout).read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + i * 4)
    raise last


def iso(s):
    return (s or "")[:19]


def days_between(a, b):
    if not (a and b):
        return None
    try:
        ta = dt.datetime.fromisoformat(iso(a).replace("Z", ""))
        tb = dt.datetime.fromisoformat(iso(b).replace("Z", ""))
        return round((tb - ta).total_seconds() / 86400, 2)
    except Exception:  # noqa: BLE001
        return None


def fetch_repo(repo):
    d = json.loads(get(API % repo))
    files = []
    for s in d.get("siblings", []):
        n = s["rfilename"]
        if n.lower().endswith(".gguf"):
            files.append(n)
    try:
        readme = get(RAW % repo).decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        readme = ""
    low = readme.lower()
    return {
        "repo": repo,
        "author": d.get("author") or repo.split("/")[0],
        "created_at": iso(d.get("createdAt")),
        "last_modified": iso(d.get("lastModified")),
        "downloads": d.get("downloads"),
        "likes": d.get("likes"),
        "gguf_files": files,
        "declares_imatrix": ("imatrix" in low) or ("importance matrix" in low),
        "readme_head": readme[:1200],
    }


def attach(meta):
    prefix = "e_g3_xing_"
    for tag in ("B1_venastine_q4km", "B2_dealign_q4km"):
        m = meta["repos"].get(tag)
        if not m:
            continue
        for task in ("gsm8k", "humaneval", "gpqa"):
            p = os.path.join(RES, "%s%s_%s.jsonl" % (prefix, tag, task))
            if not os.path.exists(p):
                continue
            rows = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    r["community_meta"] = {
                        "author": m["author"], "repo": m["repo"],
                        "created_at": m["created_at"], "downloads": m["downloads"],
                        "declares_imatrix": m["declares_imatrix"],
                        "lag_days_vs_official": meta["lag_days_vs_official"].get(tag),
                    }
                    rows.append(r)
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, p)
            print("[attach] %s %d 条" % (os.path.basename(p), len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attach", action="store_true")
    args = ap.parse_args()
    if args.attach:
        with open(OUT, encoding="utf-8") as f:
            attach(json.load(f))
        return 0

    repos, official = {}, None
    for tag, repo in REPOS.items():
        m = fetch_repo(repo)
        repos[tag] = m
        print("[meta] %-20s %-45s created=%s downloads=%s imatrix=%s"
              % (tag, repo, m["created_at"], m["downloads"], m["declares_imatrix"]))
        if tag == "official":
            official = m["created_at"]
    lag = {tag: days_between(official, m["created_at"]) for tag, m in repos.items()}
    payload = {"fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
               "official_first_gguf": official, "lag_days_vs_official": lag, "repos": repos,
               "note": "lag_days 以官方仓库创建时间为零点; §5.5 表格用 author/created/downloads/imatrix 四列"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[write]", OUT)
    print("[lag]", lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
