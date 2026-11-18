#!/usr/bin/env python
"""E-A1a: BFCL-v3 多轮 (multi_turn_base + multi_turn_long_context), 6 配置.

框架: EvalScope 1.14.1 (openai_api -> 本机 llama-server) + bfcl-eval==2025.10.27.1 打分.
每配置独占一个 llama-server 端口 (18141), 跑完即停; 结果写 results/a1a/<cfg>/ 与
results/a1a_summary.json (含 EvalScope 分数 + 行为指标: 格式错误率 / 循环率).

    E:\\CUDA11.8\\train\\a1_env\\Scripts\\python.exe e_a1a_bfcl.py --configs iq2xxs --limit 2
    E:\\CUDA11.8\\train\\a1_env\\Scripts\\python.exe e_a1a_bfcl.py
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODEL_ROOT = r"E:\CUDA11.8\model"
CUDA_BIN = r"E:\CUDA11.8\train\llama-upstream\build-cuda\bin\llama-server.exe"
CPU_BIN = r"E:\CUDA11.8\train\llama.cpp\build-tcmoe\bin\Release\llama-server.exe"
TCMOE = os.path.abspath(os.path.join(HERE, "..", "tcmoe", "out", "model-tcmoe-q4k-nl.gguf"))

CONFIGS = {
    "iq2xxs": {"path": os.path.join(MODEL_ROOT, "baselines", "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"),
               "bin": CUDA_BIN, "ngl": 99, "domain": None},
    "ud_q2kxl": {"path": os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
                 "bin": CUDA_BIN, "ngl": 99, "domain": None},
    "ud_q4km": {"path": os.path.join(MODEL_ROOT, "Qwen3.6-35B-A3B",
                                     "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
                "bin": CUDA_BIN, "ngl": 20, "domain": None},
    # 补充控制臂: UD 权重放到 CPU(同一上游 binary, -ngl 0) 以分离"CPU/部分卸载路径"与"TCQ 模型"混淆
    "ud_q2kxl_cpu": {"path": os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
                     "bin": CUDA_BIN, "ngl": 0, "domain": None},
    "ud_q4km_cpu": {"path": os.path.join(MODEL_ROOT, "Qwen3.6-35B-A3B",
                                         "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
                    "bin": CUDA_BIN, "ngl": 0, "domain": None},
    # 控制臂 3: 同一 UD 权重换到 tcmoe fork binary(-ngl 0), 分离"fork server"与"TCQ 文件"
    "ud_q2kxl_fork": {"path": os.path.join(MODEL_ROOT, "ud", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
                      "bin": CPU_BIN, "ngl": 0, "domain": None},
    # CPU 三臂同一文件/同一 binary, 只靠请求级 domain 切换
    "tcq_nl_global": {"path": TCMOE, "bin": CPU_BIN, "ngl": 0, "domain": "global"},
    "tcq_nl_math": {"path": TCMOE, "bin": CPU_BIN, "ngl": 0, "domain": "math"},
    "tcq_nl_code": {"path": TCMOE, "bin": CPU_BIN, "ngl": 0, "domain": "code"},
    # 同 domain=math、仅用于 n=40 复测（输出文件与 tcq_nl_math 隔离）
    "tcq_nl_math_n40": {"path": TCMOE, "bin": CPU_BIN, "ngl": 0, "domain": "math"},
    "base2": {"path": TCMOE, "bin": CPU_BIN, "ngl": 0, "domain": "base"},
}
PORT = 18141
PROXY_PORT = 18142
CTX = 65536 + 4096  # multi_turn_long_context 实测 prompt 33177 token, 32768 会 400; 与 E-H5 同口径
THREADS = int(os.environ.get("A1A_THREADS", "32"))  # 双路 Xeon 4215R: 16C/32T
SUBSETS = ["multi_turn_base", "multi_turn_long_context"]


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def wait_port_free(port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2)
        except Exception:  # noqa: BLE001
            return True
        time.sleep(1)
    return False


def start_server(cfg, log_path, timeout=1800):
    assert wait_port_free(PORT), "端口 %d 仍被占用" % PORT
    cmd = [cfg["bin"], "-m", cfg["path"], "-c", str(CTX), "-ngl", str(cfg["ngl"]),
           "-t", str(THREADS), "--host", "127.0.0.1", "--port", str(PORT), "-np", "1",
           "--jinja", "--chat-template-kwargs", '{"enable_thinking": false}',
           "--no-webui", "--verbosity", "1"]
    fh = open(log_path, "w", encoding="utf-8")
    fh.write(" ".join(cmd) + "\n")
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit("llama-server 退出 code=%s (见 %s)" % (proc.returncode, log_path))
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=5) as r:
                if r.status == 200:
                    log("server up %.0fs" % (time.time() - t0))
                    return proc, fh
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise SystemExit("llama-server 启动超时")


def stop_server(proc, fh):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        fh.close()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- 行为指标
def _steps(generation):
    """generation: [turn][step] -> 扁平 step 列表"""
    out = []
    for turn in generation or []:
        for step in turn:
            out.append(step)
    return out


def behavior_metrics(pred_file):
    """格式错误率 / 循环率 (样本级), 依据 EvalScope 预测缓存里的 metadata.generation。

    格式错误: 期望工具调用 (should_execute_tool_calls) 的步里, 输出是纯文本且含 { 或 [
              (说明模型试图发调用但没能形成合法 tool_call)
    循环:     单轮内同一调用连续重复 >=3 次, 或触及 20 步上限
    """
    n = fmt_bad = loop_bad = 0
    for line in open(pred_file, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        md = rec.get("metadata") or rec.get("sample_metadata") or {}
        gen = md.get("generation")
        if gen is None:
            continue
        n += 1
        expect = bool(md.get("should_execute_tool_calls", True))
        bad_fmt = False
        bad_loop = False
        for turn in gen:
            prev, run = None, 0
            for step in turn:
                key = json.dumps(step, sort_keys=True, ensure_ascii=False) if not isinstance(step, str) else None
                if key is not None and key == prev:
                    run += 1
                    if run >= 3:
                        bad_loop = True
                else:
                    run = 1 if key is not None else 0
                prev = key
                if expect and isinstance(step, str) and ("{" in step or "[" in step):
                    bad_fmt = True
            if len(turn) > 20:
                bad_loop = True
        fmt_bad += 1 if bad_fmt else 0
        loop_bad += 1 if bad_loop else 0
    if not n:
        return None
    return {"n": n, "format_error_rate": fmt_bad / n, "loop_rate": loop_bad / n}


# ---------------------------------------------------------------- E-A1g 干预开关
A1G_HINT = ("Working-directory state: cd changes the current directory and affects later "
            "relative paths. Before any relative-path operation, verify the current directory "
            "(e.g. call pwd) and never assume a previous cd succeeded. If a command fails with "
            "\"No such file or directory\", re-check state instead of retrying the same command.")


def a1g_mode():
    return (os.environ.get("A1G_MODE") or "").strip().lower()


def a1g_single_call(data):
    """E-A1g 计划先行干预: 上游响应只留第一个 tool_call, 其余丢弃并在 content 末尾留痕."""
    try:
        r = json.loads(data)
    except Exception:  # noqa: BLE001
        return data
    dropped = []
    for ch in r.get("choices") or []:
        msg = (ch or {}).get("message") or {}
        tcs = msg.get("tool_calls") or []
        if len(tcs) > 1:
            dropped += [((t.get("function") or {}).get("name") or "?") for t in tcs[1:]]
            msg["tool_calls"] = tcs[:1]
            line = "Dropped extra tool calls: " + ", ".join(dropped)
            content = msg.get("content") or ""
            msg["content"] = (content + "\n" + line) if content else line
            ch["finish_reason"] = "tool_calls"
    if not dropped:
        return data
    log("A1G single_call dropped %d extra tool call(s): %s" % (len(dropped), ", ".join(dropped)))
    return json.dumps(r, ensure_ascii=False).encode()


# ---------------------------------------------------------------- domain 注入代理
class _DomainProxy(object):
    """把 EvalScope 的 OpenAI 请求原样转给 llama-server, 对 tcmoe 三臂注入请求级 domain.

    tcmoe 补丁用 /completion 的 "domain" 字段选择专家补丁, 缺省 = 首个声明域;
    不注入则 tcq_nl_global / tcq_nl_math / base2 会退化成同一个臂.
    """

    def __init__(self, upstream_port, domain):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.domain = domain
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # 静音
                pass

            def _forward(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                if body and method == "POST":
                    try:
                        d = json.loads(body)
                        # OAI chat 端点默认不缓存前缀; 打开后 BFCL 多轮同前缀只增量 prefill
                        d.setdefault("cache_prompt", True)
                        if outer.domain:
                            d["domain"] = outer.domain
                        if a1g_mode() == "cwd_hint" and isinstance(d.get("messages"), list):
                            d["messages"] = d["messages"] + [{"role": "system", "content": A1G_HINT}]
                        body = json.dumps(d).encode()
                    except Exception:  # noqa: BLE001
                        pass
                hdrs = {k: v for k, v in self.headers.items()
                        if k.lower() not in ("host", "content-length", "connection")}
                req = urllib.request.Request("http://127.0.0.1:%d%s" % (upstream_port, self.path),
                                             data=body, headers=hdrs, method=method)
                try:
                    with urllib.request.urlopen(req, timeout=1800) as r:
                        data = r.read()
                        status = r.status
                        rh = r.headers
                except urllib.error.HTTPError as e:
                    data = e.read()
                    status = e.code
                    rh = e.headers
                if a1g_mode() == "single_call" and status == 200 and data:
                    data = a1g_single_call(data)
                self.send_response(status)
                for k, v in rh.items():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._forward("GET")

            def do_POST(self):
                self._forward("POST")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), Handler)
        self.th = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.th.start()

    def stop(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- 单配置
def run_one(name, subsets, limit):
    cfg = CONFIGS[name]
    os.makedirs(RES, exist_ok=True)
    log_path = os.path.join(RES, "a1a_%s.server.log" % name)
    work_dir = os.path.join(RES, "a1a", name)
    proc = fh = proxy = None
    try:
        proc, fh = start_server(cfg, log_path)
        proxy = _DomainProxy(PORT, cfg["domain"])
        from evalscope import TaskConfig, run_task
        tcfg = TaskConfig(
            model=name, model_id=name,
            api_url="http://127.0.0.1:%d/v1" % PROXY_PORT, api_key="EMPTY",
            eval_type="openai_api",
            datasets=["bfcl_v3"],
            dataset_args={"bfcl_v3": {
                "subset_list": subsets,
                "extra_params": {"underscore_to_dot": True, "is_fc_model": True},
            }},
            generation_config={"temperature": 0.0, "max_tokens": 4096,
                               "parallel_tool_calls": True, "timeout": 900, "stream": False},
            eval_batch_size=1, limit=limit, seed=20261111,
            work_dir=work_dir, no_timestamp=True,
        )
        t0 = time.time()
        res = run_task(tcfg)
        dur = (time.time() - t0) / 60.0
        try:
            report = {k: (json.loads(v.model_dump_json()) if hasattr(v, "model_dump_json") else str(v))
                      for k, v in (res or {}).items()}
        except Exception:  # noqa: BLE001
            report = {"raw": str(res)[:2000]}
        beh = {}
        for p in glob.glob(os.path.join(work_dir, "predictions", "*", "bfcl_v3_*.jsonl")):
            beh[os.path.basename(p)] = behavior_metrics(p)
        rec = {"config": name, "path": os.path.basename(cfg["path"]), "ngl": cfg["ngl"],
               "domain": cfg["domain"], "subsets": subsets, "limit": limit,
               "minutes": round(dur, 1), "report": report, "behavior": beh}
    finally:
        if proxy is not None:
            proxy.stop()
        stop_server(proc, fh)
    out = os.path.join(RES, "a1a_%s.json" % name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    log("[%s] done %.1f min -> %s" % (name, dur, out))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    names = [x for x in args.configs.split(",") if x]
    subsets = [x for x in args.subsets.split(",") if x]
    summary = {}
    for name in names:
        prev_path = os.path.join(RES, "a1a_%s.json" % name)
        if os.path.exists(prev_path):
            try:
                prev = json.load(open(prev_path, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                prev = None
            if (prev and prev.get("subsets") == subsets
                    and prev.get("limit") == args.limit and prev.get("report")):
                log("[%s] 已完成 (limit=%s, %s), 跳过" % (name, args.limit, ",".join(subsets)))
                summary[name] = prev
                continue
        summary[name] = run_one(name, subsets, args.limit)
    path = os.path.join(RES, "a1a_summary.json")
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = {}
    old.update(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    log("all done -> %s" % path)


if __name__ == "__main__":
    main()
