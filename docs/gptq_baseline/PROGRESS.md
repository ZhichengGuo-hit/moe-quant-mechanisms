# E-GPTQ 状态：本机被判分工具阻塞（脚本已就绪）

状态（2026-11-18）：**未完成量化**。阻塞点在"装不上可用的 GPTQ 量化器"，
不是算力或显存；量化脚本与评测脚本已写好，换环境即可直接跑。

## 1. 已完成

- `PREREG.md`：协议（3-bit / group 128 / damp 0.01，128 条 C4 校准）、
  评测（GSM8K 200 + HE 164，与 E17 同题同判分）、三条判定线、体积不比。
- `quantize_gptq.py`：gptqmodel 7.x 路径，校准集从 `tcq_exp/data/*_calib.jsonl`
  轮转取 128 条，seqlen 2048，逐层 offload。
- `eval_gptq.py`：复用 E17 的 prompt 模板与判分（`extract_number` / `extract_code` /
  子进程跑 HumanEval 测试），greedy、1024、断点续跑。

## 2. 阻塞证据（可复现）

| 尝试 | 结果 |
| --- | --- |
| `pip install gptqmodel==7.11.0` | 依赖 `pypcre`；其构建脚本从 GitHub clone PCRE2 源码，本机失败 → `RuntimeError: Failed to clone PCRE2 source` |
| `pip install gptqmodel==5.6.12 --no-build-isolation` | 同一 `pypcre` 构建失败 |
| `pip install --no-deps gptqmodel==7.11.0` | 安装成功，但 import 报 `ModuleNotFoundError: No module named 'pcre'` |
| 版本安全门 | 7.11 要求 torch≥2.13；本机 kt_env 为 torch 2.4.1+cu124。设置 `GPTQ_BYPASS_SAFETY_CHECKS=1` 仍缺 pcre |
| `pip install llm-compressor` | 当前镜像无该包（`No matching distribution`） |
| `auto-gptq==0.7.1` | 有轮子，但其架构表早于 Qwen3.5-MoE，无法加载 `Qwen3_5MoeForConditionalGeneration` |
| `exllamav2==0.3.2` | 有轮子，但需转成 ExLlama 自有格式并单独校准；未验证支持该新架构，且同样受 16 GB 显存约束 |

## 3. 复现命令（在可用环境）

```bash
# Linux + torch>=2.13 + gptqmodel>=7.11（或能访问 GitHub 构建 pypcre 的 Windows）
python quantize_gptq.py --model /path/Qwen3.6-35B-A3B \
    --out out/qwen36-3bit-gptq --bits 3 --group 128 --damp 0.01 --calib 128
python eval_gptq.py --model out/qwen36-3bit-gptq --task gsm8k
python eval_gptq.py --model out/qwen36-3bit-gptq --task humaneval
```

预计量化 12–24 h（16 GB 显存逐层 offload），评测 3 机时。

## 4. 对论文的影响

- M2 的回应暂时只能停留在"RTN 族内"的措辞，不能声明"连补偿式量化也被饱和带吸收"。
- 若后续在可用环境完成：按 PREREG 的三条判定线对号；无论哪个方向，
  都能把组件阶梯的适用范围说清楚。
- 这不是负结果，而是工具链约束；论文如需提交前补齐，建议在 Linux 机器上跑一次。
