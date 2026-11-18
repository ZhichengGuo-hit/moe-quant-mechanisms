# E-GPTQ 预注册：补偿式量化基线（回应 M2）

冻结时间：2026-11-18。
目标：给"组件阶梯"提供非 RTN 量化器对照——检验"唯一任务级组件"是否只是 RTN 族内现象。

## 1. 量化协议

- 模型：Qwen3.6-35B-A3B（HF safetensors，`E:\CUDA11.8\model\Qwen3.6-35B-A3B`）
- 量化器：GPTQ（gptqmodel 7.x；bits=3, group_size=128, damp_percent=0.01,
  desc_act=false, sym=true），校准集 128 条 C4 文本、seqlen 2048。
- 显存策略：16 GB GPU + 128 GB RAM，逐层 offload 量化（预计 12–24 h 后台）。
- 产出：HF 格式 3-bit GPTQ 权重（`gptq_baseline/out/qwen36-3bit-gptq`）。
- 体积不与 GGUF 比较（审稿人明确允许只比任务质量）；论文需写清协议差异。

## 2. 评测协议

- GSM8K 前 200 + HumanEval 164（与 E17 完全同题、同判分、greedy、1024 上限）。
- 路径 A：gptqmodel 推理后端直接跑（transformers 接口）；
- 路径 B：`convert_hf_to_gguf.py` 转 GGUF 后用既有 llama.cpp 评测器跑（更接近 E18 口径）。
- 配对对象：E18 网格里的 RTN-3bit 档（base3）与同预算 TCQ 臂。

## 3. 判定线（预注册）

- 差异 < MDE（GSM8K ±5.6 pt / HE ±4.2–6.5 pt，E-F7 口径）→ **饱和带吸收补偿**：
  机制地图可以声明普适性"连补偿式量化也被带内吸收"。
- GPTQ 显著更好 → 组件阶梯加限定词"RTN 族内"，机制节新增
  "补偿是第六个边界条件外的自由度"。
- GPTQ 显著更差 → 记录为"低比特下补偿反而放大离群"，机制节加一条反例。

## 4. 本机可行性（2026-11-18 实测，先于任何量化运行记录）

| 尝试 | 结果 |
| --- | --- |
| `pip install gptqmodel==7.11.0` | 依赖 `pypcre` 需从 GitHub clone PCRE2 源码构建，本机网络不可达 → 失败 |
| `pip install gptqmodel==5.6.12 --no-build-isolation` | 同一 `pypcre` 构建失败 |
| `pip install --no-deps gptqmodel==7.11.0` | 装上，但 import 时 `ModuleNotFoundError: pcre`；且 7.11 要求 torch≥2.13（本机 2.4.1+cu124），
设置了 `GPTQ_BYPASS_SAFETY_CHECKS=1` 仍缺 pcre → 不可用 |
| `pip install llm-compressor` | 当前镜像无该包 |
| `auto-gptq==0.7.1` | 有轮子，但架构支持早于 Qwen3.5-MoE（`Qwen3_5MoeForConditionalGeneration`） |
| `exllamav2==0.3.2` | 有轮子，但需 ExLlama 自己的模型格式与校准路径，未验证支持该新架构 |

结论：**本机无法安装可用的 GPTQ 量化器**，E-GPTQ 不能在本机完成。
`quantize_gptq.py` 与评测脚本已写好，可在具备 gptqmodel（或 Linux + 新 torch）的机器上直接运行。
建议路径：Linux + torch≥2.13 + gptqmodel≥7.11（或 Windows 上能访问 GitHub 构建 pypcre 的环境）。

## 5. 状态

- [x] 协议与判定线冻结
- [x] 量化/评测脚本（可移植）
- [ ] 量化执行（本机阻塞，见 §4）
- [ ] 评测与判定
