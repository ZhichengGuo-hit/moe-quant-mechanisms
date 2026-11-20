# 复现信息（Reproducibility Sheet）

生成时间：2026-11-20（Asia/Shanghai）。适用对象：`OPENSOURCE/`（代码 + 结果 + llama.cpp patch）与本工作区 `E:\CUDA11.8\train`。

本文件回答六件事：依赖版本、机器环境、llama.cpp 基线与 patch、最小一键流程、外部制品/数据的来源与校验、未入库大文件如何获取。

---

## 0. 复现分层（先看这里）

| 层级 | 目标 | 依赖 | 现状 |
|---|---|---|---|
| L1 结果核验 | 从 `results/` 复算论文表格、图 | Python + numpy/pandas | 开箱可跑（基础解释器） |
| L2 量化/评测重跑 | 重新构建 GGUF、跑 GSM8K/HE/GPQA、路由导出 | llama.cpp 双仓 patch + CUDA 构建 + 外部 GGUF | 可复现（需下载约 0.9 TB 制品） |
| L3 离线 INT4/torch 分析 | `exp1–16`、`d1*`、`int4_ext` 等 | torch 环境 | **部分缺口**：基础解释器无 torch，历史版本未完整记录（见 §8） |

---

## 1. 硬件与操作系统

| 项 | 值 |
|---|---|
| OS | Microsoft Windows 10 企业版 LTSC，10.0.17763，x64 |
| CPU | 2 × Intel Xeon Silver 4215R @ 3.20 GHz（16 核 / 32 线程） |
| 内存 | 127.7 GB |
| GPU | NVIDIA Quadro RTX 5000，16 GB，compute capability 7.5 |
| 驱动 | 582.08（`nvidia-smi` 报告 CUDA API 13.0） |
| CUDA Toolkit | 12.4（`nvcc` V12.4.99） |
| 磁盘 | 论文相关制品约 2.0 TB（`E:\CUDA11.8\model`） |

注：驱动 API 13.0 与 Toolkit 12.4 并存是正常组合（构建用 12.4，运行由驱动向后兼容）。

---

## 2. 工具链

| 工具 | 版本 |
|---|---|
| CMake | 4.4.0 |
| MSVC | 14.44.35207（VS 2022 BuildTools） |
| Windows SDK | 10.0.26100.0（安装在 `G:\Windows Kits\10`） |
| CUDA 构建开关 | `-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release` |
| CPU-only 构建 | `llama.cpp/build-tcmoe`：`GGML_CUDA=OFF` |

MSVC 环境由 `coserving/msvc_env.ps1` 注入（`cl` / `INCLUDE` / `LIB`），构建前必须 dot-source。

---

## 3. Python 与依赖

基础解释器：`C:\Users\administered\miniconda3\python.exe`，**Python 3.13.14**。

| 包 | 版本（基础解释器） |
|---|---|
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| pyarrow | 25.0.1 |
| transformers | 5.5.0 |
| tokenizers | 0.22.2 |
| datasets | 4.3.0 |
| sentencepiece | 0.2.2 |
| safetensors | 0.8.0 |
| pillow | 12.3.0 |
| psutil | 7.2.2 |
| requests | 2.34.2 |
| tqdm | 4.68.2 |

**基础解释器缺失**：`torch`、`scipy`、`matplotlib`、`evalscope`、`gguf`（pip 包）。其中：

- `gguf` 由 `llama.cpp/gguf-py` 提供（脚本里 `sys.path` 指向该目录），无 pip 版本；
- `torch` 仅在 conda 环境可用：`py311` → torch 2.13.0；`unsloth` → torch 2.6.0+cu124；
- 历史 GPTQ 记录提到 torch 2.4.1+cu124，与现存两个环境均不一致（见 §8）。

依赖清单见同目录 `requirements.txt`（基础层可装部分 + 可选层注释）。

---

## 4. llama.cpp 基线与 patch

| 仓库 | 本地目录 | 基线 commit | 提交日期 | patch | 内容 |
|---|---|---|---|---|---|
| llama.cpp（tcmoe fork） | `llama.cpp/` | `5047748` | 2026-10-30 | `llamacpp/llama.cpp-tcmoe.patch`（163.4 KB） | TCMOE Q2NL 量化类型、server `domain` 字段、`tcmoe-verify`、`LLAMA_MOE_DUMP`、`llama-route-dump` |
| llama.cpp（upstream） | `llama-upstream/` | `18ab62a` | 2026-11-06 | `llamacpp/llama-upstream.patch`（16.5 KB） | Clover `clover_moe` 架构放行、`LLAMA_MOE_DUMP`、`llama-route-dump` |

patch 由 `git diff HEAD` 生成（未跟踪新文件先 `git add -N`），包含新增文件 `src/llama-tcmoe.cpp/.h`、`tools/tcmoe-verify/*`、`examples/eval-callback/route-dump.cpp`。`llamacpp/new-files/` 另存一份新文件全文，便于直接查阅。

### 应用命令

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 5047748
git apply -p1 ..\OPENSOURCE\llamacpp\llama.cpp-tcmoe.patch

# upstream（E-G4 Clover 与 E-R2 路由导出）
git clone https://github.com/ggml-org/llama.cpp.git llama-upstream
cd llama-upstream
git checkout 18ab62a
git apply -p1 ..\OPENSOURCE\llamacpp\llama-upstream.patch
```

### 构建命令（Windows / CUDA）

```powershell
. .\coserving\msvc_env.ps1
cmake -S llama.cpp -B llama.cpp/build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-cuda --config Release -j 16 --target llama-server llama-quantize llama-route-dump
```

产物位置依生成器而定：MSVC 多配置 → `build-cuda\bin\Release\`；Ninja → `build-cuda\bin\`。仓库内现存的 CPU 构建为 `llama.cpp\build-tcmoe\bin\Release\llama-cli.exe`。

---

## 5. 最小一键流程

前置：已按 §4 应用 patch；已下载至少一个 GGUF（见 §6）。

```powershell
# 1) 环境自检
. .\coserving\msvc_env.ps1
cmake --version
python -c "import numpy,pandas; print('py ok')"

# 2) 构建（首次）
cmake -S llama.cpp -B llama.cpp/build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-cuda --config Release -j 16 --target llama-server llama-quantize llama-route-dump

# 3) 冒烟（CPU 构建；CUDA 构建替换为对应 bin 路径）
.\llama.cpp\build-tcmoe\bin\Release\llama-cli.exe -m E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf -p "1+1=" -n 8 -ngl 99 --no-warmup

# 4) 自检（导出表可重建）
python export_opensource.py
```

或用封装脚本（放在 `OPENSOURCE/` 内）：

```powershell
powershell -ExecutionPolicy Bypass -File .\repro_check.ps1 -Model E:\CUDA11.8\model\ud\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
```

`repro_check.ps1` 只做只读检查与可选冒烟，不修改仓库、不下载数据。

---

## 6. 外部模型制品（GGUF）

来源仓库（均以 HF card 的 LICENSE 为准）：主模型 `unsloth/Qwen3.6-35B-A3B-GGUF`（@ `a483e9e6cbd595906af30beda3187c2663a1118c`，license apache-2.0，base `Qwen/Qwen3.6-35B-A3B`）；跨模型见下表角色与 §6.1 仓库清单。

下表 SHA256 取自 Hugging Face LFS 元数据（`X-Linked-Etag`），并逐项核对 `X-Linked-Size` 与本地文件字节数一致；其中 IQ2_XXS 已用本地 `Get-FileHash` 复算，与 LFS 值完全一致，验证了该口径。

| 角色 | 文件 | GiB | 字节 | SHA256（上游 LFS） |
|---|---|---|---|---|
| 主模型基线 | `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` | 10.02 | 10,756,586,464 | `2E8F5F705355C56311432D0A8A5D14A696DBB7E4B197D05C75BA805FC1857BEF` |
| 主模型基线 | `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | 12.30 | 13,211,155,424 | `9C964E657212FEA1F24905DD7B0A89B82FD807D19FAB0B41DA14251B07B88FBE` |
| 主模型基线 | `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` | 11.45 | 12,290,628,576 | `96B9C0AF5C77A4ECAABE3983175112B5ECE763261C1ECE12B2494B692A70DAD7` |
| 主模型基线 | `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | 15.69 | 16,845,511,648 | `A832B9689925F1BD335BBE985CDFB06C36BF2CF268F4F8F6ECEAFA3CEB515617` |
| 主模型基线 | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | 20.82 | 22,360,456,160 | `707A55A8A4397ECDE44DE0C499D3E68C1AD1D240D1DA65826B4949D1043F4450` |
| 主模型基线 | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 20.61 | 22,134,528,992 | `AC0E2C1189E055FAA36EFF361580E79C5BD6F8E76BFFB4CE547F167D53E31A61` |
| E-G1 A | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 13.27 | 14,249,047,104 | `A7C5BC715F5FF8E99A3E8901CE7D2B42B402C669BF24F7C5250747633D0F5891` |
| E-G1 B | `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` | 15.84 | 17,010,980,576 | `EF728C8E0C337FD1067B947AF006E38A9EF2419E56FECED4FD29B4BF0636E30C` |
| E-G2 A | `MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf` | 19.71 | 21,166,255,008 | `9ECF91E814D7E9ADC64325B1B65F8628A9B1BB8DA4F61A92869CA7489BFD888E` |
| E-G3 A | `xing4_0-29b-iq4_xs.gguf` | 17.02 | 18,272,862,944 | `1A55EA0784C51AA59B224A625485920D56D1643BCFA8309575E291C87891AA94` |
| E-G3 B1 | `Xing4.0-29B-A4B-Q4_K_M.gguf` | 17.65 | 18,950,630,112 | `09148C314610AA3F341B64023CF806342EE7702BA8857CFDFECFC09443DE71F2` |
| E-G3 B2 | `Xing4.0-29B-A4B-CRACK-Q4_K_M.gguf` | 17.65 | 18,950,630,368 | `03ACBEB8EDA3157EC6A977FF257833B5FA6B70EC2FF5BC3904B25402FAB615E1` |
| E-G4 A | `Clover-35B-A3B-Preview-Q4_K_M.gguf` | 21.42 | 22,998,009,248 | `4D439BB832487FBB91D0A739A5746CF3089BDE4137C17050576B24E6DBA3539C` |
| E-G4 B | `Clover-35B-A3B-Preview-AD-Q4_K-IQ4_XS.gguf` | 18.74 | 20,125,924,224 | `F3A034C808531A7C355ED7558E22F446843171D99C150EC51B3BF69D517DB95A` |
| E-G5 A | `GLM-4.7-Flash-UD-Q2_K_XL.gguf` | 11.07 | 11,888,759,136 | `08A432581D3A797AF07A021455ADA33499185213AD250E85FB26DAF3FE34C421` |
| E-G5 B | `GLM-4.7-Flash-UD-Q3_K_XL.gguf` | 12.84 | 13,782,004,064 | `582728C0F6931A1CF734D5A7CF60EC73522FB79811E004CBC12476DD69B033BF` |
| E-G6 A | `Nemotron-Cascade-2-30B-A3B.i1-Q2_K.gguf` | 16.68 | 17,908,994,944 | `F6F0C001902B93451B39E7F5423493A4CFED501D2975F02009DF5567C0C4689E` |
| E-G6 B | `Nemotron-Cascade-2-30B-A3B.i1-Q3_K_M.gguf` | 18.49 | 19,849,236,352 | `86F9EADCB4C8B7014F52D0F78916F9E139E6F123BBFC95F70BB333B4C0BB5DFC` |
| E-G7 A | `raon-1-mini-preview-IQ4_XS.gguf` | 14.97 | 16,071,801,280 | `10BD6AED887A10C3EA962DCBBB88EF12F9A9228DA914B9072FB424118F5B5BDC` |

### 6.1 跨模型仓库清单

| 角色 | 上游仓库 | 文件 |
|---|---|---|
| E-G1 A/B | unsloth/gemma-4-26B-A4B-it-qat-GGUF / unsloth/gemma-4-26B-A4B-it-GGUF | qat-UD-Q4_K_XL / UD-Q4_K_XL |
| E-G2 A | openbmb/MiniCPM-V-4.7-35B-A3B-gguf | Q4_K_M |
| E-G3 A/B1/B2 | XingChen-AGI/Xing4.0-29B-A4B-GGUF / Venastine-Research/Xing4.0-29B-A4B-GGUF / dealignai/Xing4.0-29B-A4B-CRACK-GGUF | iq4_xs / Q4_K_M / CRACK-Q4_K_M |
| E-G4 A/B | Diffuspace/Clover-35B-A3B-Preview-GGUF / AtomicChat/Clover-35B-A3B-Preview-GGUF | Q4_K_M / AD-Q4_K-IQ4_XS |
| E-G5 A/B | unsloth/GLM-4.7-Flash-GGUF | UD-Q2_K_XL / UD-Q3_K_XL |
| E-G6 A/B | mradermacher/Nemotron-Cascade-2-30B-A3B-i1-GGUF（源 nvidia/Nemotron-Cascade-2-30B-A3B） | i1-Q2_K / i1-Q3_K_M |
| E-G7 A | KRAFTON/Raon-1-mini-preview-gguf（源 KRAFTON/Raon-1-mini-preview） | IQ4_XS |

### 6.2 下载命令

```powershell
# 单文件（HF 镜像）
curl.exe -L -o Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf `
  https://hf-mirror.com/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf

# 跨模型批量（G2–G7，含 F16/BF16 源）
python tcq_exp/e_g_dl.py --only g2,g3 --with-hf
python tcq_exp/e_g_dl.py --only g4,g5
python tcq_exp/e_g_dl.py --only g6,g7 --with-hf
```

## 7. 基准数据

本地目录 `tcq_exp/data/`（约 282 MB，含 LongMemEval 265 MB）。来源与校验：

| 文件 | 字节 | SHA256（本地文件） |
|---|---|---|
| `gsm8k_full.jsonl` | 802,067 | `323951F6EFCFC3EB70A805D9FBACA43B35F7E5F6B3415932EC8D15F541656EEF` |
| `gsm8k_200.jsonl` | 120,328 | `7529B30D3DEF771DCEEB4B4D21DEBC2477382665FC3A58F51BB781341666F523` |
| `humaneval_164.jsonl` | 179,219 | `07F391DF5D1F9125A4C0DDC12B63B2A3694B05A0EC7A75A712FADA2BF9668A2D` |
| `gpqa100.jsonl` | 70,640 | `A0F2B8B06FE89EC3930D4C56FD22BE40DB7855CFBB9C88D2281FFFDA5DB14A85` |
| `math500_100.jsonl` | 28,939 | `96771B1D0353E0D2180F309FB81AF5B6519F4A507D50632536CF9B254F14D2AB` |
| `aime30.jsonl` | 18,462 | `8DA548A1191D0EB51D9C9D336E7CB747E133FDB182F0D282036B5792DB4DA4E0` |
| `longmemeval_oracle.json` | 15,388,478 | `821A2034D219AB45846873DD14C14F12CFE7776E73527A483F9DAC095D38620C` |

| 文件 | 来源 | 版本/协议 | 许可 |
|---|---|---|---|
| `gsm8k_full.jsonl`、`gsm8k_200.jsonl`、`gsm8k_test.parquet` | `openai/grade-school-math`（GitHub） | test split 1319 题 | MIT |
| `humaneval_164.jsonl`、`humaneval.parquet` | `openai/human-eval`（GitHub，HumanEval.jsonl.gz） | 164 题 | MIT |
| `gpqa100.jsonl` | GPQA Diamond 前 100 题（`Idavidrein/gpqa`） | 4 选 1 | CC BY 4.0 |
| `math500_100.jsonl` | `HuggingFaceH4/MATH-500` 前 100 题 | — | MIT |
| `aime30.jsonl` | AIME 2024/2025 公开试题 30 题 | — | 无正式开源许可（竞赛题，研究用途） |
| `longmemeval_oracle.json` / `longmemeval_s.json` | `xiaowu0162/longmemeval`（hf-mirror） | oracle 证据集 / 全量 115k | MIT |
| `alphabet` 系（Alpaca 派生）`general_*`、`prose_*`、`trans_*`、`summ_*` | `tatsu-lab/stanford_alpaca` | 52k | CC BY-NC 4.0 |
| BFCL-v3 | EvalScope 1.14.1 + bfcl-eval 2025.10.27.1 | multi_turn_base / long_context | Apache-2.0 |

`tcq_exp/data` 与 `data_extract` 均未入库；`make_data.py` / `make_data_stage2.py` 可从上述原始数据重建（需先手动放置原始仓库）。

---

## 8. 未入库大文件与获取方式

| 类别 | 体积 | 是否入库 | 获取方式 |
|---|---|---|---|
| 外部 GGUF（§6） | ≈0.9 TB | 否 | HF / ModelScope，见 §6 命令 |
| F16/BF16 参考与 TCQ 自建制品 | ≈0.6 TB | 否 | F16/BF16 从上游下载；TCQ 制品用 `e_g1_build.py` / `e_g_build.py` + imatrix 重建 |
| `tcq_exp/stores`（INT2/3/4 打包库） | 90.18 GB | 否 | `build_packed_stores.py` / `build_low_stores.py` 重建 |
| 路由概率 `probs_*.bin` | 3 × 448 MB + 2 × 18.7 MB | 否 | `r2_run.py` / `nr_run.py` 重跑导出 |
| imatrix（`e_g1/e_g2/e_g3_imatrix.gguf`） | 0.106 / 0.384 / 0.183 GB | 否 | `e_g1_build.py` 等从校准语料重建 |
| `longmemeval_s.json` | 265 MB | 否 | `xiaowu0162/longmemeval` |
| 全部模型目录 | 1997.6 GB | 否 | 见上 |

校验命令（对任意本地制品）：

```powershell
Get-FileHash -Algorithm SHA256 <file>          # 慢
& 'C:\Program Files\Git\usr\bin\sha256sum.exe' <file>   # 快
```

---

## 9. 已知缺口（复现时必须知道）

1. **torch 缺位**：基础解释器 3.13.14 无 torch；`moe_lib.py`、`int4_ext`、`d1*`、`exp15` 等无法直接运行。现有 `py311`（torch 2.13.0）与 `unsloth`（2.6.0+cu124）均非历史环境；历史记录中出现过 2.4.1+cu124。建议固定 `torch==2.6.0+cu124`（与 CUDA 12.4 匹配）并重跑 L3。
2. **evalscope / bfcl-eval 未安装**：E-A1a 的 BFCL 评测依赖 `evalscope==1.14.1`、`bfcl-eval==2025.10.27.1`，基础环境不可用。
3. **`gguf` 无 pip 固定版本**：来自 `llama.cpp/gguf-py`，随基线 commit 变化。
4. **AIME 30 无正式许可**；论文仅作研究性使用。
5. **Windows 绝对路径**：部分脚本含 `E:\CUDA11.8\...`，迁移到其他机器需改 `MODEL_ROOT` / `DATA`。
6. **E-GPTQ 未产出数字**：本机工具链阻塞（`gptqmodel` 需 torch≥2.13 且 `pypcre` 构建失败），结果目录仅有预注册与脚本。

---

## 10. 文件清单

| 文件 | 用途 |
|---|---|
| `REPRODUCIBILITY.md` | 本文件 |
| `requirements.txt` | 基础层依赖（可直接 pip 安装） |
| `repro_check.ps1` | 环境自检 / 可选冒烟 |
| `llamacpp/*.patch` | 双仓改动 |
| `results/` | 论文结果文件 |
| `code/` | 实验脚本（含 vendored `moe_lib.py`） |
