# moe-quant-mechanisms

**Mechanism-guided low-bit quantization for MoE models on llama.cpp/GGUF** — Lloyd-Max codebooks, per-expert mixed precision, and single-file multi-patch containers with µs-level domain switching.

面向 MoE 大模型的任务条件化量化（Task-Conditioned Quantization, **TCQ**）：利用专家路由的领域差异性，为每个专家分配按域定制的精度配额，并在 llama.cpp/GGUF 上以「单文件多补丁容器 + 运行时域切换」的形式落地。

---

## 核心思路

MoE 模型在不同任务下激活的专家子集高度不同，这一现象贯穿全流程：

1. **路由是域条件化的。** 同一层里，数学、代码、通用三类输入命中 top-k 专家的重叠度显著低于随机基线，且这种分离在层深方向上有稳定结构。
2. **所以精度预算也应当域条件化。** 与其对全体专家用统一位宽，不如先按路由用量（`usage`）、敏感度（`s`）、加权代理（`q̂`）排出配额，再把高价值专家升级到 INT4、其余压到 INT2/INT3。
3. **多域要能共存。** 每个域训练一份「补丁」（patch），与共享的低比特底座一起装进同一个 GGUF 容器；运行时按域切换补丁，切换代价做到微秒级，无需重载模型。

围绕这条主线，仓库覆盖了从机制测量（路由分离度、残差谱、换装代价）到工程实现（低位宽打包、嵌套 scale、码本量化）再到端到端评测（GSM8K / HumanEval / GPQA / AIME / BFCL / KLD）的完整链路。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `code/tcq_exp/` | 主体实验脚本（111 个），按 `a*` / `b*` / `c*` / `d*` / `e*` 编号组织 |
| `code/tcmoe/` | 单文件多补丁容器的打包与逐位一致性校验 |
| `code/reverse/` | 逆向实验：由容器反解类型表、配对评测 |
| `code/route_consistency/` | 路由一致性复测 |
| `code/nemotron_ext/` | Nemotron 系列模型的外延实验（含 `route/` 子目录） |
| `code/gptq_baseline/` | GPTQ 基线量化与评测 |
| `docs/` | 各实验组的**预注册（PREREG）** 与 **执行记录（PROGRESS）** |
| `llamacpp/` | 对上游 llama.cpp 的补丁（`git diff`）+ 新增源文件副本 |

## 关键文档

| 文档 | 说明 |
|---|---|
| [`C5_method_framework.md`](C5_method_framework.md) | §5 方法框架：符号表、数学定义、算法与代码锚点 |
| [`CODE_INDEX.md`](CODE_INDEX.md) | 147 个源文件的逐文件说明（取每个文件首段 docstring） |
| [`EXPERIMENT_INVENTORY.md`](EXPERIMENT_INVENTORY.md) | 实验总台账：E1–E21、B / D / A / C 路线、E-F / E-G / E-H 主线、E-R / E-N |
| [`DATA_TABLE.md`](DATA_TABLE.md) · [`DATA_TABLE.csv`](DATA_TABLE.csv) | 按实验编号排列的数据表，每行附代码位置与结果文件路径 |
| `docs/<track>/PREREG.md` | 各组实验的先验假设与判定标准（跑之前写） |
| `docs/<track>/PROGRESS.md` | 执行过程与偏差记录 |

## 实验覆盖

已在 6 个 MoE 架构上完成端到端评测，对照官方或社区的低比特基线（`iq2xxs`、`UD-Q2_K_XL`、`UD-Q3_K_XL`、`UD-Q4_K_M`）：

| 模型 | 规模 | 主要配置数 | 参考实验 |
|---|---|---|---|
| Qwen3-30B-A3B | 30B / A3B | 4 配置 | E-F 系列 |
| gemma-4-26B-A4B | 26B / A4B | 6 配置 | E-G1 |
| MiniCPM-V-4.7-35B-A3B | 35B / A3B | 6 配置 | E-G2 |
| Xing4.0-29B-A4B | 29B / A4B | 7 配置（含 2 个社区基线） | E-G3 |
| GLM-4.7-Flash 30B-A3B | 30B / A3B | 5 配置 | E-G5 |
| Nemotron-Cascade-2-30B-A3B | 30B / A3B | 5 配置 | E-G6 |

评测集包括 GSM8K（含全量 1319 题）、HumanEval（pass@1 与 `tests_passed` 连续指标）、GPQA、AIME30、BFCL，以及 token 级 KLD / JSD / top-1 对齐。所有配对比较均报告置信区间与 McNemar / bootstrap 检验，明细见 `DATA_TABLE.md`。

几个具代表性的读数（均摘自实验台账，非复述结论）：

- E-F2：GSM8K 全量，TCQ 配置 0.9515 vs `iq2xxs` 0.9492（+0.23 pt，CI95 [−1.21, +0.83]，McNemar p=0.766）。
- E-F9：token 级 KLD(ALL)，`iq2xxs` 0.3103 → `nl_global` 0.1466，top-1 一致率 0.845 → 0.882。
- E-F4：Qwen3-30B-A3B 的 CPU prefill（pp512 中位）`iq2xxs` 31.24 tok/s，`tcmoe_q4k` 20.14 tok/s。
- E-F10b：imatrix 加权共享码本把加权 WMSE 从 3.76e−5 降到 1.36e−5（−64%），但 HumanEval 代码侧出现 −2.44 pt 回退，判定为 partial。

## llama.cpp 集成

`llamacpp/` 提供两套补丁，基线版本的 commit 在各文件行内注明：

| 文件 | 作用 |
|---|---|
| `llama-upstream.patch` | 对上游 llama.cpp 的改动（`git diff` 形式） |
| `llama.cpp-tcmoe.patch` | TCQ 容器、补丁装载与域切换 |
| `new-files/llama.cpp/src/llama-tcmoe.{h,cpp}` | 多补丁容器的核心实现 |
| `new-files/llama.cpp/tools/tcmoe-verify/` | 容器逐位一致性校验工具 |
| `new-files/*/examples/eval-callback/route-dump.cpp` | 路由与专家命中导出 |

## 关于结果文件

`tcq_exp/results/` 下的原始结果（JSON / JSONL / 日志 / 图表）**未随仓库分发**——总体积远超代码本身。`DATA_TABLE.md` 中每一行都标注了对应结果文件的相对路径，需要时可按该路径在本地复现目录中定位。

## 复现

代码不含硬编码的绝对路径，但依赖以下外部条件：

- Python 3 环境，含 `numpy` / `torch`；部分脚本需要 `llama-cpp-python` 或直接驱动 `llama-server`
- 量化后的 GGUF 模型权重（脚本以路径参数传入，不入库）
- 评测数据：GSM8K、HumanEval、GPQA、AIME、BFCL 需自行按各脚本 docstring 中的数据格式准备

每个脚本首段的 docstring 说明了输入、输出与运行方式，`CODE_INDEX.md` 是汇总入口。

## License

[Apache-2.0](LICENSE)
