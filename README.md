# moe-quant-mechanisms

> **Where Low-Bit Quantization Damages MoE Models: A Seven-Family Mechanism Study with an Operability Demonstration**
> Manuscript under review at *Neurocomputing*, 2026.

Low-bit quantization of Mixture-of-Experts models is supplied through two tracks — vendor "official" checkpoints and the community GGUF ecosystem — yet **where** and **when** quantization damages quality remains unmapped. This repository is the code, data ledger, and llama.cpp integration behind a seven-family mechanism study that answers those questions, plus **TCQ**, a pipeline derived mechanically from the resulting mechanism map.

---

## 三大主张

论文在七个 2026 年 MoE 家族、五种注意力范式上，用配对协议加实测统计功效，给出三个结论：

### (i) 归因：任务条件性就是路由暴露

把专家敏感度分解为 `s = usage × q`（`q` 为内在可量化性）。跨域方差 **103.6% 来自路由用量、0.3% 来自内在可量化性**（份额超过 100% 是负协方差项）。

- 「哪些专家怕量化」是权重的属性（跨域 Pearson 0.92–0.96）；
- 「哪些专家重要」是路由的属性。
- 预算匹配消融把社区前沿拆解成组件阶梯：**只有 type-budget 重分配在任务级可检测**，per-layer / per-tensor 增量停留在似然分辨率，随机列表对照进一步关闭了补丁列表排序的问题。

### (ii) 四层损伤图

| 层 | 损伤载体 | 关键测量 |
|---|---|---|
| **Token** | 码本失配承载跨域损伤 | 均匀码本下 math 补丁在 HumanEval 崩到 0.579；**Lloyd-Max 修复 +30~36 pt**（0.579→0.939、0.567→0.902） |
| **Route** | 逐 token 专家集漂移大、聚合排序稳 | top-8 精确匹配仅 **18.7%**（Jaccard 0.71），但逐层 top-64 重叠 **91.2%** → 逐 token 路由脆弱，usage 矩阵稳定 |
| **Task** | 饱和带 + 五个边界 | 校准缺失（Xing −12 pt）、多跳链（RULER-mh: base2 0.10）、竞赛难度（AIME 丢终止）、比特率敏感任务（GLM GPQA）、行为分级退化 |
| **Behavior** | 分级退化 + 校准冲突异常 | Qwen3.6 五档 Q2→Q6 准确率跨度仅 3.0 pt；**Nemotron 上 imatrix 反而有害**（GSM8K 0.305 vs imatrix-off 0.715） |

外加一个**四家族校准谱**：Qwen3.6 有效 / MiniCPM 弱（方向性）/ Gemma-4 无效 / Xing 必需。校准是否有效是一阶问题，用哪套语料是二阶。

### (iii) 可操作性演示：TCQ

一条**每个设计决策都是机制图直读结果**的流水线，在 **10–30 CPU 分钟**内达到与社区前沿构建在任务级无法区分的质量。论文明确声明这是**存在性证明，不是最优性主张**——同体积下其分布保真度落后 near-volume 社区版约 5×，这一点论文直接报告。

## 七个家族与判定

| 模型 | 架构 | 对照基线 | 判定 |
|---|---|---|---|
| Qwen3.6-35B-A3B | GDN (Gated-DeltaNet) | iq2xxs, UD-Q2 | **GSM8K-1319 TOST 等价**（+0.23 pt，CI [−0.61, +1.06] ⊂ ±1.5 pt） |
| Gemma-4-26B-A4B | GQA | QAT, UD-Q4 | 平坦，且体积 −20% |
| MiniCPM-V 4.7 35B-A3B | GDN | Q4_K_M, mix | 不劣于 |
| Xing4.0-29B-A4B | MLA | iq4_xs, community | **校准修复**（缺校准 −12 pt，自采 imatrix +14 pt） |
| GLM-4.7-Flash 30B-A3B | MLA | UD-Q2, UD-Q3 | 任务×组件交互（GPQA +9 pt/档） |
| Nemotron-Cascade-2 30B-A3B | Mamba2 | i1-Q2_K, i1-Q3_K_M | **校准冲突**：imatrix 有害（−22 pt），imatrix-off 稳定 0.970 |
| Raon-1-mini-preview 28.8B-A4B | RDA+MLA | iq4_xs | 逐点 ≥ |

统计功效口径：配对 McNemar + 1000 次序列级 bootstrap，GSM8K-200 的 MDE 为 **±5.6 pt**、HE-164 为 **±4.2–6.5 pt**。全文的「无差异」「平坦」都是**分辨率陈述，不是等价性主张**，唯一的 TOST 等价在 GSM8K-1319 上。

## TCQ：设计决策与机制依据

| 设计决策 | 机制依据 | 验证 |
|---|---|---|
| 2-bit Lloyd-Max 底座（2.125 bpw） | token 层码本失配 | E21 wMSE / NLL、C2 KLD |
| 按路由质量 top-64 选补丁专家 | 路由暴露归因 | H1 组件阶梯、C1.2 随机列表、E-F3 |
| imatrix 默认开（ISQ 码本） | 校准谱 | C1.3 on/off 矩阵 |
| 单文件多补丁 + 指针切换 | 饱和带的部署价值 | E-H4b bound set |
| 跨模型可移植 | 饱和带的普适性 | G5–G7 配对 |

系统实现要点：base 与全部域补丁装在**一个文件**里（23.01 GiB 全载 / 14.57 GiB 单补丁 bound set）；请求级 `domain` 字段只改写逐层**索引表指针**（不动权重字节），微秒级，端到端切换秒级；`mul_mat_id` 混合类型节点（base+patch 同节点，100 轮交替 `max|dNLL| = 0`）+ 2-bit AVX2 LUT 内核（`vpermps`），补丁专家按原生 Q4_K 速度跑。冷启动 4.1 s（23 GiB 文件），decode 7.11 t/s（较 iq2xxs **+17%**），prefill 是最弱轴。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `code/tcq_exp/` | 主体实验脚本（112 个，含 vendored `moe_lib.py`），`a*` 换装与运行时、`b*` 路由分离度、`c*` 统计与配对、`d*` 残差谱与换装代价、`e*` 端到端评测 |
| `code/tcmoe/` | 单文件多补丁容器打包与逐位一致性校验 |
| `code/reverse/` | 由容器反解类型表、配对评测 |
| `code/route_consistency/` | 路由一致性复测 |
| `code/nemotron_ext/` | Nemotron 外延实验（含 `route/`） |
| `code/gptq_baseline/` | GPTQ 基线量化与评测 |
| `docs/` | 五组实验的**预注册 PREREG** 与**执行记录 PROGRESS** |
| `llamacpp/` | 上游与 TCQ 两套补丁 + `llama-tcmoe.{h,cpp}`、`tcmoe-verify` |
| `results/` | 各模块原始结果，按 `results/<module>/results/` 组织 |

## 关键文档

| 文档 | 说明 |
|---|---|
| [`C5_method_framework.md`](C5_method_framework.md) | §5 方法框架：符号表、数学定义、算法与代码锚点 |
| [`CODE_INDEX.md`](CODE_INDEX.md) | 逐文件说明（取每个文件首段 docstring） |
| [`EXPERIMENT_INVENTORY.md`](EXPERIMENT_INVENTORY.md) | 实验总台账，含状态与逐项结果 |
| [`DATA_TABLE.md`](DATA_TABLE.md) · [`DATA_TABLE.csv`](DATA_TABLE.csv) | 按实验编号排列的数据表，附代码位置与结果文件路径 |
| `docs/<track>/PREREG.md` | 各组实验的先验假设与判定标准（跑之前写） |
| `docs/<track>/PROGRESS.md` | 执行过程与偏差记录 |

## 快速上手

```bash
# 容器逐位一致性校验
cd llamacpp/new-files/llama.cpp/tools/tcmoe-verify && cmake -B build && cmake --build build

# 打补丁到 llama.cpp
git -C /path/to/llama.cpp apply llamacpp/llama.cpp-tcmoe.patch
```

复现依赖：Python 3（`numpy` / `torch`，部分脚本需 `llama-cpp-python` 或直接驱动 `llama-server`）、量化后的 GGUF 权重（路径参数传入，不入库）、以及 GSM8K / HumanEval / GPQA / AIME / RULER 评测数据（按各脚本 docstring 的格式准备）。每个脚本首段 docstring 说明了输入输出与运行方式。

## 边界与已知局限

- **Nemotron 的校准冲突**是机制图里最极端的校准单元，其路由漂移是 Qwen 的 8× 却不产生任务代价——递归状态架构上二者为何解耦仍是开放问题。
- 无校准基线（HQQ/SINQ）与误差补偿量化器（GPTQ/AWQ 类）在 GGUF 生态内没有对应产物，相关比较停留在文献层面；**全部结论限于 RTN + imatrix**。
- 任务条件化配额分配在所研究的模型、码本、预算与任务下**失败**，向此外的迁移未验证。
- 轨迹级单元格为 n=20；math 补丁的点估计偏移（−7.5 pt）方向一致但不显著，属探索性观察。
- 硬件限于两台单机（双路 Xeon；RTX 5000/4070），不做跨硬件主张。

## 路径约定与结果文件

仓库根的 `code/` 对应原工作目录。文档中出现的不带前缀的脚本路径（如 `tcq_exp/e_f2_gsm8k.py`）**均相对于 `code/`**（实际位置 `code/tcq_exp/e_f2_gsm8k.py`）。

各模块的原始结果（JSON / JSONL / CSV / SVG）随仓库提供，共 **157 个文件**，位于仓库根的 `results/`：

| 目录 | 文件数 | 内容 |
|---|---|---|
| `results/tcq_exp/results/` | 148 | 组件阶梯、KLD、配对检验、码本拟合、换装代价、逐题输出、跨模型汇总 |
| `results/nemotron_ext/results/` · `route/results/` | 4 | Nemotron 位宽曲线与路由统计 |
| `results/reverse/results/` | 3 | 容器反解元数据与配对评测 |
| `results/route_consistency/results/` | 2 | 逐层路由一致性 |

`DATA_TABLE.md` 中每一行的结果路径可直接对照定位。ERC（残差补偿）实验按导出约定不含在内；`_tmp_*`、`*smoke*` 等调试产物亦未纳入。

## 引用

```bibtex
@article{moe_quant_mechanisms_2026,
  title  = {Where Low-Bit Quantization Damages MoE Models:
            A Seven-Family Mechanism Study with an Operability Demonstration},
  note   = {Manuscript under review at Neurocomputing},
  year   = {2026}
}
```

## License

[Apache-2.0](LICENSE)
