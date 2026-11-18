# E-R2 预注册：路由一致性诊断（M5 的一半 + M6 的路由臂）

冻结时间：2026-11-18，在运行任何配置前写定判定规则与统计口径。
隔离目录：`route_consistency/`；只读引用 `tcq_exp/data`、模型文件与 llama.cpp 源码树，
不写 `tcq_exp/`、`coserving/`。

## 1. 要检验的前提

$s = \text{usage} \times q$ 隐含"损伤独立可加"，且默认量化误差不改变专家组合。
本实验直接测量：同一文本、同一 tokenizer 下，量化态与 BF16 态的路由是否一致。

## 2. 协议

- 文本：`tcq_exp/data/{math,code,general}_eval64.jsonl` 各 64 条（与 E1 同文本），
  按 math → code → general 顺序拼接，段间用 `%%__TEXT_BOUNDARY__%%` 分隔；
  单条文本超过 1024 token 截断（实际远小于此）。
- 逐 token、逐层记录 `ffn_moe_probs`（256 个专家的 softmax 概率，llama.cpp 主图）。
- teacher-forcing 下同时记录每个 token 的 NLL（本工具日志；配置间同协议）。
- 三配置（全部 CPU、16 线程，排除 GPU kernel 差异这个混杂因子）：
  1. **BF16 参考**：`E:\CUDA11.8\model\e20\qwen36_f16.gguf`，上游 llama.cpp，`-ngl 0`；
  2. **TCQ-nl_global**：`tcmoe/out/model-tcmoe-q4k-nl.gguf`，tcmoe fork，`-ngl 0`，
     不传 domain（GGUF 元数据默认 `global` 补丁）；
  3. **iq2xxs**：`model/baselines/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`，上游 llama.cpp，`-ngl 0`。
- ctx 4096、batch/ubatch 1024、temperature 0、无采样、关闭 MTP/NextN 旁路。
- 对齐：三配置的 token 序列必须逐位相同，否则报错中止（不做事后对齐）。

## 3. 指标

对每个 token（文本内位置 < 8 的 token 只进入 (a)(b)，不进入 (c)）：

- **(a) top-8 集合一致性**：精确匹配率（集合完全相等）、平均 Jaccard
  $|A\cap B|/|A\cup B|$；按层、按域给出。
- **(b) 路由权重分布差异**：全 256 维概率向量的 Jensen–Shannon 散度（bits），
  按层、按域给出均值与 95% CI（文本级 cluster bootstrap，1000 次）。
- **(c) 翻转 token 的下游代价**：
  $\Delta\mathrm{NLL}_t = \mathrm{NLL}^{quant}_t - \mathrm{NLL}^{BF16}_t$；
  比较"翻转 token"（top-8 集合不等）与"未翻转 token"的 $\Delta$NLL；
  统计量 = 两组均值差，文本级 cluster bootstrap 1000 次的 95% CI。

"系统方向"预注册定义：翻转组与未翻转组的 $\Delta$NLL 均值差 95% CI 不含 0，
且 $|\text{均值差}| \ge 0.01$ nats。

"翻转集中"预注册定义：按专家统计 $\overline{|\Delta p|}$，前 10 个专家占总变差
$\sum_e \overline{|\Delta p_e|}$ 的比例 ≥ 40%。

## 4. 判定线（预注册）

只以 **TCQ-nl_global** 为主臂（iq2xxs 为外部量化器对照），主指标取三域合并：

- **P1（前提成立）**：精确匹配率 ≥ 95%，且 (c) 无系统方向
  → "量化不改变专家组合"成立，M5 关闭（对本文有利）。
- **P2（量化诱导路由偏移）**：精确匹配率 85–95%，且翻转集中（前 10 专家 ≥ 40%）
  → 新增"量化诱导路由偏移"节，机制地图加第四层；M5 不关闭，写限定语。
- **P3（路由不可忽略）**：精确匹配率 < 85%
  → 归因章必须加路由偏移限定语；$s=usage\times q$ 仅作为近似。
- 中间态（如 85–95% 但翻转不集中）：按 P3 的限定语处理，但不新增机制节，
  记为"分散型路由扰动"。

## 5. 功效与分辨率

- 192 条文本、约 1.4 万 token、40 层 → 每层约 1.4 万个样本；
  合并精确匹配率的二项标准误约 0.2–0.4 pp，足以把 95% 与 85% 判定线分开。
- NLL 差的分辨率：同协议配对差，文本级 bootstrap；判定阈值 0.01 nats 为保守值。

## 6. 与既有结果的衔接

- 若 P1：E21/机制章可把"路由不变"作为 $s=usage\times q$ 的实证支撑；
- 若 P2/P3：M5 的回应从"假设"改为"已测量并限定"，并把翻转 token 的 NLL 代价
  作为"损伤非可加"的直接证据；
- Nemotron 路由臂（i1-Q3_K_M vs Q6_K）另立目录 `nemotron_ext/route/`，
  用同一工具与同一指标，判定线同上。
