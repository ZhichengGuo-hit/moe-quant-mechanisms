# E-N1 预注册：Nemotron 档位扩展（M6 收尾臂）

冻结时间：2026-11-18（在分析脚本运行前写定判定规则）
数据来源：`tcq_exp/results/c4_nemotron_sweep.jsonl`（2026-11-16 采集完成，5 档 × 100 题）
本目录隔离：只读引用 `tcq_exp/` 与 `tcq_exp/results/`，不写、不改原目录。

## 1. 问题

C4 此前只给出两档证据（Q3_K_S / Q3_K_M）支持"低比特 CoT 断崖"。M6 要求把证据从
单点两档升为曲线：第三/四档验证断崖位置与单调性；同时给出与 Qwen3.6（GDN 混合）
C4 曲线的并置，用于回答"为何同架构不同族"。

## 2. 协议（与 C4 完全一致，数据已按此采集）

- 模型：`Nemotron-Cascade-2-30B-A3B` i1 系列 GGUF，四档
  `i1-Q2_K / i1-Q3_K_S / i1-Q3_K_M / i1-Q4_K_M`，另加 `i1-Q6_K` 作高锚。
- 任务：GSM8K 前 100 题（`gsm8k-0 .. gsm8k-99`）。
- 推理：llama-server，`-ngl 20`、ctx 4096、temperature 0、top_k 1、
  n_predict 1024、`cache_prompt=false`；Nemotron 自带模板，保留思维链。
- 行为分类：C4 规则分类器（truncated > loop > reasoning > direct）。

## 3. 指标

1. acc（附 Wilson 95% CI）；配对 McNemar（Q2_K vs Q3_K_S）。
2. CoT 长度：`predicted_n` 的均值/中位数，序列级 bootstrap 95% CI。
3. 终止模式分布：reasoning / direct / truncated / loop。
4. 相邻档差分：`Δacc` 与 `pn_ratio = mean_pn(next)/mean_pn(prev)`。

## 4. 判定线（跑前冻结）

- **断崖判据**（沿用 C4）：相邻两档 `Δacc ≤ -15 pp` **或** `pn_ratio ≤ 0.70`。
- **A. 断崖成立**：Q2_K→Q3_K_S 满足断崖判据，且 Q3_K_S 及以上恢复
  （Q3_K_S、Q3_K_M、Q4_K_M 的 acc 均不低于 Q2_K，CoT 长度不低于 Q2_K 的 85%）
  → 保留"不连续模式切换"机制断言，M6 关闭。
- **B. 四档渐变**：四档内不存在满足断崖判据的相邻差分（acc 极差 < 15 pp 且
  `pn_ratio` 全部 > 0.70）→ 异常缩窄为"该族对低比特敏感"，机制断言降级为
  "Mamba2 族对位宽的敏感度远高于 GDN"，M6 关闭。
- **C. 其他**（非单调但无断崖 / 仅单侧）→ 如实报告曲线，判定降级为
  "不连续证据不足"，M6 以限定语关闭。

## 5. 与 Qwen3.6 并置的口径

Qwen3.6-35B-A3B 无同协议的 C4 扫描；用 E17 GSM8K（同 200 题集，取前 100 题与
Nemotron 对齐，1024 上限）的 6 个位宽臂构造可比的 acc–bpw / CoT–bpw 曲线：
`iq2xxs / ud_q2kxl / iq3xxs / ud_q3kxl / ud_q4kxl / q4km`。
**协议差异必须在结论中标注**：E17 为 200 题（此处取前 100）且 prompt 模板不同，
只用于同模型内的位宽趋势并置，不做跨模型绝对分数比较。

## 6. 统计纪律

- 只认配对显著性，不认点估计方向；acc 差异用 McNemar 精确检验，
  CoT 长度差异用 1000 次序列级 bootstrap。
- 100 题的 acc 分辨率约 ±4 pp（Wilson），单档 2–3 pp 的起伏不作为证据。
