# E-R1 预注册（跑前写死）

日期：2026-11-18（Asia/Shanghai）
实验：Reverse 名单对照臂（补丁名单维度的收尾臂）
目录：`E:\CUDA11.8\train\reverse\`（隔离；只读引用 `tcq_exp/` 产物，不改动其文件）

## 1. 对照与变量

| 臂 | 名单（10/40 层专家升档） | 来源 |
|---|---|---|
| G2（usage 基准） | blk.0–9 | `tcq_exp/results/e_h1_types_meta.json` |
| G_rand（随机） | seed 20261104 随机 10 层 | `tcq_exp/results/c1_grand_meta.json` |
| **Reverse（本臂）** | 全局 usage 最低的 10 层 | 本目录 `results/c1_reverse_meta.json` |

单变量：类型模板（experts IQ2_XXS / 名单层 IQ4_XS / 非专家 Q5_K）、imatrix
（`im_global2.gguf`）、预算 11.45 ± 0.50 GiB、量化器（llama-upstream CUDA build）
全部与 G_rand 相同，**只改名单**。

名单粒度说明：GGUF 每层专家是单个三维张量，`--tensor-type-file` 不能逐专家指定，
因此构建与 G_rand 一样落在**层粒度（10/40 层）**。评分口径：
`score(l) = Σ_e Σ_d usage[l,e,d]`（sens_matrix，E1 产物），取升序前 10 层。
规格中的"每层最低 usage 64 专家"按域计算后写入 meta（`per_domain_bottom64`），
但本次构建不可表达；这一点与 C1.2 G_rand 完全同构，因此对照仍然干净。

## 2. 评测协议（与 C1.2 同口径，不重跑基线）

- GSM8K 200 题 + HumanEval 164 题；greedy；`n_predict=1024`（relaxed）；`-t 16`；
  `--cpus 0-15`（启动后 affinity 0xff）；同进程顺序起停，单 server；逐题 jsonl 断点续跑。
- 基线逐题结果直接引用：`tcq_exp/results/c1_g2_tasks.jsonl`（G2）、
  `tcq_exp/results/c1_grand_tasks.jsonl`（G_rand），与 C1.2 报告口径一致。
- 统计：McNemar 精确检验（双侧二项）+ 1000 次序列级 bootstrap（seed 20261104），
  名义 p，未做多重校正；配对按 (task, task_id)。

## 3. 功效声明（先于结果）

按 E-F7 实测口径：GSM8K 200 题 MDE ≈ ±5.6 pt；HumanEval 164 题 MDE ≈ ±4.2–6.5 pt。
判定只认显著性（p<0.05 且 bootstrap CI 不含 0），**不认点估计方向**。

## 4. 判定表（跑后对号）

| reverse vs G1/G2（对照） | 解读 | 后续 |
|---|---|---|
| GSM8K 或 HE 显著更差 | 热专家保护方向性确认；G_rand（中性）+ reverse（负）+ usage（基准）构成名单单边谱闭环；"冷专家精准化"在补丁范式下证伪 | 名单维度关闭；相关工作写"名单方向性已封口" |
| 双任务均无差异 | 与 G_rand 一致，名单只影响 HE 单点（C1.2 结论巩固） | 名单维度关闭；q 工程成为粒度细化唯一活路 |
| reverse 显著更好 | 意外，冷专家精准化复活；查名单与 B4"code vs NL 边界"交互（冷专家是否恰好是 code 侧专家） | 升级为独立小论文候选 |

主对照：**reverse vs G2**（usage 基准）；次对照：**reverse vs G_rand**。
两个对照分别报告 McNemar b/c/p 与 bootstrap CI；任一任务显著即按对应行解读，
双任务均不显著才落入第二行。

## 5. 产物

```
reverse/results/c1_reverse_meta.json        名单 + 体积校验
reverse/results/c2_reverse_tasks.jsonl      逐题（task/task_id/acc/...）
reverse/results/c2_reverse_summary.json     acc + CI + 两个配对的 McNemar
reverse/results/c2_reverse_pairwise.json    配对明细
reverse/results/c1_reverse_build.log        量化日志
reverse/PROGRESS.md                         执行记录
```

本文件在构建与评测开始前写入；后续不得修改判定口径与阈值。
