# E-R1 执行记录（Reverse 名单对照）

状态：**已完成**；判定：**双任务均无差异（名单维度关闭）**
日期：2026-11-18；目录：`E:\CUDA11.8\train\reverse\`（隔离，未改动 `tcq_exp/`、`coserving/`）

## 1. 构建

| 项 | 值 |
|---|---|
| 名单层（usage 升序前 10） | l8、l9、l31、l32、l34、l35、l36、l37、l38、l39 |
| 层分 `score(l)=Σ_e Σ_d usage` | 606 / 616 / 540 / 554 / 532 / 404 / 393 / 414 / 361 / 416 |
| 模板 | experts IQ2_XXS / 名单层 IQ4_XS / 非专家 Q5_K（与 G2、G_rand 相同） |
| imatrix | `E:\CUDA11.8\model\e20\im_global2.gguf` |
| 预估体积 | 11.641 GiB（G2 同值，dev 0.000） |
| 实际构建 | rc=0，24.1 min，**11.652 GiB**，753 张量/元素数一致，within_tol |
| 产物 | `out/c1_reverse_im.gguf` |

与 G_rand 的构建体积（11.652 GiB）逐字节同量级，类型模板相同，只换名单 → 单变量对照成立。

限制（与 C1.2 相同）：GGUF 专家张量为每层一个三维张量，`--tensor-type-file` 无法逐专家
指定，因此名单落在层粒度。规格中的"每层每域 usage 最低 64 专家"已写入
`results/c1_reverse_meta.json::per_domain_bottom64`，但构建不可表达。

## 2. 评测

协议与 C1.2 基线完全一致（`c1_run_tasks.py` 同款：greedy、n_predict=1024、-t 16、
`--cpus 0-15`（affinity 0xff）、单 server 顺序起停、逐题 jsonl 断点续跑、
llama-upstream CUDA build、-ngl 99）：

| 任务 | n | 正确 | acc | 用时 |
|---|---|---|---|---|
| GSM8K | 200 | 192 | **0.960** | 16.4 min |
| HumanEval | 164 | 149 | **0.909** | 9.6 min |

总评测 26.0 min（远低于预估 3.5 h，本机 GPU 路径未受 CPU 限速）。

## 3. 配对统计（McNemar 精确 + 1000 次 bootstrap，seed 20261104）

| 对照 | 任务 | n | b | c | p | acc 基线→reverse | Δ | 95% CI |
|---|---|---|---|---|---|---|---|---|
| vs G2（usage 基准） | GSM8K | 200 | 1 | 5 | 0.2188 | 0.940 → 0.960 | +0.020 | [0.000,+0.040] |
| vs G2 | HumanEval | 164 | 5 | 6 | 1.0000 | 0.902 → 0.909 | +0.006 | [−0.031,+0.049] |
| vs G_rand | GSM8K | 200 | 3 | 4 | 1.0000 | 0.955 → 0.960 | +0.005 | [−0.020,+0.030] |
| vs G_rand | HumanEval | 164 | 7 | 12 | 0.3593 | 0.878 → 0.909 | +0.030 | [−0.018,+0.085] |
| vs G1 | GSM8K | 200 | 3 | 6 | 0.5078 | 0.945 → 0.960 | +0.015 | [−0.015,+0.045] |
| vs G1 | HumanEval | 164 | 8 | 4 | 0.3877 | 0.933 → 0.909 | −0.024 | [−0.067,+0.018] |

功效声明（先于结果）：GSM8K ±5.6 pt、HE ±4.2–6.5 pt（E-F7 口径）。上表所有 |Δ| 均小于
MDE，且 p 全部 ≥0.05、CI 全部包含 0。

## 4. 判定

落入预注册判定表第二行：**双任务均无差异**。

- 与 G_rand（中性）一致：名单选择只影响 HE 单点，且本臂相对 G_rand 的 HE +3.0 pt
  与 G2/G_rand 之间的既有差异同量级，不显著。
- 方向性：reverse 在 GSM8K 上数值最高（0.960 vs 0.940/0.955/0.945），但 b=1、c=5，
  仅 4 题净差，远低于 MDE，不构成"冷专家精准化复活"的证据。
- 名单维度（层粒度）至此关闭：usage 基准、随机、reverse 三点共线，无显著分化；
  q 工程（粒度细化）成为唯一活路。

## 5. 产物

```
reverse/PREREG.md                          预注册（跑前）
reverse/reverse_types.py                   名单生成 + 构建 + 校验
reverse/run_tasks.py, e_g1_eval.py         评测（复制并改到本目录，产物隔离）
reverse/pair_eval.py                       配对统计
reverse/h1_types/c1_reverse.txt            类型文件
reverse/out/c1_reverse_im.gguf             11.652 GiB
reverse/results/c1_reverse_meta.json       名单/体积/校验
reverse/results/c2_reverse_tasks.jsonl     364 条逐题
reverse/results/c2_reverse_summary.json    acc + CI + 三组配对
reverse/results/c2_reverse_pairwise.json   配对明细
reverse/results/c1_reverse_build.log       量化日志
reverse/results/c2_reverse.run.log         评测日志
```

## 6. 备注

- 名单与 G2（blk.0–9）在 l8、l9 上重叠；G2 并非严格 usage top-10（usage 排名最高的是
  l0–l3、l11、l23 等），因此本臂相对 G2 不是严格补集。若后续要更干净的"最高 vs 最低"，
  应以 usage top-10 重建基准臂；当前判定不受影响（两种解读下均无显著差异）。
- 复现命令：
  ```powershell
  python reverse\reverse_types.py --write --build
  python reverse\run_tasks.py --model reverse\out\c1_reverse_im.gguf --tag reverse_im `
      --prefix c2_ --out reverse\results\c2_reverse_tasks.jsonl `
      --log reverse\results\c2_reverse.run.log --ports 18231,18232
  python reverse\pair_eval.py --target reverse\results\c2_reverse_tasks.jsonl `
      --baseline g2=tcq_exp\results\c1_g2_tasks.jsonl `
      --baseline grand=tcq_exp\results\c1_grand_tasks.jsonl `
      --baseline "g1=tcq_exp\results\e_h1_g1_im_{task}.jsonl" `
      --out reverse\results\c2_reverse_pairwise.json
  ```
