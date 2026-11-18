# 代码清单

生成时间：2026-11-18；扫描 `tcq_exp/`、`tcmoe/`、`reverse/`、`route_consistency/`、`nemotron_ext/`、`gptq_baseline/`（实验脚本与 TCQ 打包工具），排除 `_tmp` 调试脚本、数据/结果目录与 `erc/`（开源导出不含 ERC 内容）；`llamacpp/` 为对上游 llama.cpp 的本地改动（`git diff` patch + 未跟踪新文件副本，基线 commit 见行内说明）；说明取每个文件首段 docstring（缺失时取首个注释行）。

| 文件 | 大小 KB | 说明 |
|---|---|---|
| `tcq_exp/a1_two_bit_runtime.py` | 11.4 | A1: 2bit 真实容器的运行时接入 + 内核吞吐测量。 |
| `tcq_exp/a2_delta_switch.py` | 7.6 | A2: 补丁间差集换装 + 预填充重叠的双缓冲异步换装。 |
| `tcq_exp/a3_switch_policy.py` | 3.0 | A3: lift 自适应换装决策表 (依赖 B6/B4 的域对 lift 与 CI)。 |
| `tcq_exp/analyze.py` | 5.3 | 实验分析: H1(Jaccard/Spearman)、代理有效性(ΔL相关)、H4(配额收益) -> results/summary.json |
| `tcq_exp/b1_topk_replay.py` | 10.2 | B1: Qwen3-30B-A3B 重放路由, 离线限制 top-k ∈ {2,4,8,16,32} (K/N = 1.6%..25%)。 |
| `tcq_exp/b2_layer_curve.py` | 4.9 | B2: 逐层分离曲线 (Jaccard / Spearman 随层深) 与层条件化补丁可行性。 |
| `tcq_exp/b3_concentration.py` | 5.9 | B3: 路由集中度画像 (exp(entropy)/Gini/top20 质量) 与逐层分离度 (1 - Jaccard) 的跨层相关。 |
| `tcq_exp/b4_domain_pairs.py` | 9.4 | B4: gemma-4-26B-A4B 新增域对 (code vs 翻译 / 摘要) 的 H1 分离度。 |
| `tcq_exp/b5_sparsity_curve.py` | 3.8 | B5: "稀疏性-分离度"曲线补点。 |
| `tcq_exp/b6_lift.py` | 5.0 | B6: 三模型逐域 top-20% Jaccard 的 lift 与 200 次单元 bootstrap 95% CI（统一口径）。 |
| `tcq_exp/bitpack.py` | 6.0 | 真实低位宽打包 (离线校验用, 不改 moe_lib/int4_ext 运行时)。 |
| `tcq_exp/build_e20_calib.py` | 2.7 | E20: 构建 per-domain imatrix 校准文本（math/code/general/global）。 |
| `tcq_exp/build_low_stores.py` | 4.3 | 构建低精度专家库(INT2/INT3), 布局与 int4_store 完全一致(group=128, nibble打包)。 |
| `tcq_exp/build_packed_stores.py` | 4.6 | 离线构建真实 3bit / 2bit 专家库 (密排格式, 仅供体积与一致性校验; 引擎不读取)。 |
| `tcq_exp/c1_pairwise.py` | 5.4 | !/usr/bin/env python |
| `tcq_exp/c1_rand_types.py` | 7.1 | !/usr/bin/env python |
| `tcq_exp/c1_run_tasks.py` | 5.0 | C1: 用一个臂跑 GSM8K200 + HumanEval164 (协议 = e17 / e_g1_eval, 与 E-H1c 完全相同)。 |
| `tcq_exp/c2_same_source.py` | 6.3 | C2: 同源题库 N-token 分类复测 (关闭 E14 的"数据源格式" caveat)。 |
| `tcq_exp/c3_eval128.py` | 5.8 | C3: eval128 主表复测 + general 按任务类型拆分。 |
| `tcq_exp/c4_awq_compensated.py` | 7.9 | C4: 补偿型量化 (AWQ-lite) 下重跑 E5/E6 的分配对照。 |
| `tcq_exp/c4_nemotron_sweep.py` | 27.9 | C4: Nemotron 位宽扫描 + CoT 长度曲线。 |
| `tcq_exp/d1_residual_spectrum.py` | 10.2 | D1: 补丁残差谱测量 —— INT4 补丁能否被低秩 LoRA 近似。 |
| `tcq_exp/d1b_weighted_spectrum.py` | 7.7 | D1b: 激活加权残差谱 —— 权重空间 SVD 之后必须排除的"度量错误"。 |
| `tcq_exp/d1c_delta_bits.py` | 2.3 | 临时检查: 补丁行 w4 能否用 "3bit 码 + 1 bit 残差" 编码 (即 packing.json 的 3.375bit 口径). |
| `tcq_exp/d2_switch_cost.py` | 6.0 | D2(改): 零写入换装与合批代价 —— 取代 LoRA 路线的工程实验。 |
| `tcq_exp/d6_nested_scale.py` | 11.9 | D6: 共 scale 嵌套量化 (nested / shared-scale) —— 让补丁只花 1 bit。 |
| `tcq_exp/download_ms.py` | 4.0 | ModelScope 并行分段下载 (HF 镜像不稳时的替代方案)。 |
| `tcq_exp/e17_task_eval.py` | 13.8 | E17 任务级评测: HumanEval pass@1 + GSM8K accuracy。 |
| `tcq_exp/e20_patch_meta.py` | 2.4 | E20: 删除 --prune-layers 遗留的 MTP 元数据并重写量化文件。 |
| `tcq_exp/e21_codebook.py` | 11.2 | E21 step 1: non-uniform 4-level codebook headroom at the fixed 34 B/128 budget. |
| `tcq_exp/e21_encode.py` | 11.8 | E21: fit per-tensor non-uniform 4-level codebooks and re-encode the 2-bit base. |
| `tcq_exp/e21_nll.py` | 8.6 | E21: eval64 NLL with the non-uniform 2-bit codebook (type-45 precursor). |
| `tcq_exp/e_a1_chain.py` | 3.2 | !/usr/bin/env python |
| `tcq_exp/e_a1a_bfcl.py` | 15.1 | !/usr/bin/env python |
| `tcq_exp/e_a1b_longmemeval.py` | 16.3 | !/usr/bin/env python |
| `tcq_exp/e_a1c_traj_drift.py` | 8.4 | E-A1c: 轨迹重放保真 (零环境依赖). |
| `tcq_exp/e_a1d_agent_kld.py` | 14.4 | E-A1d: 带 tool schema 的 agent 上下文里, tcmoe 的 token 级分布误差是否显著变大。 |
| `tcq_exp/e_a1f_decomp.py` | 5.4 | E-A1f: BFCL 多轮失败轨迹机制分解（零模型调用，只读 A1a reviews 产物）。 |
| `tcq_exp/e_a1g_intervene.py` | 11.8 | E-A1g: BFCL-v3 multi_turn_base (20 题) 两个干预实验, 对 tcq_nl_global 基线 acc 0.15. |
| `tcq_exp/e_a1h_chain.py` | 13.2 | !/usr/bin/env python |
| `tcq_exp/e_f10a_tensor_types.py` | 10.4 | E-F10a: UD 三档 vs IQ2_XXS 的 per-tensor 类型/精度归因 (纯元数据, 不加载权重数据)。 |
| `tcq_exp/e_f10b_codebook_im.py` | 15.3 | E-F10b: imatrix 加权的 4 级共享码本（每 (layer, proj) 一张）。 |
| `tcq_exp/e_f10b_judge.py` | 7.0 | E-F10b 判定：imatrix 加权共享码本的 HumanEval 代码侧修复是否成立。 |
| `tcq_exp/e_f2_gsm8k.py` | 7.2 | E-F2: GSM8K 全量 1319 题 x 2 配置 (iq2xxs vs tcmoe_nl_global), 单 server 16 线程 + 分块交替。 |
| `tcq_exp/e_f2_summary.py` | 4.6 | E-F2 汇总: 配对 accuracy + 1000 次序列级 bootstrap + McNemar + 功效。 |
| `tcq_exp/e_f3_nl_sufficiency.py` | 15.9 | E-F3: 专家重要性排序键的充分性判定 —— usage 单独是否足够 (论文 §5.2 边界)。 |
| `tcq_exp/e_f4_prefill.py` | 4.1 | E-F4 汇总: 3 轮 llama-bench 的 pp512 / tg1 中位数 + 极差。 |
| `tcq_exp/e_f6_pareto.py` | 10.7 | E-F6: 双面板 Pareto 汇总 (零机时, 纯读 results)。 |
| `tcq_exp/e_f7_case.py` | 2.0 | E-F7 单例执行器: 把 HumanEval 的 check() 拆成逐条 assert 计通过数。 |
| `tcq_exp/e_f7_tests_passed.py` | 11.4 | E-F7: HumanEval 连续指标 (每题通过 unit test 数) + 配对 t 检验。 |
| `tcq_exp/e_f8_hard_probe.py` | 16.4 | E-F8: 难题基准小探针 (AIME 30 / GPQA-100), 验证"饱和带"的边界条件。 |
| `tcq_exp/e_f9_kld.py` | 35.2 | E-F9: token 级输出分布 KLD 对照 (量化模型 vs 全精度参考)。 |
| `tcq_exp/e_g1_build.py` | 10.0 | E-G1 构建驱动: 路由统计 -> 类型规则 -> imatrix 语料 -> llama-imatrix -> 三份 GGUF -> 体积校验 |
| `tcq_exp/e_g1_eval.py` | 16.8 | E-G1 跨模型评测: gemma-4-26B-A4B (MoE) 三任务 (gsm8k / humaneval / gpqa)。 |
| `tcq_exp/e_g1_eval_matrix.py` | 4.8 | E-G1 评测矩阵: 5 配置 × 3 任务, 按块交错 (同一块内轮换配置, 降低机器漂移影响)。 |
| `tcq_exp/e_g1_imatrix_corpus.py` | 2.8 | E-G1: 把 math / code / general 三域校准集合并成 llama-imatrix 的纯文本语料。 |
| `tcq_exp/e_g1_route_stats.py` | 11.5 | E-G1: gemma-4-26B-A4B 的 MoE 路由使用率统计 (只做 prefill, 不 generate)。 |
| `tcq_exp/e_g1_typefiles.py` | 16.5 | （无文件头说明） |
| `tcq_exp/e_g2_route_stats.py` | 8.8 | E-G2: MiniCPM-V-4.7-35B-A3B 文本主干的路由使用率统计 (只 prefill)。 |
| `tcq_exp/e_g3_community_meta.py` | 4.9 | E-G3: 社区构建元数据补录 (作者 / 发布日期 / 下载量 / 是否声明 imatrix / 滞后天数)。 |
| `tcq_exp/e_g4_eval.py` | 11.9 | G4 Clover (dLLM) A/B 正式评测: GSM8K 200 + MATH-500 100 + HumanEval 164。 |
| `tcq_exp/e_g4_smoke.py` | 6.7 | G4 Clover (dLLM) 冒烟: 等 G3 释放 GPU 后, 用 llama-diffusion-cli 跑 1 道 GSM8K。 |
| `tcq_exp/e_g_build.py` | 14.5 | E-G2/G3 构建驱动 (复用 G1 的角色对齐 llama-quantize 模板)。 |
| `tcq_exp/e_g_cross_model_summary.py` | 7.3 | E-G: 跨模型总表 (G1/G2/G3 各配置 x 三任务 acc + 体积 + 生成耗时)。 |
| `tcq_exp/e_g_dl.py` | 13.0 | E-G2/G3 权重下载 (ModelScope 主源, hf-mirror 兜底; 分块 + 断点续传 + 单实例锁)。 |
| `tcq_exp/e_g_eval_matrix.py` | 9.1 | E-G2/G3/G5/G6/G7 评测矩阵 (与 G1 完全同协议): 配置 × 3 任务, 块交错 50 题。 |
| `tcq_exp/e_g_gguf_header.py` | 6.0 | G2/G3 第0步: 只拉 GGUF 头 (Range 前 N MB), 读 arch / 张量命名, 判断专家布局与分档可行性。 |
| `tcq_exp/e_g_hf_probe.py` | 1.3 | G2/G3 第0步: HF 仓库盘点 (走 hf-mirror, 本机只有镜像可解析)。 |
| `tcq_exp/e_g_hf_sizes.py` | 0.6 | 列指定仓库的 gguf 文件大小 (MB) —— 用于选配置。 |
| `tcq_exp/e_g_queue.py` | 17.7 | E-G2/G3 夜间队列: 等依赖 -> 构建 -> 评测, 单线程顺序, 全程落日志。 |
| `tcq_exp/e_g_typefiles.py` | 9.6 | E-G2/G3: 由 GGUF 张量清单生成 llama-quantize 的 --tensor-type-file 并估算体积。 |
| `tcq_exp/e_g_watch_g2.py` | 2.8 | 等 G2 评测矩阵进程结束后: 双基线汇总 -> 写 .g2_eval_done, 让 chain_b 接着跑 G3。 |
| `tcq_exp/e_h1_build.py` | 4.0 | !/usr/bin/env python |
| `tcq_exp/e_h1_im_extend.py` | 6.2 | !/usr/bin/env python |
| `tcq_exp/e_h1_im_ones.py` | 1.4 | !/usr/bin/env python |
| `tcq_exp/e_h1_nll.py` | 6.4 | !/usr/bin/env python |
| `tcq_exp/e_h1_typefiles.py` | 9.3 | !/usr/bin/env python |
| `tcq_exp/e_h2_tost.py` | 5.9 | E-H2: TOST 等价检验 (零机时, 复用已有配对数据)。 |
| `tcq_exp/e_h3_dprime.py` | 3.4 | !/usr/bin/env python |
| `tcq_exp/e_h3_mcnemar.py` | 2.8 | !/usr/bin/env python |
| `tcq_exp/e_h4_hard_stats.py` | 2.6 | !/usr/bin/env python |
| `tcq_exp/e_h4_workset.py` | 8.4 | !/usr/bin/env python |
| `tcq_exp/e_h5_eval.py` | 19.5 | !/usr/bin/env python |
| `tcq_exp/e_h5_feas.py` | 5.8 | !/usr/bin/env python |
| `tcq_exp/e_h5_mech_check.py` | 8.2 | !/usr/bin/env python |
| `tcq_exp/e_h_queue.py` | 4.5 | !/usr/bin/env python |
| `tcq_exp/exp10_large_eval.py` | 5.0 | E10: 扩评测集复测 (64 条/域) —— 阶段 0 结论是否在更小噪声下保持。 |
| `tcq_exp/exp11_switch.py` | 9.4 | E11: 在线切换端到端 —— 用前 8 token 路由识别域, 换装对应高精度补丁。 |
| `tcq_exp/exp12_pack_check.py` | 6.2 | E12: 真实位宽打包的一致性/体积校验。 |
| `tcq_exp/exp13_mixed_eval64.py` | 7.4 | E13: 混合配额 eval64 复测 —— 判定"单一共享混合补丁"能否成为论文级结论。 |
| `tcq_exp/exp14_prefix_curve.py` | 4.8 | E14: N-token 前缀分类滑动曲线 —— "前缀越短越准"图。 |
| `tcq_exp/exp15_packed_runtime.py` | 9.6 | E15: 3bit 真实容器运行时接入 —— H4 在真实 3.38bit 上跑通。 |
| `tcq_exp/exp16_cross_model.py` | 9.9 | E16: 跨模型普适性对照 —— H1(路由域可分) + H3(前缀路由分类)。 |
| `tcq_exp/exp18_pareto.py` | 7.3 | E18: Pareto 扫描 —— 2bit 底座 + INT4 补丁行的体积–NLL 前沿。 |
| `tcq_exp/exp19_five_domain.py` | 10.6 | E19: 五域结构验证 —— 主模型 H1 矩阵与 gemma-4-26B-A4B 同构性。 |
| `tcq_exp/exp1_sens.py` | 5.1 | 实验1: 全量(40层×256专家×3域)任务条件敏感度矩阵。 |
| `tcq_exp/exp2_delta.py` | 3.9 | 实验2: 抽样专家的直接 ΔL 测量 (任务条件敏感度的金标准验证)。 |
| `tcq_exp/exp3_alloc.py` | 3.6 | 实验3: 等平均位宽下的任务条件化精度配额 vs 全局配额 (H4 核心验证)。 |
| `tcq_exp/exp4_router.py` | 4.2 | 实验4: 路由特征的任务域分类 (H3 验证)。 |
| `tcq_exp/exp5_decomp.py` | 7.2 | 实验5: s 分解 (s ≈ usage × 内在量化误差) 与配额选择的归因。 |
| `tcq_exp/exp6_marginal.py` | 7.4 | 实验6: marginal-value 对照 (usage × Δerr)。 |
| `tcq_exp/exp7_bias_check.py` | 7.4 | 实验7: 3.5(a) 中 7232 个"三域都使用"单元的选择偏差自查。 |
| `tcq_exp/exp8_group_delta.py` | 6.5 | E8: 成组 ΔL 金标准 —— 验证 s / usage 排序在组级扰动下是否有效。 |
| `tcq_exp/exp9_mixed_quota.py` | 5.4 | E9: 非等比例配额与混合校准 (H4 延伸)。 |
| `tcq_exp/int2_ext.py` | 1.5 | int2_ext 构建/加载 helper: 2bit 密排底座 + INT4 补丁 混合 FFN 扩展 (A1)。 |
| `tcq_exp/make_data.py` | 3.3 | 构建三域校准/评测数据: math(GSM8K), code(HumanEval), general(Alpaca)。 |
| `tcq_exp/make_data_stage2.py` | 5.7 | 阶段 2 数据构建: |
| `tcq_exp/pin.py` | 5.0 | CPU 亲和性绑定 (Windows 双路 Xeon, 2 NUMA node)。 |
| `tcq_exp/smoke_probe.py` | 4.6 | 冒烟测试: 验证 Qwen3.6-35B-A3B INT4 引擎在本机(16GB GPU + CPU offload)可跑前向, |
| `tcq_exp/smoke_speed.py` | 2.7 | 速度对比: GPU缓存prefill(默认) vs CPU批量直接INT4 prefill, 以及decode。 |
| `tcq_exp/tcq_lib.py` | 11.3 | 任务条件化量化实验公共库: |
| `tcmoe/pack_tcmoe.py` | 17.3 | !/usr/bin/env python3 |
| `tcmoe/verify_bitwise.py` | 4.3 | !/usr/bin/env python3 |
| `reverse/e_g1_eval.py` | 17.0 | E-G1 跨模型评测: gemma-4-26B-A4B (MoE) 三任务 (gsm8k / humaneval / gpqa)。 |
| `reverse/pair_eval.py` | 4.5 | E-R1 配对统计: McNemar 精确 + 1000 次序列级 bootstrap (seed 20261104)。 |
| `reverse/reverse_types.py` | 6.8 | !/usr/bin/env python |
| `reverse/run_tasks.py` | 5.0 | C1: 用一个臂跑 GSM8K200 + HumanEval164 (协议 = e17 / e_g1_eval, 与 E-H1c 完全相同)。 |
| `route_consistency/r2_analyze.py` | 14.7 | E-R2 分析：三配置路由一致性 + 翻转 token 的 NLL 代价 + 预注册判定。 |
| `route_consistency/r2_run.py` | 4.2 | E-R2 采集：对三个配置导出 eval64 三域的逐 token 路由概率 + NLL。 |
| `nemotron_ext/n1_analyze.py` | 12.0 | E-N1: Nemotron 四档 + Q6 锚 vs Qwen3.6 六档的 acc/CoT-位宽曲线与断崖判定。 |
| `nemotron_ext/route/nr_analyze.py` | 6.1 | E-R2 Nemotron 臂分析：Q3_K_M vs Q6_K 的路由一致性。 |
| `nemotron_ext/route/nr_run.py` | 2.5 | E-R2 Nemotron 臂采集：i1-Q3_K_M vs i1-Q6_K 的逐 token 路由。 |
| `gptq_baseline/eval_gptq.py` | 4.2 | E-GPTQ 评测：GPTQ-3bit（HF 格式）在 GSM8K 200 + HumanEval 164 上。 |
| `gptq_baseline/quantize_gptq.py` | 2.5 | E-GPTQ 量化脚本（在具备 gptqmodel 的环境运行）。 |
| `llamacpp/llama.cpp-tcmoe.patch` | 163.4 | llama.cpp；基线 5047748；tcmoe 共享码本量化类型 + server 端 domain 路由 + tcmoe-verify 校验工具 + LLAMA_MOE_DUMP 路由导出（E-R2） |
| `llamacpp/new-files/llama.cpp/src/llama-tcmoe.cpp` | 15.0 | 新增文件完整副本（patch 已含，此处便于直接查阅） |
| `llamacpp/new-files/llama.cpp/src/llama-tcmoe.h` | 3.6 | 新增文件完整副本（patch 已含，此处便于直接查阅） |
| `llamacpp/new-files/llama.cpp/tools/tcmoe-verify/CMakeLists.txt` | 0.5 | 新增文件完整副本（patch 已含，此处便于直接查阅） |
| `llamacpp/new-files/llama.cpp/tools/tcmoe-verify/main.cpp` | 34.7 | 新增文件完整副本（patch 已含，此处便于直接查阅） |
| `llamacpp/new-files/llama.cpp/examples/eval-callback/route-dump.cpp` | 8.6 | 新增文件完整副本（patch 已含，此处便于直接查阅） |
| `llamacpp/llama-upstream.patch` | 16.5 | llama-upstream；基线 18ab62a；Clover dLLM（clover_moe）架构放行（E-G4 冒烟） + LLAMA_MOE_DUMP 路由导出（E-R2） |
| `llamacpp/new-files/llama-upstream/examples/eval-callback/route-dump.cpp` | 8.6 | 新增文件完整副本（patch 已含，此处便于直接查阅） |
