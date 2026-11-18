# §5 任务条件化量化（TCQ）方法：写作框架

> 本文只给可改写进正文的定义、算法与代码锚点；数值均来自代码或 `tcq_exp/results/`。方括号编号对应文末脚注。

## 5.1 符号表

| 符号 | 含义 |
| --- | --- |
| $l\in[0,40)$, $e\in[0,256)$ | 专家层编号；每层路由专家编号 |
| $d\in\{\text{global},\text{math},\text{code}\}$ | 补丁域；`global` 为三域得分之和 |
| $\pi_t\in\Delta^{255}$ | 第 $t$ 个 token 的路由 softmax 概率 |
| $u_{l,e,d}$, $s_{l,e,d}$ | 域 $d$ 下专家 $e$ 的路由权重和、敏感度 |
| $W_{l,p,e}\in\mathbb{R}^{r\times c}$ | 投影 $p\in\{\text{gate},\text{up},\text{down}\}$ 的 BF16 权重 |
| $d_{l,e,b}$, $c_{l,p}\in\mathbb{R}^4$ | 每 128 权重的组尺度；每投影 4 项码本 |

## 5.2 数学定义

### 5.2.1 用量统计与 top-$k$ 配额

对校准集 $\mathcal{C}_d$ 中的 token，路由 top-$r$（代码 $r=8$）的计数与权重和分别为[^u]

$$
n_{l,e}=\sum_{d}\sum_{t\in\mathcal{C}_d}\sum_{j=1}^{r}\mathbf{1}\{\operatorname{top}_j(\pi_t)=e\},\qquad
u_{l,e,d}=\sum_{t\in\mathcal{C}_d}\pi_{t,e}\,\mathbf{1}\{e\in\operatorname{TopK}_r(\pi_t)\}.
$$

代码保存的计数是**三域合计**（`assign_cnt`），权重和才保留域维；$\pi$ 为全专家 softmax，top-$r$ 后不再重归一。每层配额为[^k]

$$
k=\lceil \rho E\rceil=\lceil 0.25\times 256\rceil=64 .
$$

实际补丁清单由域条件敏感度 $s_{l,e,d}=\sum_t \pi_{t,e}\,\mathrm{relErr}_{t,e}$ 排序，而非 $u$[^s]；E-F3 以三域合计的 $u$/$s$ 排序作对照，判定 `usage_close`（NLL 差 CI 覆盖 0），故正文可写 $u$，但须与代码口径对齐。`global` 取三域求和[^global]。

### 5.2.2 非均匀 2-bit 码本

底座保持 34 B/128 权重的容器（GGML type 45），组尺度与码字含义为[^cb]

$$
d_{l,e,b}=\frac{\max_{i\in b}|W_{l,e,i}|}{1.5},\qquad
Q(z)=c_{q(z)},\qquad q(z)=\sum_{j=1}^{3}\mathbf{1}\{z>b_j\},\qquad z_i=W_i/d_i .
$$

固定分割点后，码本按加权 Lloyd-Max 迭代[^lloyd]：

$$
b_j=\frac{c_j+c_{j+1}}{2},\qquad
c_j=\frac{\sum_i a_i z_i\,\mathbf{1}\{q_i=j\}}{\sum_i a_i\,\mathbf{1}\{q_i=j\}} .
$$

初值为分位点 $(j+0.5)/4$，最多 40 轮或 $\max_j|\Delta c_j|<10^{-6}$ 停止；码本按 (层, 投影) 共享，体积仍为 2.125 bpw。

### 5.2.3 imatrix 加权目标

令 $a_i$ 为 imatrix 的逐输入通道能量（`.in_sum2`，按专家均值归一），拟合目标为[^obj]

$$
\min_{\{c\}}\ \sum_i a_i\bigl(W_i-d_i c_{q_i}\bigr)^2 ,
$$

其矩阵块写法即 $\min\sum_g w_g\|W_g-Q(W_g)\|_F^2$，其中 $w_g$ 取自 imatrix 条目。代码在 $z=W/d$ 上加权时额外乘 $d_i^2$，使误差回到权重单位；落地编码沿用容器中的 fp16 $d$ 不重拟合，V1 对照的组尺度重拟合仅用于上界分析。

## 5.3 伪代码

```latex
\begin{algorithm}[t]\small
\caption{TCQ 单文件构建}
\KwIn{BF16 专家权重 $W$；三域校准集 $C_{\text{math}},C_{\text{code}},C_{\text{general}}$；$\rho=0.25$}
\KwOut{GGUF：2-bit 底座 + 每域 Q4\_K 补丁表}
1: 逐域前向 $C_d$，累计 $u_{l,e,d},s_{l,e,d}$（top-8；$k=\lceil\rho E\rceil=64$）\;
2: $I_{l,d}\leftarrow\arg\operatorname{top}_k(s_{l,\cdot,d})$；$I_{l,\text{global}}\leftarrow\arg\operatorname{top}_k(\sum_d s_{l,\cdot,d})$\;
3: 对每个 $(l,p)$：$z\leftarrow W/d$，以 $a_i$ 为权重拟合 4 项码本 $c_{l,p}$\;
4: 底座按 $q=\arg\min_j|z-c_{l,p,j}|$ 写 type 45；补丁对 $\bigcup_d I_{l,d}$ 的专家写 Q4\_K\;
5: 写 \texttt{idx/gate/up/down} 与 \texttt{tcmoe.cb}，校验张量数、元素数与字节数\;
\end{algorithm}
```

## 5.4 运行时切换流程

```text
请求 {"domain":"math"}
 1. resolve_tcmoe_domain：域名字→domain_id；"base"/"-1"→-1        [server-context.cpp:1788-1815]
 2. set_tcmoe_domain：写 cparams.tcmoe_domain，图按新域重建   [llama-context.cpp:1409-1420]
 3. 若 slot.prompt_domain 不同：清 slot.prompt 前缀缓存        [server-context.cpp:2024-2028]
 4. qwen35moe：取 tcmoe.{gate,up,down}_exps[d] 与两张 id 表     [qwen35moe.cpp:492-505]
 5. graph：ids_base=tab_base[sel]，ids_patch=tab_patch[sel]     [llama-graph.cpp:2315-2329]
 6. mul_mat_id_mixed(base, patch, ids_base, ids_patch)          [llama-graph.cpp:2332-2338]
 7. 首次触碰补丁行→mmap 缺页，按页调入；无权重拷贝             [llama-model.cpp:2355-2389]
```

正文若写“patch_map 重写”，应说明重写的是**每层每域的索引表指针**，不是权重内容。补丁张量在加载时以独立 mmap 绑定到 CPU buffer，页由缺页按需调入；因此“零权重写入”成立，“零 I/O/零延迟”不成立：域序列请求墙钟为 74.2/117.0/104.4 s，热缓存后 73.3/75.5 s（均含 3,609-token prompt）[^page]。

## 5.5 体积约束求解

H1 把类型分配写成离散可行性问题，而非质量优化[^h1]：

$$
\min_{a\in\mathcal{A}}\ \Bigl|\frac{1}{2^{30}}\sum_t \frac{n_t\,b(a_t)}{8}-B\Bigr|,\qquad
\text{s.t. } a_t\in\mathcal{Q},\ \ b(a_t)\ \text{按块大小与降级规则计算},\qquad B=11.45\ \text{GiB}.
$$

$B$ 取官方 UD-Q2_K_XL 的文件体积 12,290,628,576 B（11.446 GiB）[^budget]。搜索策略是**小网格枚举 + 最近预算选择**，不是贪心或联合优化[^search]：G0 对哈希混合比例（IQ2\_S/IQ3\_XXS）做 40 次二分；G1 固定“专家 IQ2\_S / 非专家 Q5\_K”；G2 枚举 4 种专家底座 × 顶层（blk.0–9）升档类型；G3 枚举 3 种底座 × `ffn_down` 升 1/2 档 × 顶层升 1/2 档；G4 逐张量复刻 UD 清单；`best_fit` 取 $|{\rm est}-B|$ 最小者。选择用含 blk.40/nextn 的全模型字节，`core_gib` 仅报告、不参与约束。质量项不在求解器内，由后续 NLL/任务评测承担。H1a 的验收带为 $\pm0.30$ GiB，C1.2 随机名单臂复用 $\pm0.50$ GiB；10 个构建实际为 11.469–11.710 GiB[^tol]。

## 5.6 可复现性清单

- 类型文件与构建：`python tcq_exp/e_h1_typefiles.py --write`；`python tcq_exp/e_h1_build.py --arms g0,g1,g2,g3,g4 --im on,off`；评测 `python tcq_exp/e_h1_nll.py run && python tcq_exp/e_h1_nll.py report`。
- 码本：`python tcq_exp/e21_encode.py fit`（专家 `0,85,170,255`，`--row-stride 4`，`--iters 40`）；`encode`；`verify --layer 0`（均匀码本复算 q 字节）。
- 打包：`python tcmoe/pack_tcmoe.py --skeleton <UD-Q4_K_M.gguf> --base tcmoe_q2nl --p2-dir tcq_exp/stores/packed2_v1 --patch-type q4_k --domains global,math,code --frac 0.25`。
- 随机种子：20261108（G0 张量名哈希、H1 NLL bootstrap）；20261104（C1.2 随机层、C1.3、E-F3、G5–G7 的 bootstrap）[^seeds]。
- 校验项：源模型 753 张量 / 35,505,251,456 元素；10/10 构建张量数与元素数一致；TCQ 文件 1,213 张量 / 24,711,991,680 B；读取层校验 `idx` 严格递增、`ne2=len(idx)`、补丁为 Q4\_0/Q4\_K 且 `ne0` 对齐[^repro]。

## 5.7 与实验对应的锚点

| 定义/机制 | 验证实验 | 结果文件 |
| --- | --- | --- |
| $u,s$ 与 $k=64$ | H1a 类型组合；H1c 任务级；C1.2 随机名单；E-F3 排序键充分性 | `e_h1_types_meta.json`、`e_h1_summary.json`、`c1_grand_meta.json`、`e_f3_nl_sufficiency.json` |
| 11.45 GiB 预算与枚举 | H1a 十配置构建 | `e_h1_build_state.json`、`e_h1_types_meta.json` |
| 2-bit 非均匀码本 | E21 V1 权重 wMSE；E21 NLL；C2b 同口径 KLD | `e21_codebook.json`、`e21_nll_v1.json`、`c1_kld_tcq_same.json` |
| imatrix 加权 | C1.3 on/off 全数值表 | `c1_h1_complete.json` |
| 域切换与 page-in | E-H4b 冷/热切换工作集 | `e_h4_workset.json` |
| 跨模型可迁移性 | G5–G7 配对检验 | `e_g567_pairwise.json` |

## 5.8 待作者确认的空位

1. **记号**：正文写 $u$ 还是 $s$？代码实际按 $s$ 排序，E-F3 仅说明 $u$ 排序 NLL 等价；二者不可混写。
2. **体积带**：H1a 用 $\pm0.30$，C1.2 用 $\pm0.50$，实际 11.469–11.710；需统一正文报告口径。
3. **类型表述**：代码中专家底座候选为 IQ2\_XXS/IQ2\_XS/IQ2\_S/Q2\_K，非专家在 G1–G3 固定 Q5\_K、G0 哈希混 IQ2\_S/IQ3\_XXS；没有“非专家 IQ2\_XXS/IQ2\_S”组合，需核对任务描述或补一组 typefile。
4. **收敛性**：Lloyd-Max 只给 40 轮/1e-6 停止条件，未给收敛证明；是否在正文声明“固定迭代 + 经验稳定”由作者定。
5. **切换措辞**：“零写入”可写，“零 I/O/零等待”不可写；H4 已给出域切换请求墙钟（含 prompt）。

---

[^u]: `tcq_exp/tcq_lib.py:120`（`TOPK=8`）、`132-143`（softmax 后 top-8，`topv` 未重归一）；`tcq_exp/exp1_sens.py:52,93,99`（`assign_cnt` 三域合计、`usage` 分域累加）。
[^k]: `tcmoe/pack_tcmoe.py:64-68`（`k=max(1,int(E*frac))`）、`:241`（`--frac` 默认 0.25）；`tcq_exp/bitpack.py:14-15`（`N_EXPERTS=256`）。
[^s]: `tcq_exp/exp1_sens.py:2-5,98,107`（敏感度定义）；`tcmoe/pack_tcmoe.py:258-265`（实际读取 `sm["s"]`）；`tcq_exp/e_f3_nl_sufficiency.py:14-20` 与 `results/e_f3_nl_sufficiency.json`（`usage_close`）。
[^global]: `tcmoe/pack_tcmoe.py:56-61`（`global` 为 `s.sum(-1)`）、`:240`（默认 `domains=global,math,code`）。
[^cb]: `llama.cpp/ggml/include/ggml.h:436-437`（type 44/45）；`ggml/src/ggml-common.h:194-204`（块 128、34 B）；`ggml/src/ggml-quants.c:542-563`（解码 `cb[q]*d`）；`tcq_exp/bitpack.py:45-52`（`d=amax/1.5`）。
[^lloyd]: `tcq_exp/e21_codebook.py:108-133`（初始化、分割、质心、停止条件）；`tcq_exp/e21_encode.py:63-91`（`boundaries`/`fit_cb`）。
[^obj]: `tcq_exp/e21_codebook.py:156-159`（`a=imp*d²`）；`tcq_exp/e21_encode.py:46-60`（`.in_sum2` 读取、按专家均值归一）、`:118-125`（拟合权重）。
[^h1]: `tcq_exp/e_h1_typefiles.py:29-31,34-45,150-159`；`tcq_exp/e_h1_build.py:31,72-75`（`llama-quantize` 与 imatrix）。
[^budget]: `tcq_exp/results/e_h4_workset.json:31-34`（`ud_q2kxl.file_bytes=12,290,628,576`）；`tcq_exp/e_h1_typefiles.py:29`（`TARGET=11.45`）。
[^search]: `tcq_exp/e_h1_typefiles.py:112-135`（G0 二分 40 次）、`:138-144`（G1）、`:162-178`（G2）、`:181-204`（G3）、`:207-224`（G4）、`:150-159`（`best_fit`）。
[^tol]: `tcq_exp/e_h1_typefiles.py:30`（`TOL=0.30`）；`tcq_exp/c1_rand_types.py:43`（`TOL=0.50`）；`tcq_exp/results/e_h1_build_state.json`（11.469–11.710）。
[^seeds]: `tcq_exp/e_h1_typefiles.py:31` 与 `e_h1_nll.py:35`（20261108）；`c1_rand_types.py:33`、`e_f3_nl_sufficiency.py:235`（20261104）。
[^repro]: `tcq_exp/results/c1_grand_meta.json:7983-7984`（753 / 35,505,251,456）；`e_h1_build_state.json`；`e_h4_workset.json:63-64`（24,711,991,680 B / 1,213 张量）；`llama.cpp/src/llama-tcmoe.cpp:311-382`（读取层校验）。
[^page]: `llama.cpp/src/llama-model.cpp:2355-2389`（补丁独立 mmap）、`:2448-2464`（id 表）；`tcq_exp/results/e_h4_workset.json:74-115`（冷 74.2/117.0/104.4 s，热 73.3/75.5 s）。
