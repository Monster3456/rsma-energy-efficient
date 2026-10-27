# 引理 L：标量对偶乘子在逐点约束下的失效

> M1 产出（`THEORY_RESEARCH_PLAN.md` §2.1）· 2026-10-27
> 用途：(i) 命题 T 的引理/特例；(ii) 解释 36 条失败方向中对偶族的机理；
> (iii) 为"必须动可行域几何"提供必要性论证。
> 状态：**证明草案**，假设与结论待同行检验；转录进学位论文前须复核符号一致性。

---

## 1. 设定

设策略参数空间为 $\Theta\subseteq\mathbb{R}^d$，目标 $J:\Theta\to\mathbb{R}$。
问题有 $m$ 个**逐点约束**（$m$ 个工作点各一条）：

$$g_i(\theta)\ \ge\ 0,\qquad i=1,\dots,m$$

本问题中 $g_i(\theta)=\zeta_i(\theta)-\zeta_i^{\rm SDMA}$ 与
$\bar Q_i^{\rm SDMA}\cdot 1.1-\bar Q_i(\theta)$ 的组合，$i$ 索引评估 SNR 点。
**关键**：这些约束**不可按 $i$ 分解**——单个策略 $\pi_\theta$ 须同时满足全部 $m$ 条。

**真问题**（逐约束对偶）：

$$\max_{\theta\in\Theta}\ J(\theta)\quad\text{s.t.}\quad g_i(\theta)\ge0\ \ \forall i
\tag{P}$$

其 Lagrangian 为 $L(\theta,\lambda)=J(\theta)+\sum_{i=1}^m \lambda_i g_i(\theta)$，
$\lambda=(\lambda_1,\dots,\lambda_m)\in\mathbb{R}^m_{\ge0}$。

**标量对偶方法**只用**单一对偶变量** $\lambda\in\mathbb{R}_{\ge0}$：

$$\max_{\theta}\ \min_{\lambda\ge0}\ \Big[J(\theta)+\lambda\sum_{i=1}^m g_i(\theta)\Big]
\tag{S}$$

即把 $m$ 条约束**聚合成一条** $\sum_i g_i(\theta)\ge0$ 后施加对偶压力。
（Lagrangian 单乘子、PID 单通道、$\lambda$ 上限阶梯、标量罚项，均属此类。）

---

## 2. 引理陈述

> **引理 L（标量对偶失效）**. 设 $\theta^\star\in\Theta$ 是 (P) 的 KKT 点，对应逐约束
> 最优乘子 $\lambda^\star=(\lambda_1^\star,\dots,\lambda_m^\star)$，约束梯度
> $\{\nabla g_i(\theta^\star)\}_{i=1}^m$ 线性无关。若 $\lambda_1^\star,\dots,\lambda_m^\star$
> **不全相等**，则 $\theta^\star$ **不是** (S) 的驻点，即不存在 $\lambda\ge0$ 使
> $\nabla_\theta\big[J+\lambda\sum_i g_i\big](\theta^\star)=0$。
>
> 换言之：标量对偶法**无法返回**逐约束问题的解 $\theta^\star$；其驻点 $\hat\theta$ 必然
> (i) 违反至少一条约束，或 (ii) 在可行域内但目标值 $J(\hat\theta)<J(\theta^\star)$。

---

## 3. 证明

$\theta^\star$ 是 (P) 的 KKT 点，故满足：

**(a) 平稳性**：$\nabla J(\theta^\star)+\sum_{i=1}^m \lambda_i^\star \nabla g_i(\theta^\star)=0$

**(b) 互补松弛**：$\lambda_i^\star\, g_i(\theta^\star)=0\ \ \forall i$

**(c) 原始/对偶可行**：$g_i(\theta^\star)\ge0,\ \lambda_i^\star\ge0$

（假设 (a) 成立即 $\theta^\star$ 是驻点。）

反设 $\theta^\star$ 是 (S) 的驻点，即存在 $\lambda\ge0$ 使

$$\nabla J(\theta^\star)+\lambda\sum_{i=1}^m \nabla g_i(\theta^\star)=0 \tag{S-a}$$

用 (S-a) 减 (a)：

$$\sum_{i=1}^m \big(\lambda-\lambda_i^\star\big)\,\nabla g_i(\theta^\star)=0$$

由**约束梯度线性无关**的假设，各系数必为零：

$$\lambda-\lambda_i^\star=0\quad\forall i\ \Longrightarrow\ \lambda_1^\star=\lambda_2^\star=\cdots=\lambda_m^\star=\lambda$$

与"$\lambda_i^\star$ 不全相等"矛盾。故 $\theta^\star$ 不是 (S) 的驻点。$\blacksquare$

**第二部分**：设 $\hat\theta$ 是 (S) 的驻点。若 $\hat\theta$ 可行（$g_i(\hat\theta)\ge0\ \forall i$），
由 (a) 的唯一性（在给定乘子下）与第一部分，$\hat\theta\ne\theta^\star$。若 (P) 的最优解唯一，
则 $J(\hat\theta)<J(\theta^\star)$；若 $\hat\theta$ 不可行，则已违反至少一条约束。$\blacksquare$

---

## 4. 假设的可检验性与强度

| 假设 | 强度 | 可检验性 |
|---|---|---|
| $\theta^\star$ 是 KKT 点 | 常规 | 理论假设 |
| **约束梯度线性无关**（LICQ） | 常规约束规范 | 数值可验（$[\nabla g_1,\dots,\nabla g_m]$ 满秩） |
| $\lambda_i^\star$ 不全相等 | **本问题的经验事实** | 见 §5 |

**若 LICQ 不成立**：结论需弱化为"存在一组系数 $\{c_i\}$ 使 $\sum_i c_i\nabla g_i=0$，
且 $\lambda_i^\star$ 在该零空间方向上不恒定"。此时标量对偶仍可能失败，但需重述。

**若 $\lambda_i^\star$ 全相等**：标量对偶**可行**——这正是"逐点压力相同"的退化情形。
故引理的适用边界是：**各工作点需要的约束压力不同**。本问题的 SNR 异质性（低 SNR
噪声受限、高 SNR 深过载）正是这一条件的物理来源。

---

## 5. 与实证的一致性（$T_2$ 门）

| 实证 | 与引理的一致性 |
|---|---|
| $\lambda$ 解饱和（`f30/f31`）：$\lambda$ 被 50 截断"等价系统性偏低、与 2 个未达标 SNR 点持续违约精确同构" | 单 $\lambda$ 无法对低 SNR 点施加足够压力，同时不过压其它点——正是引理的"欠压/过压"二分 |
| CVPO 精确对偶（`f09`）：每 batch SLSQP 解 $(\eta,\lambda_1..\lambda_4)$ 仍掉点 | 注意：该方向已尝试**多乘子**，失败说明仅有乘子结构不够，还需可行域几何（与命题 T 的推论一致） |
| PID anti-windup（`f00`）：设计前提即"消除不可行桶 windup 烧功率" | 提案者已认定单乘子在不可行（低 SNR）桶上失效 |
| 低 SNR 缺口在全部 28 条判定下持续存在 | 低 SNR 点的 $\lambda^\star$ 系统性高于其它点——$\lambda_i^\star$ 不共线的直接证据 |

**一致性检验的判据**：引理的适用条件（$\lambda_i^\star$ 不共线）在本问题成立；
引理预言的失败模式（欠压导致违约 / 过压导致保守）在对偶族失败记录中均可见。

---

## 6. 与命题 T 的关系

- **引理 L 是 T 的特例**：L 证"标量对偶"这一具体机制失效；T 证"任何只改奖励/目标、
  不改可行域几何与初始化的方法"有天花板。
- **L 是 T 的证明线索**：L 的"标量无法分辨逐点压力"可推广为 T 的路线 B
  （标量不可辨——$J'$ 是标量，无法区分"哪一点在违约"）。
- **推论（方法导向）**：由 L，约束压力必须**逐点可分辨**地施加。但 H3 的正命题已反驳
  "逐点参数化即可"（逐桶参数 3 条同样失败）。故唯一出路是**可行域几何**——
  一个参数化同时落在 $\bigcap_i\mathcal{F}_i$ 内。

---

## 7. 待办

- [ ] 复核 KKT 条件与符号的一致性（尤其 $g_i$ 的方向：$\ge 0$ 还是 $\le 0$）
- [ ] 检验 LICQ 在本问题的数值成立性（用 `diff_engine` 计算 $\nabla g_i$，验满秩）
- [ ] 补充"$\lambda_i^\star$ 不共线"的数值证据（若可从 36 条失败记录中提取）
- [ ] 转录进学位论文时用 `\begin{lemma}` 环境 + `\begin{proof}`（hithesis 已定义）
- [ ] 考虑是否补一条"$\theta^\star$ 唯一"的假设以使第二部分严格
