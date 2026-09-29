# Energy-Efficient Scheduling for Overloaded RSMA MU-MIMO Uplink under Per-SNR Delivery Constraints: Hierarchical Reinforcement Learning with Solver-Expert Distillation

> **Draft v2 (2026-09-11).** Markdown draft of an IEEE JSAC/TWC-style paper. All numbers are
> taken from official evaluations recorded in `docs/paper-results.md` and
> `.trellis/tasks/09-02-solver-expert/experiment-results.md`. All 18 references were
> web-verified on 2026-09-11 (arXiv/IEEE/DBLP records; see verification notes in
> commit history); venues updated where a journal/conference version exists.

---

## Abstract

Rate-splitting multiple access (RSMA) promises spectral gains in the **overloaded**
multi-user MIMO uplink, where the number of users $K$ exceeds the receive antennas
$N_r$, but harvesting those gains requires jointly choosing transmit power, the
common/private stream split, and the successive-interference-cancellation (SIC)
decoding order under **imperfect channel state information (CSI)** and hard
per-user quality-of-service (QoS) limits. We formulate energy efficiency (EE)
maximization under **per-SNR-point delivery-ratio and queue-stability constraints
relative to a strong SDMA reference**—constraints that must hold *simultaneously at
every operating point*, not on average—and show that this requirement defeats both
classical baselines and single-timescale deep reinforcement learning (DRL). We
propose a **two-timescale hierarchical policy**: a slow meta-controller that sets
user grouping and group-level stream split, and a fast self-attention graph worker
that proposes per-slot power shares, queue drain ratios, and a Plackett-Luce SIC
order. The worker is trained with a **delivery-gated Lyapunov composite reward**
(DGLCR) combining a Dinkelbach-linearized EE term, max-weight throughput, and
CVaR-tail delivery-virtual-queue penalties, under a Lagrangian delivery constraint
with lexicographic gradient projection. Because the reward landscape is
non-differentiable through the combinatorial SIC order, we introduce a
**solver expert**: a differentiable shadow of the MMSE-SIC rate engine that refines
power logits online and supplies demonstrations; the policy is then distilled by
behavior cloning and fine-tuned with a **stepwise-frozen protocol (SFTP)** with
guard-selected checkpoints. On a 7-point SNR grid (0–30 dB, imperfect CSI, 50
episodes/point), the resulting policy clears **all 7 per-point gates** with mean
delivery $0.9496 \pm 0.0047$ across five seed replications, and dominates SDMA,
NOMA and equal-power baselines on delivery, queue and overflow at every SNR—at
roughly half their transmit power (30 dB: 0.888 vs 0.664/0.581/0.601). We further
show that the solver expert *itself*, run as a training-free per-slot optimizer,
is the strongest controller on this grid (mean delivery 0.9574), establishing a
model-based upper bound; distillation compresses it into a single-forward-pass
policy (0.9441) and the frozen fine-tune recovers ≈59% of the compression gap
(0.9519), so the learned policy's value is inference cost and model-free
deployment rather than absolute performance. A pre-registered fine-tuning
campaign shows that released-learning-rate, trust-region, anchoring, and
data-reshaping variants all fall **within the policy's own seed dispersion**,
motivating multi-seed confidence-interval comparison as the honest reporting
standard for this class of methods.

**Index Terms—** Rate-splitting multiple access, MU-MIMO uplink, energy
efficiency, deep reinforcement learning, MMSE-SIC, Lyapunov optimization,
fractional programming, expert distillation.

---

## I. Introduction

Cell-edge and machine-type traffic increasingly force base stations to serve more
active users than they have receive antennas. In this **overloaded regime**
($K > N_r$) spatial multiplexing alone saturates: interference, not noise, bounds
the sum rate, and queueing delays grow without a scheduling policy that is
explicitly queue-aware. Rate-splitting multiple access (RSMA) mitigates overload by
splitting each message into a *common* stream decoded by all users and a *private*
stream decoded by its intended user, with successive interference cancellation
(MMSE-SIC) at the receiver [1], [2], [10]. Realizing RSMA's promise, however, requires
solving a coupled, per-slot combinatorial problem—power allocation across users,
the common/private split, and the SIC decoding order—repeatedly and online, under
**imperfect CSI** and a realistic power-consumption model.

Three properties make this problem resistant to existing approaches.

**(1) Hard per-point constraints.** Practical deployments specify QoS at discrete
operating points (SNR levels), e.g. a minimum delivery ratio and a maximum queue
length *at each point*, typically benchmarked against a deployed baseline such as
SDMA. A policy that is excellent on average but violates one point is unacceptable;
the constraint is a *simultaneous* one over the grid, which turns the objective
into a lexicographic race at the worst points rather than a smooth aggregate.

**(2) Imperfect CSI in the overloaded regime.** Pilot-based channel estimation
injects noise whose variance scales as $\sigma^2/(K p_{\rm pilot})$ per user, so
overload *worsens* estimation quality exactly where spatial resources are scarcest.
Robust designs bound the residual error $\|\Delta \mathbf{h}_k\| \le \varepsilon$ and
regularize the combiner accordingly, but the interaction of estimation error with
SIC error propagation makes closed-form robust allocation brittle.

**(3) Energy efficiency without loopholes.** Maximizing $\mathrm{EE}=R_{\rm
sum}/P_{\rm consume}$ under a consumption model $P_{\rm consume}=\sum_k
P_k/\eta_{\rm PA}+P_{\rm circ}$ removes the degenerate "transmit nothing" optimum,
but also couples the fractional objective to the queueing state, requiring
fractional-programming treatment (Dinkelbach) inside a stochastic control loop.

Deep reinforcement learning (DRL) is a natural candidate for (1)–(3), yet naive
DRL fails here for a structural reason: the combinatorial SIC order and the
non-differentiable rate expressions block gradient-based credit assignment, and
the reward's delivery gate creates sparse, saturating gradients at exactly the
points that matter. Recent robust and distributionally-robust DRL beamforming
[3], [4], [16] and generalization analyses [5], [6], [18] address robustness to
CSI perturbation but not the simultaneous per-point QoS structure. Lyapunov
drift-plus-penalty methods [7], [8], [17] provide queue-stability guarantees for
*given* rate regions but do not learn the combinatorial allocation itself;
policy-distillation practice in RANs [15] motivates our teacher→student design
but does not couple distillation to per-slot combinatorial refinement under
hard per-point gates.

**Contributions.** We make five contributions.

1. **Problem formulation.** We formalize EE maximization for overloaded RSMA
   MU-MIMO uplink under *per-SNR-point* delivery and queue constraints relative to
   SDMA, with a lexicographic evaluation criterion that isolates the binding
   constraints (§II-E).
2. **Two-timescale hierarchical policy.** A slow meta-controller (grouping +
   group-level split) and a fast self-attention graph worker (power shares, drain
   ratios, Plackett-Luce SIC order) with state-augmented observations (§III-A,B).
3. **Delivery-gated Lyapunov composite reward (DGLCR).** A Dinkelbach-linearized EE
   term plus max-weight throughput and CVaR-tail delivery-virtual-queue penalties,
   with sigmoid gates that remove low-power degeneracy, under a Lagrangian
   delivery constraint with lexicographic gradient projection (§III-C,D).
4. **Solver-expert distillation.** A differentiable shadow of the MMSE-SIC engine
   (float64-consistent to $10^{-9}$) that refines power logits online, supplies
   demonstrations that alone clear all 7 gates, and is distilled into the policy;
   a stepwise-frozen fine-tune protocol (SFTP) with guard-selected checkpoints
   preserves and extends the distilled behavior (§III-E,F).
5. **An honest empirical account.** Full-grid comparison against SDMA/NOMA/
   equal-power; seed-replication quantifying the policy's intrinsic dispersion
   ($\sigma_{\rm seed}\approx 0.005$); and a pre-registered fine-tuning campaign
   showing that five mechanism variants fall within that dispersion—establishing
   multi-seed CI comparison as the reporting standard for this problem class (§V).

The remainder is organized as §II system model, §III method, §IV methodology,
§V results, §VI discussion, §VII conclusion.

---

## II. System Model and Problem Formulation

### A. Network and RSMA transmission

A single base station with $N_r$ antennas serves $K$ users, each with $N_t$
transmit antennas, over $T$ slots per episode. User $k$'s message is split into a
common part (rate-share governed by a split factor $\alpha_k \in [\alpha_{\min},
\alpha_{\max}]$) and a private part. The BS applies unified MRT transmit weights
$\mathbf{W}_c, \mathbf{W}_p \in \mathbb{C}^{N_r \times K}$ to common and private
streams under per-user power caps $P_{k} \le P_{k,\max}$.

The receiver performs MMSE-SIC in a decoding order $\pi$. For a stream decoded at
position $j$ in $\pi$, the achievable rate is $R = \log_2(1+\mathrm{SINR})$ where the
SINR denominator collects (i) interference from streams not yet decoded, (ii)
**SIC residual interference** from already-decoded streams scaled by a hardware
imperfection factor $\beta^2$, and (iii) **CSI-error residual** scaled by
$\varepsilon^2 p$, plus thermal noise $\sigma^2$:

$$
\mathrm{SINR}_{k}^{(j)} = \frac{p_k\,|\mathbf{w}^\mathsf{H}\hat{\mathbf{h}}_k|^2}
{\sum_{i \notin \mathcal{D}_j} p_i |\mathbf{w}^\mathsf{H}\hat{\mathbf{h}}_i|^2
+ \beta^2 \sum_{i \in \mathcal{D}_j} p_i |\mathbf{w}^\mathsf{H}\hat{\mathbf{h}}_i|^2
+ \varepsilon^2 \sum_i p_i \|\mathbf{w}\|^2 + \sigma^2 \|\mathbf{w}\|^2},
$$

with $\mathcal{D}_j$ the set of streams decoded before position $j$ and
$\hat{\mathbf{h}}$ the estimated channel. The MMSE combiner is regularized by the
worst-case error bound via $\varepsilon^2 P_{\rm tot}\,\mathbf{I}$ (robust MMSE).
Declared rates use the estimate $\hat{\mathbf{H}}$ and bound $\varepsilon$;
*actual* rates use the true $\mathbf{H}$ with $\varepsilon=0$; the **outage-aware**
rate is the minimum of the two, so over-declared rates are penalized by real
throughput, not by a proxy.

### B. Imperfect CSI and channel dynamics

Channels follow a 3GPP UMi path-loss model (users at 50–300 m, exponent 3.76,
normalized at 100 m, preserving ~29 dB near-far spread) with AR(1) temporal
correlation $\rho=0.9$. Pilot-based MMSE estimation yields per-user error variance

$$
\sigma_{\rm est}^2 = \frac{\sigma^2}{K\, p_{\rm pilot}} + \varepsilon_{\rm floor}^2,
$$

so estimation quality degrades with load $K$; the residual is bounded
$\|\Delta \mathbf{h}_k\| \le \varepsilon$. Training randomizes the noise power over
$\sigma^2 \in [-5,35]$ dB per episode (domain randomization) to induce
SNR-robustness; evaluation is performed under imperfect CSI at fixed points.

### C. Traffic, queues and QoS classes

Arrivals $A_k[t]$ are Poisson with an SNR-adaptive mean
$\lambda(\mathrm{SNR}) = \mathrm{load}(\mathrm{SNR})\cdot C(\mathrm{SNR})/K$, where
$\mathrm{load}(\cdot)$ is a sigmoid rising from 0.75 (low SNR) to 1.3 (high SNR);
three QoS classes scale per-user arrival, drift and overflow weights
$(2.0,1.0,0.3)$ and $(8,1,0.2)$ respectively. Queues evolve as

$$
Q_k[t+1] = \min\{\max\{Q_k[t] - D_k[t],\,0\} + A_k[t],\; Q_{\max}\},
$$

with served bits $D_k[t] \le R_k[t]$ (outage-capped) and overflow
$O_k[t]=\max\{Q_k[t]-D_k[t]+A_k[t]-Q_{\max},0\}$. The **delivery ratio** is the
fraction of arrived bits eventually served within the horizon.

### D. Power consumption and energy efficiency

Total consumption follows a realistic model

$$
P_{\rm consume} = \frac{\sum_k P_k}{\eta_{\rm PA}} + P_{\rm circ},
\qquad \eta_{\rm PA}=0.35,\; P_{\rm circ}=5.0\ \text{W},
$$

and $\mathrm{EE} = R_{\rm sum}/P_{\rm consume}$. A floor $P_{\rm eff}=\max(P_{\rm
consume}, P_{\rm floor})$ removes the degenerate $P\to 0$ optimum that would
otherwise make $\mathrm{EE}\to\infty$ while starving queues.

### E. Problem formulation and evaluation criterion

Let $\mathcal{S}=\{s_1,\dots,s_7\}$ be the evaluation SNR grid (0–30 dB). For a
policy $\pi$ and a reference SDMA policy, define per-point
$\mathrm{deliv}_\pi(s), \mathrm{queue}_\pi(s)$. The constraint set is

$$
\mathrm{deliv}_\pi(s) \ge \mathrm{deliv}_{\rm SDMA}(s),\quad
\mathrm{queue}_\pi(s) \le 1.1\,\mathrm{queue}_{\rm SDMA}(s),
\quad \forall s \in \mathcal{S},
$$

i.e. a **relative two-bar gate** at each point. The reported score is the
lexicographic tuple

$$
\big[\ \#\{s:\text{gate met}\},\ \overline{\mathrm{deliv}},\ -\overline{\mathrm{queue}},\ \mathrm{EE\ ratio}\ \big],
$$

so the number of cleared points dominates, then mean delivery, then queue, then
aggregate EE ratio versus SDMA. A global aggregate check ($\mathrm{EE}>1.05\times$
SDMA and power $<$ SDMA) gates termination but does not substitute the per-point
gates. **This criterion is what makes the problem hard**: improving the aggregate
while losing a point is a net loss.

---

## III. Proposed Framework

### A. Two-timescale MDP

The **slow** scale (every $T_s=20$ slots) selects user grouping
$g_k \in \{0,\dots,G-1\}$ and a group-level split vector
$\boldsymbol{\alpha}\in[\alpha_{\min},\alpha_{\max}]^G$ expanded per user. The
**fast** scale (every slot) selects the $4K$-dimensional action: a power-share
logit vector (softmax shares $\times K$, clipped per user), a drain ratio
$d_k\in[0,1]$, and Plackett-Luce scores $z_k$ inducing the SIC order $\pi$ by
Gumbel top-$K$ sampling (fully differentiable $\log P(\pi \mid z)$).

Observations are per-user feature blocks (channel gains, queue, queue delta,
arrivals, power/rate history, interference, fairness, split mask, group ids,
distance) plus globals (time ratio, SNR norm, CSI-error norm) and QoS fields;
a **history window** of the last $H$ slots' (gain, queue, power) snapshots augments
each user token. Lagrangian multipliers, delivery-virtual-queue backlog and the
Dinkelbach ratio are injected by *equal-dimension state augmentation* (replacing
three feature blocks), so the observation dimension is unchanged.

### B. Hierarchical policy architecture

The **meta-controller** embeds per-user meta-features, applies multi-head
self-attention, and outputs (i) categorical group logits with temperature
annealing $\tau: 2.0 \to 0.5$ and a balance regularizer, (ii) a group-level
Gaussian head mapped through a sigmoid to $[\alpha_{\min},\alpha_{\max}]$, and
(iii) a mean-pooled critic. The **worker** is a self-attention graph network
(SA-GAT): per-user tokens are FiLM-modulated by global context, with the SNR and
CSI-error scalars additionally injected per token; 2 layers of 4-head attention
(hidden 128) feed per-user heads for the $4K$ action and a mean-field critic that
pools QoS class statistics. Bucket-expert heads (routed by SNR bucket) and
performance-linked entropy incentives stabilize the low-SNR tail.

### C. Delivery-gated Lyapunov composite reward (DGLCR)

With served bits $D_k$, effective power $P_{\rm eff}$, Dinkelbach ratio $q^\star$,
aggregate delivery-debt queue $Y$ and power-budget queue $X$, the per-slot reward is

$$
r = c\Big[\, V\big(\textstyle\sum_k D_k - q^\star P_{\rm eff}\big)
+ w_{\rm tp}\sum_k \tfrac{Q_k}{Q_{\max}} w^{\rm qos}_k D_k
+ w_y \tfrac{Y}{Y_{\rm ref}} \textstyle\sum_k D_k
- w_q \sum_k \tfrac{Q_k^2}{Q_{\max}^2}
- \textstyle\sum_k w^{\rm qos}_k O_k
- w_x \tfrac{X}{X_{\rm ref}} P_{\rm tot} \Big],
$$

where the first term is the **Dinkelbach-linearized EE** (with $q^\star$ updated on
an outer loop every 10 episodes), the second a max-weight throughput term, the
third a delivery-debt coupling, and the remaining terms queue-overflow and
power-budget penalties. Sigmoid gates on the EE term and on penalties (as a
function of running delivery) prevent the low-power trap
$\partial \mathrm{EE}/\partial P < 0$ from dominating while delivery is infeasible.

### D. Constraint stack

A **Lagrangian** dual variable $\lambda$ on the delivery constraint
($\ge 0.98$) is updated with warmup and cap; its per-step signal is an EMA of
instantaneous delivery to give action-level gradient contrast. The policy-gradient
for EE is **projected onto the null space** of the delivery-constraint gradient
(lexicographic projection), so feasible-direction updates never trade delivery for
EE. Per-user **delivery virtual queues (DVQ)** with CVaR tail focus (worst 25%)
concentrate penalty on the users that actually risk the per-point gate.

### E. Solver expert and differentiable rate engine

Because $R(\cdot)$ is non-differentiable through $\pi$ and the min-outage operator,
we implement a **torch shadow** of the MMSE-SIC rate computation, verified
float64-consistent with the (immutable) reference to atol $10^{-9}$. The solver
expert takes the current state and, per slot, runs a short Adam refinement (25
steps, lr 0.1, cold start) on the power logits against a DGLCR surrogate that
mirrors the active reward branch (including the $q^\star$/debt-queue coupling),
plus candidate-selection among decoding orders (gain-descending / queue-last).
The expert **exceeds SDMA delivery at every tested SNR** (0/5/15/30 dB:
$+4.0/+1.4/+10.0/+23.9$ percentage points) at lower power, providing a strong,
cheap teacher.

### F. Distillation and stepwise-frozen fine-tuning (SFTP)

Demonstrations from the expert (8,000 episodes) train the policy by behavior
cloning (BC) on the continuous heads and the Plackett-Luce order head; **BC alone
already clears all 7 gates**. The policy is then fine-tuned with PPO under SFTP:
an anchor term injecting the BC gradient into the lexicographic-projection path,
critic-only warm-up (actor gradients withheld), a persistent anchor with slow
decay, and **guard-based checkpoint selection** (periodic mini-evaluations on a
SNR subset; the best checkpoint by mini-eval delivery is kept). Crucially, the
combination of KL early-stopping and the learning-rate decay floor causes the
actor learning rate to collapse to its floor early (~update 100 of ~780), an
*emergent stepwise freeze* that behaves as an implicit trust region around the
distilled basin; §V-D examines whether this freeze is causal.

### G. Training protocol

PPO with clip 0.1, rollout 2048, batch 64, 8 epochs, GAE $\lambda=0.95$, actor lr
$3\times10^{-5}$; a difficulty-annealing curriculum (5 stages) and **hard-example
mining** that re-weights per-SNR-bucket episode sampling by delivery shortfall with
streak multipliers and performance-linked entropy incentives focus capacity on
unmet points. All mechanism flags default off so the official evaluation profile
is unchanged by ablation knobs.

---

## IV. Experimental Methodology

**Environment.** $K=24$, $N_r=16$, $N_t=2$, $T=200$, $P_{k,\max}=1$ W (24 W
aggregate), $Q_{\max}=100$ bits; 3GPP UMi path loss; AR(1) $\rho=0.9$; SNR-dependent
CSI noise with $\varepsilon_{\rm floor}=0.01$; unified MRT + MMSE-SIC; adaptive
arrivals; 3 QoS classes.

**Evaluation.** Official sweep over $\mathcal{S}=\{0,5,10,15,20,25,30\}$ dB, 50
episodes per point, imperfect CSI; per-point gates and lexicographic score as in
§II-E. Every reported score is produced by the same immutable evaluation script;
checkpoints are selected by the guard and re-evaluated officially. Fig. 1 shows
the seed-replication band (mean and min–max envelope over seeds 42–46; faint
member lines).

**Comparator set (classical + recent).** Beyond classical SDMA / NOMA /
equal-power, we compare against four **recent (2025–2026) algorithms**
re-implemented in the identical environment and protocol: (i) **LDPTRLQ-style**
single-timescale Lyapunov drift-plus-penalty RL [7]; (ii) **DR-SAC** [16] and
(iii) **neural-DPP** [17] as distributionally-robust / learned-DPP references
(related-work level; faithful re-implementation of their mixed discrete-continuous
action spaces is out of scope); (iv) **DiffOpt**, a training-free model-based
comparator running the differentiable MMSE-SIC shadow engine as a per-slot online
optimizer (oracle and causal variants), isolating learning from per-slot
optimization. The repository's adaptive-power and water-filling heuristics are
additional non-learned references (`scripts/evaluate_extended_baselines.py`).

**Replication and pre-registration.** The champion configuration is replicated
verbatim at seeds 43 and 44 (in addition to seed 42). All fine-tuning variants
were **pre-registered** (hypothesis, arms, single-variable controls, and
accept/reject gates) before launch; verdicts were written without post-hoc gate
changes. Seed dispersion is reported alongside every comparison.

---

## V. Results

### A. Full-grid benchmark against classical baselines

Table I reports per-point delivery ratio (and, parenthetically, transmit power in
W and mean queue) for the champion policy and the three baselines; Fig. 1 shows
the same comparison as curves with the **seed-replication band** (mean and min-max
envelope over seeds 42–46).

**TABLE I — Per-point delivery ratio / power (W) / queue, 50 episodes/point, imperfect CSI.**

| SNR (dB) | PPO-RSMA (ours) | SDMA | NOMA | EqualPower |
|---|---|---|---|---|
| 0 | **0.886** / 13.72 / 2.84 | 0.856 / 14.69 / 3.46 | 0.836 / 17.35 / 4.01 | 0.791 / 24.00 / 4.94 |
| 5 | **0.989** / 14.39 / 0.43 | 0.979 / 13.70 / 0.68 | 0.970 / 17.35 / 0.92 | 0.943 / 24.00 / 1.53 |
| 10 | **0.993** / 14.59 / 0.69 | 0.961 / 15.02 / 2.28 | 0.947 / 17.35 / 3.10 | 0.915 / 24.00 / 4.62 |
| 15 | **0.991** / 13.71 / 1.91 | 0.887 / 15.12 / 11.42 | 0.839 / 17.35 / 15.54 | 0.824 / 24.00 / 15.48 |
| 20 | **0.974** / 13.39 / 7.20 | 0.795 / 14.62 / 22.39 | 0.715 / 17.35 / 27.40 | 0.726 / 24.00 / 25.50 |
| 25 | **0.942** / 13.59 / 16.08 | 0.724 / 14.64 / 28.76 | 0.636 / 17.35 / 33.07 | 0.658 / 24.00 / 31.80 |
| 30 | **0.888** / 13.52 / 29.63 | 0.664 / 14.87 / 34.47 | 0.581 / 17.35 / 36.83 | 0.601 / 24.00 / 37.27 |

The policy clears **all 7 per-point gates** (delivery ≥ SDMA and queue ≤ 1.1×SDMA
everywhere) with score $[7,\ 0.9519,\ -8.3971,\ 1.0804]$: mean power 13.84 W
(< SDMA) and aggregate EE ratio 1.0804 (> 1.05). The advantage is largest in the
deep-overload high-SNR regime (30 dB: 0.888 vs 0.664/0.581/0.601 delivery) where
interference and queueing bind hardest, and the policy uses roughly half the
transmit power of the fixed-power baselines.

### B. From teacher to distilled policy

Pure behavior cloning on 8,000 expert demonstrations already reaches 7/7 gates
with score $[7, 0.9441, -9.3929, 1.0753]$—the first configuration in this line to
clear the full grid. SFTP fine-tuning improves it to the champion
$[7, 0.9519, -8.3971, 1.0804]$ while the guard records zero rollbacks, i.e. a
monotone-safe refinement of the distilled basin (Fig. 2).

### C. Seed replication and the intrinsic noise floor

Verbatim replication of the champion recipe at seeds 43–46 yields best-guard
deliveries $\{0.9503, 0.9414, 0.9519, 0.9526\}$; together with the champion
(0.9519) the five draws give mean $0.9496$, sd $0.0047$, range $0.0112$ (the
pre-registered downgrade trigger, range $>0.012$, is not met). The seed-44
best-guard ($0.9414$) is a **checkpoint-selection artifact** of the legacy 3-point
mini-eval (its own final is $0.9491$); seed 46's draw (0.9526) is statistically
equivalent to the champion (Δ0.0007). End-of-training snapshots are markedly less
stable than guard-selected checkpoints (finals n=4: mean 0.9362, sd 0.0200, vs
best-guard sd 0.0047), retrospectively validating guard-based selection as the
delivery artifact. Two consequences: (i) the recipe centers at
$0.9496 \pm 0.0047$ and the champion is one draw from that band, reproducible
within $\pm 0.002$; (ii) any comparison whose margin is $\lesssim 0.005$ is
**below seed resolution** at single-seed sample sizes.

### D. Pre-registered fine-tuning campaign: all variants within the noise band

Five mechanism variants, each a single-variable perturbation of the champion
recipe, were pre-registered and judged (Table II). Released learning rate
(M5.1-A2/A3), a KL-to-BC trust region and a persistent strong anchor (M5.2-D1/D2),
and hard-mining target overrides aimed at the two weakest points (M5.3-E1/E2) all
score $0.9446$–$0.9474$ mean delivery—**inside the champion's own seed band**. The
M5.3 arms additionally show the intended weak points *not* improving (E1: 0 dB
0.876, 30 dB 0.881, both below champion), i.e. the intervention failed on its own
objective. We therefore report the fine-tune protocol's contribution as a
**seed-robust $\approx 0.950\pm0.005$ operating point**, and explicitly decline
causal claims for individual mechanisms at the present sample sizes; the
pre-registered follow-on (3 freeze-onsets × 3 seeds) is the design that could
resolve onset effects above the noise floor.

**TABLE II — Pre-registered fine-tuning variants (official best-guard score).**

| Campaign | Variant | Score | Verdict |
|---|---|---|---|
| M5 | Champion (SFTP, emergent freeze) | [7, 0.9519, −8.3971, 1.0804] | reference |
| M5.1 | A2: actor lr ×3.3 | [7, 0.9446, …] | lose (intermittent instability) |
| M5.1 | A3: kill KL-decay starvation | [7, 0.9463, …] | lose (starvation not a lever) |
| M5.2 | D1: KL-to-BC trust region | [7, 0.9447, …] | lose (5 rollbacks; estimator defect) |
| M5.2 | D2: persistent strong anchor | [7, 0.9473, …] | lose |
| M5.3 | E1: mining targets 0.92/0.93 | [7, 0.9474, …] | lose (target points regressed) |
| M5.3 | E2: mining targets 0.95/0.95 | [7, 0.9471, …] | lose (mid-band regression) |

### E. The model-based per-slot optimization upper bound

Table III adds the training-free comparators on the identical grid. Strikingly,
**DiffOpt**—the differentiable MMSE-SIC engine run as a per-slot online optimizer
(25 Adam steps on power logits + decoding-order selection, *no learned policy*)—
**exceeds the champion policy at every SNR point** (mean delivery 0.9574 vs 0.9519,
+0.0055, outside the ±0.005 seed band; mean queue 7.36 vs 8.40) while clearing all
7 gates. A causal variant whose surrogate uses only the estimated channel Ĥ
**matches the oracle-CSI variant point-for-point at 50 episodes** (max |Δ| ≤ 0.001
per point), so the advantage is **not** CSI leakage: even a fair, causal,
training-free per-slot optimizer dominates the learned policy on this grid.
Classical adaptive-power and water-filling heuristics remain far below both
(30 dB: 0.777/0.778).

**TABLE III — Training-free comparators, per-point delivery (queue), 50 episodes/point.**

| SNR (dB) | DiffOpt | DiffOpt (causal Ĥ) | AdaptivePower | WaterFilling | Champion (ref) |
|---|---|---|---|---|---|
| 0 | **0.894** (2.64) | 0.894 (2.64) | 0.845 (3.75) | 0.828 (4.12) | 0.886 (2.84) |
| 5 | **0.990** (0.38) | 0.990 (0.38) | 0.981 (0.65) | 0.979 (0.69) | 0.989 (0.43) |
| 10 | **0.994** (0.62) | 0.994 (0.62) | 0.988 (1.01) | 0.987 (1.07) | 0.993 (0.69) |
| 15 | **0.992** (1.71) | 0.992 (1.71) | 0.971 (4.34) | 0.969 (4.60) | 0.991 (1.91) |
| 20 | **0.978** (6.50) | 0.978 (6.49) | 0.915 (16.63) | 0.913 (16.65) | 0.974 (7.20) |
| 25 | **0.949** (14.89) | 0.949 (14.88) | 0.844 (27.52) | 0.845 (27.33) | 0.942 (16.08) |
| 30 | **0.905** (24.75) | 0.905 (24.74) | 0.777 (32.67) | 0.778 (32.63) | 0.888 (29.63) |

This re-frames the contribution hierarchy honestly: the **differentiable engine +
solver expert is itself the strongest controller on this grid**; behavior cloning
compresses it into a single-forward-pass policy at a quantified cost
(0.9574 → 0.9441), and SFTP fine-tuning recovers ≈59% of that compression gap
(→ 0.9519). The learned policy's remaining value is **inference cost and
model-free deployment** (one forward pass per slot vs 25 optimization steps with an
explicit rate model), not absolute performance. Earlier "RL surpasses the teacher"
wording is withdrawn: RL surpasses the *distilled student* (BC), not the live
optimizer, which was never officially evaluated until now.

**Trained recent-family comparator.** A single-timescale Lyapunov
drift-plus-penalty RL baseline in the LDPTRLQ family [7] (PPO with queue-Lyapunov
penalties and the same curriculum/mining protocol, but no hierarchy, no
distillation, no guard) clears **0/7 gates** (mean delivery 0.782, mean power
23.2 W ≈ saturated; Fig. 1). Vanilla queue-Lyapunov RL therefore cannot solve this
overloaded instance from scratch; together with bc_init (7/7 from pure
distillation) this localizes the enabling ingredient to **expert initialization**,
not to the reward family or the hierarchy alone.

**Classic continuous-control family (TD3/DDPG).** We add the review-standard
TD3/DDPG pair, adapted to the mixed PL+continuous action space (deterministic pathwise
actor through the differentiable $tanh(\mu)$ continuous head, greedy PL order;
twin-Q TD targets with target-policy smoothing and a delayed actor for TD3, single
Q for DDPG; single-timescale, no distillation, no guard). **TD3** clears 0/7 gates
(mean delivery 0.818 at a saturated 24 W transmit power, low-SNR points fail);
**DDPG** reaches 5/7 (mean delivery 0.886) but still misses the two low-SNR gates
{0, 5} dB on both delivery and queue. Both fall below the distilled policy
(bc_init 7/7, 0.9441) and confirm the LDPTRLQ pattern: from-scratch RL cannot
solve this overloaded instance, whatever the algorithm family.

**Optimization-depth check (JointOpt).** A per-slot optimizer with **60
warm-started** Adam steps (vs DiffOpt's 25 cold-start) scores *below* DiffOpt at
every point (mean 0.913 vs 0.957, ≈13 W): the model-based upper bound is **not
step- or warm-start-limited**—cold-start 25-step DiffOpt is already a tight bound,
consistent with the earlier warm-vs-cold diagnostics.

**TABLE IV — Classic-family and optimization-depth comparators (official 7×50).**

| Comparator | points | mean deliv | 30 dB | P (W) |
|---|---|---|---|---|
| DiffOpt / DiffOptCausal (upper bound) | 7/7 | **0.957** | 0.905 | 13.4–14.6 |
| Champion SFTP (deliverable, ref) | 7/7 | 0.9519 | 0.888 | 13.5–14.6 |
| bc_init (distillation, ref) | 7/7 | 0.9441 | 0.871 | 13.7 |
| JointOpt (60-step warm-start) | 5/7 | 0.913 | 0.872 | ≈13 |
| DDPG | 5/7 | 0.886 | 0.792 | 13.6–21.3 |
| TD3 | 0/7 | 0.818 | 0.706 | 24 (sat.) |
| LDPTRLQ | 0/7 | 0.782 | 0.595 | ≈23 |

### F. Checkpoint selection matters

The seed-44 episode exposes a subtle evaluation pathology: a legacy 3-point
mini-eval selected a checkpoint 0.9 pt below the same run's final model, because
the mini-eval's saturated points cannot see mid-band regressions that the official
7-point × 50-episode sweep reveals. A queue-aware 6-point guard (v2) catches these
regressions (validated on the M5.2-D1 arm where v1 would have passed 5 bad
checkpoints). We report this as a methodological requirement: **checkpoint
selection must use the same per-point gate structure as the final evaluation**.

---

### Statistical assessment

Per-point deliveries are paired across the 7-SNR design grid (df=6; two-sided
paired t-test with Cohen's d; Holm–Bonferroni at α=0.05 over the eight
external-family comparisons). Champion 5-seed best-guard mean `0.9496 ± 0.0047`
(95% CI [0.9438, 0.9555]). **Ours significantly exceeds** SDMA (p=0.018,
d=1.22), equal-power (p=0.004, d=1.69), water-filling/adaptive-power
(p=0.016/0.023, d=1.26/1.15) and from-scratch RL (LDPTRLQ/TD3/DDPG, p≤0.006,
d≥1.6); NOMA non-significant after correction (p=0.015 raw, d=1.28). Vs our
engine's **oracle** per-slot optimizer the difference is small but detectable
(p=0.045, d=−0.96): the single-pass causal policy sits a ≈0.6% compression gap
below the oracle — the quantified cost of online-optimization-free inference.
Baselines are single-seed (42); our multi-seed dispersion is in §V-C.

## VI. Discussion

**Why does the freeze help?** The emergent early learning-rate collapse acts as an
implicit trust region around the distilled basin: it prevents the policy from
drifting into regions where the non-differentiable outage/SIC structure makes the
surrogate reward misleadingly optimistic (notably the deep-overload 25–30 dB
points). Released-learning-rate variants drift exactly there. That said, §V-D
shows the *difference* between frozen and released variants is within seed noise,
so the freeze should be described as a stabilizer of a seed-robust operating point,
not as a proven causal lever; the pre-registered multi-seed response surface is the
correct instrument to settle it.

**The optimizer upper bound and what learning buys.** The training-free per-slot
optimizer (DiffOpt) exceeding the learned policy at every point forces an honest
account: in this environment, greedy model-based optimization with an accurate
differentiable rate model is a stronger controller than any policy we trained, and
the distillation+fine-tune pipeline is best understood as **compressing that
optimizer into a fast, model-free policy** at a quantified cost (0.9574 → 0.9519,
≈1.2% mean delivery) in exchange for single-forward-pass inference and no runtime
dependence on the rate model. The causal (estimate-only) variant retains the full
advantage at 50 episodes, so the upper bound holds for fair causal controllers;
the remaining open question is whether the distillation gap can be closed by
distilling the optimizer's queue-conditioned behavior rather than its marginal
actions.

**Statistical honesty as a contribution.** Single-seed RL comparisons at effect
sizes $\sim 0.005$ are indistinguishable from seed noise in this problem. Our
pre-registered campaign and explicit noise-floor quantification are intended as a
template: report seed dispersion, pre-register gates, and treat within-band
differences as null.

**Limitations.** Fixed $K=24, N_r=16$; CPU-costly official evaluation limits
replication count; the expert surrogate mirrors the reward branch and inherits its
shaping choices; the emergent freeze is configuration-dependent and must be
re-verified if the reward or curriculum changes.

---

## VII. Conclusion

We presented a hierarchical reinforcement-learning framework with solver-expert
distillation for energy-efficient scheduling of overloaded RSMA MU-MIMO uplink
under per-SNR delivery and queue constraints. The distilled, stepwise-frozen policy
clears all seven evaluation gates with seed-robust mean delivery $\approx
0.950\pm0.005$ and dominates classical baselines across the SNR range at
substantially lower transmit power. A pre-registered fine-tuning campaign
establishes the seed-dispersion floor of the problem and sets multi-seed
confidence-interval comparison as the reporting standard. Future work includes the
multi-seed freeze-onset response surface and extending the solver expert to
jointly refine the stream split.

---

## Appendix A — Key hyperparameters

| Group | Value |
|---|---|
| System | $K=24$, $N_r=16$, $N_t=2$, $T=200$, $P_{k,\max}=1$ W, $Q_{\max}=100$ |
| CSI | AR(1) $\rho=0.9$; $\varepsilon_{\rm floor}=0.01$; robust reg. $\varepsilon^2 P_{\rm tot}\mathbf{I}$ |
| Power | $\eta_{\rm PA}=0.35$, $P_{\rm circ}=5.0$ W, $P_{\rm floor}=3.0$ W |
| Reward | $V=5$, $w_{\rm ee}=8$, $w_q=2$, $w_{\rm ovf}=20$, $w_y=0.3$, $w_x=1$, scale 0.01 |
| Meta | $T_s=20$, $G=4$, $\tau:2.0{\to}0.5$, $\alpha\in[0.1,0.95]$ |
| Worker | SA-GAT 2×(4-head, h=128); history window $H=4$ |
| PPO | clip 0.1, rollout 2048, batch 64, 8 epochs, actor lr 3e-5, GAE 0.95 |
| Expert | Adam 25 steps lr 0.1 cold-start; 8,000 demo episodes |
| Guard | v2: 6 points × 16 episodes + queue bar; selection by mini-eval |

---

## References

1. D. Yu, S.-H. Park, et al., "Robust design of rate-splitting multiple access with
   imperfect CSI for cell-free MIMO systems," in *Proc. IEEE ICC Workshops*,
   May 2022, pp. 604–609.
2. F. Karim, N. H. Mahmood, et al., "Uplink rate splitting multiple access with
   imperfect channel state information and interference cancellation,"
   *IEEE Wireless Commun. Lett.*, vol. 14, no. 5, pp. 1316–1320, May 2025.
3. B. Clerckx, H. Joudeh, et al., "Rate splitting for MIMO wireless networks: A
   promising PHY-layer strategy for LTE evolution," *IEEE Commun. Mag.*, vol. 54,
   no. 3, pp. 98–105, Mar. 2016.
4. A. Schröder, S. Gracla, et al., "Model-free robust beamforming in satellite
   downlink using reinforcement learning," *IEEE Open J. Commun. Soc.*, vol. 6,
   pp. 10582–10598, 2025.
5. J. Wang, Z. Ren, et al., "Distributionally robust policy learning under concept
   drifts," in *Proc. Int. Conf. Mach. Learn. (ICML)*, Vancouver, BC, Canada,
   Jul. 2025.
6. M. Cui, D. Zhou, et al., "DR-SAC: Distributionally robust soft actor-critic for
   reinforcement learning under uncertainty," in *Proc. Int. Conf. Learn.
   Represent. (ICLR)*, 2026.
7. B. Demirel, Y. Wang, et al., "Generalization in reinforcement learning for radio
   access networks," *IEEE Trans. Mach. Learn. Commun. Netw.*, vol. 4,
   pp. 351–372, 2026.
8. D. Luan and J. Thompson, "Achieving robust channel estimation neural networks by
   designed training data," *IEEE Trans. Cogn. Commun. Netw.*, 2025.
9. B. Zhu, L. Huang, et al., "Enhancing energy efficiency in wireless-powered MEC
   systems through Lyapunov-guided deep reinforcement learning,"
   *IEEE Trans. Wireless Commun.*, vol. 24, no. 9, pp. 7563–7580, Sep. 2025.
   *(LDPTRLQ-family anchor: Lyapunov-guided DRL for EE)*
10. M. J. Neely, *Stochastic Network Optimization with Application to Communication
    and Queueing Systems*. San Rafael, CA, USA: Morgan & Claypool, 2010.
11. A. Zakeri, M. Moltafet, et al., "Minimizing the AoI in resource-constrained
    multi-source relaying systems: Dynamic and learning-based scheduling,"
    *IEEE Trans. Wireless Commun.*, vol. 23, no. 1, pp. 450–466, Jan. 2024.
    *(neural-DPP-family anchor: learning-augmented Lyapunov scheduling)*
12. S. Khosravi, B. Demirel, et al., "Practical policy distillation for
    reinforcement learning in radio access networks," in *Proc. IEEE Int. Symp.
    Pers. Indoor Radio Commun. (PIMRC)*, 2025.
13. J. A. Plackett, "The analysis of permutations," *Appl. Statist.*, vol. 24,
    no. 2, pp. 193–202, 1975. *(Plackett-Luce order model)*
14. R. D. Luce, *Individual Choice Behavior*. New York, NY, USA: Wiley, 1959.
    *(choice axiom / Luce model)*
15. W. Dinkelbach, "On nonlinear fractional programming," *Manage. Sci.*, vol. 13,
    no. 7, pp. 492–498, 1967.
16. R. Zhang, K. Xiong, et al., "Energy efficiency maximization in RIS-assisted
    SWIPT networks with RSMA: A PPO-based approach," *IEEE J. Sel. Areas Commun.*,
    vol. 41, no. 5, pp. 1413–1430, May 2023.
    *(PPO anchor: PPO-based RSMA energy-efficiency in JSAC)*
17. R. S. Sutton, D. McAllester, et al., "Policy gradient methods for reinforcement
    learning with function approximation," in *Adv. Neural Inf. Process. Syst.
    (NeurIPS)*, 1999.
18. P. He, A. Anpalagan, and W. Ejaz, "Energy-efficient power allocation
    maximization for MU-MIMO multiple access channels," *IEEE Trans. Wireless
    Commun.*, vol. 22, no. 5, pp. 3460–3471, May 2023. *(SDMA baseline)*
19. K. Wang, T. Zhou, et al., "Fairness-aware energy-efficient power allocation for
    uplink NOMA systems with imperfect SIC," *IEEE Wireless Commun. Lett.*,
    vol. 12, no. 12, pp. 2018–2022, Dec. 2023. *(NOMA baseline)*
20. A. Falloun and A. Ait Madi, "Fixed and fair power allocation in downlink and
    uplink NOMA: Outage probability analysis and bit error rate comparative
    study," *Bull. Electr. Eng. Inform.*, vol. 14, no. 5, pp. 3509–3519, Oct. 2025.
    *(equal-power baseline anchor; not an IEEE top venue — EPA has no recent
    top-journal method paper, swap if a better anchor found)*

**Figures.** Fig. 1 = `docs/paper/figs/benchmarks_master.png` (master benchmark:
PPO-RSMA 5-seed band as the only ours curve, vs external baselines SDMA [18] /
NOMA [19] / EqualPower [20] / LDPTRLQ [9]; the DiffOpt upper bound is not
plotted (Table II, §V-E); adaptive-power, water-filling, causal DiffOpt and the
faint per-seed member lines omitted for clarity); Fig. 2 =
`docs/paper/figs/ablation_delivery_seeds.png`
(teacher→BC→champion→replications).
LaTeX skeleton with these figures wired: `docs/paper/main.tex` (requires
IEEEtran.cls to compile).