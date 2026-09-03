# 研究定案：可微引擎 + 求解器专家的代码事实依据

（2026-09-02 探索期整理，全部结论带行号可复核）

## 1. 物理链路（引擎需精确复刻的最短路径）

### 动作解析 `rsma_env._parse_action`（rsma_env.py:349-382）
- 3K 连续动作 clip 到 [-1,1]；mapped = (a+1)/2
- `group_level_alpha=True`（默认）：**α = clip(split_mask, 0, 1)**，Worker α 块死维（rsma_env.py:365-367）；False 时 α = mapped·split_mask
- `use_softmax_power_share=True`（默认）：p_logits = **原始 a[K:2K]**（不做 (a+1)/2 映射）→ 减 max → p_ratio = K·exp/(Σexp+1e-12) → clip[0,1]；p_total = ratio·P_k_max（P_k_max=1.0W）
- 分层时 p_total = min(p_total, P_target/K)（rsma_env.py:760-762）
- d_ratio = mapped[2K:3K] ∈ [0,1]
- 4K 动作尾 K 维 = decoding_order 优先级（值小先解码），rate_calc 内 argsort 升序解析（rate_calc.py:284-290）

**总功率可控性**：Σp = P_k_max·K·e^{−lmax}·(Σe^{l})/(Σe^{l})——均匀 logits（任意同值）→ Σ=24W；一个 +1 其余 −1 → Σ≈8.8W。**logit 展宽即省电**，动作空间内自由。

### 预编码 `beamforming.compute_precoding_v2`（beamforming.py:147-178）
- 每用户 dir_k = H[k] 的最大右奇异向量（SVD，Vh[0].conj()，退化回退 1/√N_t·ones，beamforming.py:184-205）
- W_c[k] = dir_k·√(max(α·p·split, 0))，W_p[k] = dir_k·√(max((1−α·split)·p, 0))
- **torch 复刻**：dir 无梯度需求（H 是常量）→ no_grad 批量 SVD 精确一致；SVD 相位不唯一但 SINR 相位不变（h^H R^{-1} h 与 outer(h,h^H) 均相位不变）→ 速率一致

### 速率 `rate_calc.compute_rates_mmse_sic`（rate_calc.py:249-429，纯 numpy）
- h_c[k] = H[k]@W_c[k] ∈ C^N_r，h_p 同理；p_c[k] = ‖W_c[k]‖²
- robust 正则：ε²·(Σp_c+Σp_p)·I（ε>0 时）
- **Phase 1 公共流**（按 argsort(decoding_order) 顺序）：
  R_int[k] = σ²I + robust + h_p[k]h_p[k]^H（自身私有流干扰）
    + Σ_{j≠k} M[j,k]·h_c[j]h_c[j]^H + ε²·Σ_{j≠k} R[j,k]·p_c[j]·I
    + Σ_{j≠k} h_p[j]h_p[j]^H（**注意与自身私有合并后 = Σ_ALL h_p[j]h_p[j]^H，与 k 无关**）
  其中 M[j,k] = β² 若（同组 且 rank[j]<rank[k]）否则 1；R[j,k] = 1 若（同组 且 rank[j]<rank[k]）否则 0
  SINR_c[k] = Re(h_c[k]^H·solve(R_int[k], h_c[k]))，max(·,0)
  h_c_norm[k] < EPS=1e-12 → SINR_c[k]=0
- **Phase 2 私有流**（同序）：
  R_int[k] = σ²I + robust
    + Σ_j 1(h_c_norm[j]≥EPS)·[β²·h_c[j]h_c[j]^H + ε²·p_c[j]·I]（所有公共流残余，**无组条件**）
    + Σ_{j≠k} M'[j,k]·h_p[j]h_p[j]^H + ε²·Σ_{j≠k} R'[j,k]·p_p[j]·I
  其中 M'[j,k] = β² 若 rank[j]<rank[k] 否则 1（私有流 SIC 无组条件）；R' 同构
  h_p_norm[k] < EPS → SINR_p[k]=0
- R = log2(1+SINR)，R_total = R_c + R_p
- **outage 模式**（imperfect CSI，rate_calc.py:431-489）：R = min(declared[H_hat, ε], actual[H_true, ε=0])，两通道 β 同传；W 都由 H_hat 生成
- env 调用点（rsma_env.py:769-794）：csi_mode=imperfect → error_bound = _channel_obs["sigma_est"]（channel 动态估计）；sic_imperfection 默认 0

### env step 关键量（rsma_env.py:824-859）
- usable = R_total；queue_before = traffic.queue_lengths.copy()
- max_sendable = max(min(R·τ, Q_before), 0)；D_k = d_ratio·max_sendable
- traffic.step(D)：Q_next = max(Q−D,0)+A（A~Poisson(λ_k)），overflow = max(Q_next−Q_max,0)，Q = min(Q_next, Q_max)
- 奖励调用：compute_reward(queue_before, served=D, power=p_total, overflow, prev_queue, sigma2, P_target, user_drift/overflow_weights(QoS), delivery_running, arrivals)
- q* 注入：env.lyapunov.dinkelbach_q = env.dinkelbach_eta（rsma_env.py:843）

## 2. Surrogate 目标（lyapunov.py DGLCR 活跃分支，精确镜像）

启用条件：enable_delivery_vq=True（config.py:210 硬编码 True）且 delivery_running 非 None 且非 DGMW panic（dgmw_enable=False 默认）。use_dinkelbach=True → ee_gate=penalty_gate=1（lyapunov.py:588-590）。

```
P_eff = max(ΣP/η + P_circuit, ee_power_floor)     # η=0.35, P_circuit=5.0, floor=3.0（待实现时从实例读）
ee_component = V · ee_weight · (ΣD − q*·P_eff)     # V=5.0, ee_weight=8.0, q*=env.dinkelbach_eta
tp_component = w_tp_dglcr · Σ(Q/Q_max · drift_w)·D # w_tp=3.0, drift_w=QoS 权重（hetero 时）
y_component = w_y_tp · (Y/dvq_y_scale) · ΣD        # Y=env.lyapunov._Y
q_penalty = ldptrlq_q_weight · ΣQ²/Q_max²          # 对当前动作无梯度（Q 是 before）
overflow_penalty = w_overflow·Σ(w_ov·OV) + w_early·ΣReLU(Q/Q_max−warn)²   # OV 期望近似 max(Q−D+λ−Q_max,0)
power_vq_penalty = w_power_vq · (X/pvq_scale) · ΣP # X=env.lyapunov._X
reward = reward_scale · (ee + tp + y − (q_pen + ov_pen + pvq_pen))
```

**实现约定：全部常数从 env.lyapunov 控制器实例属性读取**（运行时真值），不用 config 副本——天然对齐任何 config 覆盖。Y/X 只读不写（专家不改训练状态机的 VQ）。

D 单调性：ee/tp/y/overflow 均随 D 递增（或不减），d_ratio 最优恒为 1 → 专家固定 d=1（a_d = 0.999 tanh 上界保护，与 ppo.bc_pretrain 的 clamp(-0.999,0.999) 一致）。

## 3. HEK BC 通道（复用点）

- 收集：`_hek_collect_expert`（main.py:360-418）——env.reset → 构造 3K 动作 + order 拼 4K → env.step，收集 (obs, 3K)；不触碰网络与 torch RNG
- 门控：main.py:1043-1099，`hek_enabled` gate；收集后设 worker_trainer.hek_obs/act → update() 内衰减锚定自动生效（ppo.py:409-426）
- BC：`PPOTrainer.bc_pretrain`（ppo.py:220-264）——独立 Adam 只训 actor+log_std，loss=−cont_log_prob.mean()，act clamp(-0.999,0.999)；**decoding_orders=None 不训 PL 头**（本任务扩展点）
- config：ConstraintConfig hek_* 字段（config.py:350-358），默认 hek_enabled=False

## 4. 环境状态接入（专家读取清单）

- H_hat/H_true/sigma_est：env._channel_obs（rsma_env.py:765-780）
- Q：env.traffic.queue_lengths；λ_k：env.traffic（实现时确认属性名）
- split_mask/group_ids/P_target：env.split_mask（reset 默认 0.5·ones）、env.group_ids（默认 zeros(K)）、env.P_target（分层时 Meta 设置；None 时无裁剪）
- QoS 权重：env.qos_user_drift_weights / qos_user_overflow_weights（enable_hetero_qos 时）
- q*：env.dinkelbach_eta；Y/X：env.lyapunov._Y/_X
- sigma2：env.rate_calc.sigma2（每 episode 同步，rsma_env.py:597）
- τ/Q_max/P_k_max：env.tau / env.Q_max / env.P_k_max

## 5. 白名单与约束

- 新文件合规落点：src/optimization/、src/training/、src/models/、scripts/baselines/（optimize_loop_config.yaml mutable_files）；**本任务零 src/env/ 改动**
- 测试：tests/test_*.py + `pytestmark = pytest.mark.fast`（pytest.ini 已注册 marker）
- 诊断脚本 scripts/_diag_*.py 不跟踪（既有约定）
- 评估：evaluate_snr_sweep_temporal.py 包装器（h4 观测 checkpoint 需要）；官方对照 [5, 0.9163, -11.0186, 0.6914]，本机 Run 3 [5, 0.9041, -12.59, 0.6985]

## 6. 上阶段（DRL-RSMA）移植结论备忘

- 教师必须强：K8 自蒸馏（弱目标）0.856 < BC（oracle 目标）0.888 —— 专家用 surrogate 精修解而非当前策略蒸馏
- BC 后 RL 微调可能退化（0.873 < 0.888）—— 纯 BC 与 BC+PPO 双检查点都评估
- 精修超参：瓶颈是步长不是步数（lr 0.1 × ~150 步最优，收益 +0.74% 且 0.72× 耗时；本项目单槽更小用 ~25 步起步）
- 专家迭代（on-policy 重标注）在本项目成立：状态非 iid（动作→队列→未来状态分布分叉），协变量偏移真实存在
