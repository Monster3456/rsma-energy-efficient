"""系统配置：队列稳定上行MIMO-RSMA长期能效最大化"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SystemConfig:
    """物理层系统参数"""
    K: int = 24                    # 用户数 (Task#12: 过载场景 K=24>N_r=16, 强制分组/调度)
    N_r: int = 16                  # 基站接收天线数
    N_t: int = 2                   # 每用户发射天线数
    P_k_max: float = 1.0           # 每用户最大瞬时功率 (W) — 总功率上限 K*P_k_max=24W
    sigma2: float = 0.01           # 噪声功率（默认值；启用 SNR DR 时由环境每个 episode 重新采样）
    T: int = 200                   # Episode时隙数
    tau: float = 1.0               # 时隙持续时间 (ms), 归一化

    # 信道模型
    rho: float = 0.9               # AR(1)信道时间相关系数
    channel_model: str = "rayleigh"  # "rayleigh" or "rician"

    # Task#13: 3GPP Urban Micro 路径损耗模型
    # PL(d) = path_loss_ref_db + 10*path_loss_exponent*log10(d_km)
    # 50m: PL=79.2dB, 100m: PL=90.4dB, 300m: PL=108.4dB → 近远差 ~29dB
    # 通过 path_loss_ref_distance 处的 PL 做参考归一化（保持发射 SNR 标度不变，
    # 仅引入近远相对差异），保留 ~29dB 的用户间不平衡，用于驱动差异化功率分配/SIC。
    use_path_loss: bool = True
    cell_radius_min: float = 50.0          # 最近用户距离 (m)
    cell_radius_max: float = 300.0         # 最远用户距离 (m)
    path_loss_exponent: float = 3.76       # 路径损耗指数 (Urban Micro)
    path_loss_ref_db: float = 128.1        # 参考路径损耗 at 1km (dB)
    path_loss_ref_distance: float = 100.0  # 参考距离 (m), 在此距离处相对 PL=0

    # CSI不确定性
    csi_mode: str = "imperfect"   # "perfect" or "imperfect"
    epsilon_csi: float = 0.05      # 已弃用语义：旧版固定估计噪声标准差；保留作 snr_dependent_csi=False
                                   # 时的回退值，新版本中作为下界由 epsilon_floor 替代
    csi_delay: int = 1             # 反馈延迟（时隙数）
    outage_prob: float = 0.05      # 中断概率约束 ε

    # Task#20: SNR 依赖的信道估计噪声模型
    # MMSE 信道估计 sigma_est² = σ² / (N_pilot * P_pilot) + epsilon_floor²
    # 简化: N_pilot=K, P_pilot=pilot_power, 故 sigma_est² = σ² / (K*pilot_power) + epsilon_floor²
    # 这样低 SNR (大 σ²) 时估计噪声大，H_hat 反映 SNR 差异；高 SNR 时受 epsilon_floor 下界约束
    snr_dependent_csi: bool = True   # 是否启用 SNR 依赖的估计噪声 (False: 退化为固定 epsilon_csi)
    pilot_power: float = 1.0         # 导频发射功率 P_pilot
    epsilon_floor: float = 0.01      # 估计噪声下界（量化/硬件噪底）
    obs_include_csi_quality: bool = False  # 是否在 obs 中追加每用户估计不确定度（增加 K 维）

    # 预编码 / 接收机方案
    # "rzf": 统一 MRT 发射 + MMSE-SIC 接收 (默认, 设计文档 2026-06-05)
    # "mrt": 正交补 MRT 发射 + matched filter 接收 (旧路径, 向后兼容)
    precoding_type: str = "rzf"
    sic_imperfection: float = 0.0  # SIC 硬件不完美因子 β ∈[0,1]

    # 流量模型
    lambda_rate: float = 5.5       # Task#8: 高负载训练 - 总到达=5.5×16=88 bits/slot, 约93%容量@SNR=20dB; 低SNR(0-5dB)将超过容量, 迫使PPO使用高功率
    Q_max: float = 100.0           # 最大队列长度 (bits)
    packet_size: float = 1.0       # 每包大小 (bits, 归一化)

    # Power amplifier and circuit power model (Task#15)
    # 解决 PPO 总是收敛到极低功率的问题：现实功耗 P_consume = P_tx / pa_efficiency + circuit_power
    # 即使 P_tx = 0，P_consume 仍为 P_circuit，EE 上限有限，迫使 PPO 提升 R_sum 才能提高 EE
    pa_efficiency: float = 0.35        # η: PA efficiency (Class-B)
    circuit_power: float = 5.0         # P_circuit: static circuit power (W)

    # SNR Domain Randomization
    snr_domain_randomization: bool = True
    snr_min_db: float = -5.0
    snr_max_db: float = 35.0

    # Mean-Field MAPPO: per-user power is a softmax share of the total power budget
    use_softmax_power_share: bool = True

    # Task#9 / Task#27: SNR 自适应到达率 λ(SNR) = target_load_ratio * capacity(SNR) / K
    # 使各 SNR 下负载率保持在 ~80%（基于完美信道容量），让 PPO 在所有 SNR 均面临有意义的优化问题。
    # Task#27: 修复低 SNR 过载——估算公式改为 Shannon 完美容量，target=0.8，去除过高 lambda_min。
    adaptive_lambda: bool = True
    target_load_ratio: float = 0.8   # Task#27: 保留作为参考/回退；Task#44 起实际使用 SNR-adaptive ratio
    lambda_min: float = 0.2          # Task#46: 0.3→0.2，降低低 SNR 到达率下界以提高 delivery          # Task#43: 回滚 Task#41，0.5→0.3 恢复 v7 低 SNR 负载下限
    lambda_max: float = 10.0         # Task#27: 7.0→10.0，适当放宽高 SNR 上限

    # Task#44: SNR 自适应 target_load_ratio（sigmoid 形式）
    # ratio(snr_db) = r_min + (r_max - r_min) * sigmoid(steepness*(snr_db - snr_mid))
    # 低 SNR (0dB) ~0.35, 高 SNR (30dB) ~0.85
    load_ratio_min: float = 0.75      # 低 SNR 目标负载比（≥v7 常数 0.75）
    load_ratio_max: float = 1.3      # Task#48: 1.5→1.3，平衡高 SNR 吞吐与 delivery≥90%
    load_ratio_snr_mid: float = 15.0  # sigmoid 中点 (dB)
    load_ratio_steepness: float = 0.3   # Task#47: 0.15→0.3，加速过渡以保护低 SNR 不变

    # ---- 异构 QoS 配置 (v13) ----
    enable_hetero_qos: bool = True            # v13: 启用异构QoS
    qos_classes: int = 3                      # QoS 类别数
    qos_lambda_multipliers: tuple = (2.0, 1.0, 0.3)   # v15: was (1.5, 1.0, 0.5), 加大差异迫使优先级选择
    qos_drift_weights: tuple = (8.0, 1.0, 0.2)        # v15: was (3.0, 1.0, 0.5), 强化 delay-sensitive 优先
    qos_overflow_weights: tuple = (8.0, 1.0, 0.2)     # v15: was (3.0, 1.0, 0.5), 强化 delay-sensitive 溢出惩罚

    # ---- Delivery Virtual Queue (DVQ) CVaR 尾部聚焦配置 ----
    # 将 per-user DVQ 的均匀惩罚改为 CVaR 尾部聚焦：仅对虚拟积压最大的
    # worst-α 分位用户施加惩罚，把约束信用精准投到长期饿死的尾部用户，
    # 突破聚合 delivery 平台同时保护 EE。
    use_dvq: bool = True
    dvq_threshold: float = 0.98
    dvq_weight: float = 0.6
    dvq_cvar_alpha: float = 0.25       # CVaR 分位：惩罚 worst 25% 尾部用户
    dvq_warmup_episodes: int = 200
    dvq_z_max: float = 200.0

@dataclass
class LyapunovConfig:
    """Lyapunov漂移加惩罚参数

    paper_ee_v2: 引入队列感知的分层混合 reward
        w = (avg_queue_ratio) ** blend_alpha     # 队列紧迫度权重 0~1
        reward = w * drift_term + (1-w) * ee_bonus - overflow - outage
    高队列 -> 优先清队列(drift主导)；低队列 -> 优先省电(EE主导)。
    借此让功率随 SNR 自适应：高SNR->队列快清->w→0->省电；
    低SNR->队列积压->w→1->加大功率清队列。
    """
    # 大幅弱化线性 V*power 惩罚（EE 已隐式惩罚高功率）
    # Task#8: lambda_rate=5.5 高负载下极强稳定性约束 (V↓, V_max↓)
    V: float = 1.0
    V_min: float = 0.3
    V_max: float = 5.0
    V_exponent: float = 1.5
    adaptive_V: bool = True

    use_virtual_queue: bool = True
    vq_step_size: float = 0.1

    # SNR 感知 V 缩放（v2 中权重主要由 EE/queue 决定，仅作辅助）
    snr_gamma: float = 0.3
    snr_ref_db: float = 30.0
    v_eff_cap_factor: float = 3.0
    snr_floor: float = 0.9
    snr_power_exponent: float = 1.0

    # 最低功率地板：防止策略学到"几乎关机"
    P_min_floor: float = 0.5

    # 分层混合奖励（paper_ee_v2 核心）
    ee_mode: bool = True          # 启用 EE = R_sum / P_consume 激励 (Task#15: P_consume = P_tx/η + P_circuit)
    ee_weight: float = 8.0        # v22: 恢复 EE 激励（Lyapunov-DRL 统一框架需要 EE 信号）
    ee_power_floor: float = 3.0   # 恢复 v18 基线值
    # 队列紧迫度指数：>1 使低队列时快速切换到 EE 主导模式
    blend_alpha: float = 1.5      # Task#8: 让 EE 和稳定性更均匀混合

    # Lyapunov-DRL 权衡参数：V 越大越偏 EE，V 越小越保守保队列
    lyapunov_V: float = 5.0

    # CRPO 风格可行性门控：用 running delivery 对 EE 奖励项做软门控，
    # 违约时抑制 EE 让 policy 专注交付，可行时恢复 EE
    ee_gate_enable: bool = True        # 启用CRPO可行性门控
    ee_gate_threshold: float = 0.98    # 可行阈值(与delivery目标一致)
    ee_gate_band: float = 0.15         # 线性斜坡带宽, d_low=threshold-band
    ee_gate_min: float = 0.05          # 门控下限(不完全关闭EE以保留梯度)

    # P0 修复：drift 信号放大系数（原始 drift=Σ Q·ΔQ/Q_max² 量级 ±0.1，
    # 与 ee_bonus 30 量级相差 300×，policy 实际只追 EE 完全不管队列稳定。
    # 放大 50× → drift 量级 ±5，与 ee_bonus 可比，让队列稳定真有梯度信号）
    # Task#8: 进一步加强漂移信号以应对 lambda_rate=5.5 高负载（强信号驱动高功率清队列）
    # Task#15: 引入 PA + 电路功耗后 EE 数值从 ~25 降至 ~2-3，相应放大 ee_weight 至 10.0
    # Task#16 验证记录: drift_weight 60→60保持。曾尝试 120 但训练后 Delivery 反而从 92.2% 跌至 85-89%（参见 logs/full_high_delivery_ckpt_sweep.log）：PPO
    # 越过强 drift 仅增强了“面对高队列需提高功率”的局部增量梯度，但 ee_bonus=10*EE 仍使低功率局部最优占优。推荐后续走方案 B：PPO-Lagrangian。
    drift_weight: float = 60.0

    # v13s: LDPTRLQ 队列二次惩罚
    # reward = scale * (V * ee_bonus - ldptrlq_q_weight * ΣQ²/Q_max² - overflow)
    # 直接惩罚队列存在性（而非仅惩罚增长），从根本上压低稳态 Q
    ldptrlq_q_weight: float = 2.0   # v13s2: 降低10倍防止-5dB极端SNR下reward爆炸（原20→2）

    # P0 修复：早期溢出惩罚（连续渐进，避免阶跃突变）
    # 当 Q_k > overflow_warn_ratio · Q_max 时开始累积惩罚，惩罚为 (Q/Q_max - threshold)²
    # 让 policy 提前响应，不至于等到队列满才发现
    overflow_warn_ratio: float = 0.6
    w_early_overflow: float = 5.0

    # backpressure / max-weight drift reward:
    # 按队列积压加权实际服务量，将调度导向被饿死用户以抬升尾部 delivery、降队列
    w_backpressure: float = 25.0

    # 目标反转（structural innovation）：
    # 主奖励改为直接最大化交付凹效用 Σ√(D_k/deliv_ref)，凹性促使功率摊向饥饿尾部用户；
    # EE 由被门控的最大化目标降级为单侧下界软约束 -w_ee_floor·max(0, ee_floor - ee_value)。
    w_deliv: float = 3.0            # 交付凹效用权重（主目标）
    deliv_ref: float = 2.0          # 每用户交付归一化基准（bits），用于 sqrt(D_k/deliv_ref)
    ee_floor_ratio: float = 0.9     # EE 下界（相对当前 ee_value 的目标下界，单位同 ee_value）
    ee_floor: float = 30.0          # EE 绝对下界（served_bits/P_consume），低于此才罚
    w_ee_floor: float = 0.5         # EE 下界约束罚权重（静态 λ_ee）

    # DGMW: Delivery-Gated Max-Weight Pure Mode（结构性创新）
    # 当 running delivery 低于 panic_threshold 时，硬切换到纯 max-weight 吞吐量
    # 最大化模式（无 scale、无 EE、无任何惩罚稀释），创建压倒性梯度逃离低功率吸引子；
    # 进入 panic 后需 delivery 持续高于 recovery_threshold 才退出（迟滞防抖）。
    # 区别于 CRPO（仍含 q_penalty）和 DGLCR（sigmoid 混合+scale 压缩），DGMW
    # 提供未稀释的纯吞吐量梯度信道，从奖励层根除 ∂EE/∂P<0 引力。
    dgmw_enable: bool = False           # DGMW 已被优势标准化抹平幅度，关闭以消除死分支噪声
    dgmw_panic_threshold: float = 0.55  # delivery 低于此 → panic（纯吞吐量）
    dgmw_recovery_threshold: float = 0.85  # delivery 高于此 → 退出 panic（迟滞）
    dgmw_mw_weight: float = 100.0       # panic 模式下纯 max-weight 吞吐量权重（无 scale）

    # LDPS: Lyapunov Drift Potential Shaping（势函数奖励整形, Ng 1999 PBRS）
    # F(s) = -ldps_eta·ΣQ²/Q_max²，整形项 = ldps_eta·(Σprev_Q² - ldps_gamma·ΣQ²)/Q_max²，
    # 经 reward_scale 缩放后加入 reward。策略不变（PBRS 定理），提供逐步稠密的
    # 队列排水差分信用，穿透逐 SNR 桶优势标准化，帮助逃离低功率局部最优。
    ldps_enable: bool = False   # LDPS 已验证无效，关闭
    ldps_eta: float = 30.0      # 势函数强度，与队列排水信用量级匹配
    ldps_gamma: float = 0.99    # 与 PPO gamma 一致，保证望远镜求和守恒

    # Dinkelbach 变换：将 EE 比率目标 max R/P 替换为线性目标 max R - q*·P
    # 结构性消除 P→0 处的奇点吸引子（低功率局部最优的根因）。
    # 比率梯度 ∂(R/P)/∂P = (P·dR/dP − R)/P² 在低功率为负（拉向更低功率）；
    # 线性梯度 ∂(R−q*P)/∂P = dR/dP − q* 在低功率为正（log 曲线陡峭段，推高功率）。
    # q* 由外层按 episode 批量更新（线性收敛），不再由 Lyapunov 每步内部 EMA。
    use_dinkelbach: bool = True
    dinkelbach_q_init: float = 1.0        # q* 初始估计
    dinkelbach_decay: float = 0.99        # EMA 衰减系数（~100 步窗口，外层更新时不再使用）
    dinkelbach_q_min: float = 0.01        # q* 下界
    dinkelbach_q_max: float = 50.0        # q* 上界
    dinkelbach_outer_update: bool = True  # 由训练循环外层更新 q
    dinkelbach_update_interval: int = 10  # 每多少 episode 更新一次 eta
    dinkelbach_anchor_alpha: float = 0.2  # 首周期后 EMA 更新步长

    # State-Augmented 观测：将 λ 桶值 / DVQ 积压 / eta 归一化后注入观测
    # 通过等维替换保持 obs_dim 不变（替换 interference/fairness/distance 三块）
    state_aug_enabled: bool = True

@dataclass
class RewardConfig:
    w_overflow: float = 20.0  # Task#8: 高负载下强惩罚溢出（Task#16 试过 40.0但效果不明显，overflow_rate 本就~0，未激活）
    w_outage: float = 3.0
    w_power_exceed: float = 10.0
    reward_scale: float = 0.01
    ee_sat_cap: float = 8.0  # EE 满意化: 凹 tanh 饱和拐点, 抑制无限压功率换 EE


@dataclass
class ConstraintConfig:
    """PPO-Lagrangian 硬约束配置（Task#17）

    将 Delivery >= delivery_threshold 作为硬约束，引入自适应对偶乘子 λ。
    每步从 reward 中减去 λ * max(0, threshold - delivery_running)，
    每 episode 结束后通过对偶上升更新 λ。该惩罚不被队列比例缩放，可
    与 ee_bonus 平等竞争梯度。
    """
    use_lagrangian: bool = True
    delivery_threshold: float = 0.98
    # v15: 从零训练时 lr 保持 0.05，但 init 降低避免初期爆炸
    lagrangian_lr: float = 0.05
    lagrangian_init: float = 5.0       # v15: 100→5，从零开始时避免初期 penalty 爆炸
    lagrangian_max: float = 100.0      # v15: 500→100，限制最大约束强度
    lagrangian_warmup: int = 200       # v15: 500→200，更早介入约束
    lagrangian_decay: float = 0.999

    # SLCP-D: 将约束 penalty 的输入从 episode 级累积交付率改为 per-step 即时交付率的 EMA，
    # 创建动作→penalty 差异化梯度。False 时完全回退到原 running_delivery 路径。
    use_step_penalty: bool = True
    slcp_ema_beta: float = 0.85  # per-step 即时交付率 EMA 平滑系数(抗 Poisson 噪声); 越大越平滑

    # Neely per-user Delivery Virtual Queue (DVQ)：逐步逐用户 credit assignment，
    # 与全局 Lagrangian 对偶乘子互补，为长期欠交付用户提供更强梯度信号。
    use_dvq: bool = True
    dvq_weight: float = 0.5
    dvq_z_max: float = 200.0
    dvq_warmup: int = 200

    # RCPO cost-critic 优势分解：delivery 约束从 reward 减法迁移到独立 cost 通道。
    # True 时 main.py 不再从 reward 减 constraint_penalty（reward 恢复纯 EE 平稳），
    # 约束改由 policy loss 中 A_eff = A_r - λ_scaled·A_c 施加，cost 用每步归一化
    # 交付赤字 max(0, thr·arrivals-served)/arrivals 提供稠密因果信用。
    use_rcpo_cost_critic: bool = False   # RCPO 已验证无效，回退到简单 Lagrangian 减法
    rcpo_lambda_scale: float = 0.2      # λ(reward尺度) 映射到归一化 advantage 尺度的系数
    cost_critic_coef: float = 0.5       # cost value loss 权重
    cost_gamma: float = 0.99            # cost GAE 折扣

    # Delivery-Barrier 安全层（动作层干预，区别于奖励层/优势层方法）：
    # running delivery 不可行（run_deliv < delivery_threshold）时，将 worker
    # 连续动作 p_total/d_ratio 向上投影到状态相关地板（随队列积压与交付赤字
    # 抬升），并用 evaluate_actions 重算 log_prob 保证 PPO on-policy 一致，
    # 强制策略跳出低功率吸引子。False 时行为与现状完全一致（零干预）。
    use_delivery_safety_layer: bool = False  # DSL 已验证无效，关闭
    dsl_floor_base: float = 0.35        # p_total 地板基值(约对齐 SDMA 功率水平)
    dsl_queue_gain: float = 0.4         # 地板随积压抬升
    dsl_deficit_gain: float = 0.3       # 地板随欠交付抬升(融合方案2赤字信号)
    dsl_floor_cap: float = 0.9
    dsl_d_base: float = 0.6             # d_ratio 地板基值
    dsl_d_queue_gain: float = 0.4

    # PLPI: Policy-Level Power Incentive — 直达策略梯度的功率引导
    # delivery < threshold 时在 PPO loss 中直接推动功率输出，绕过噪声 throughput 梯度
    # pi_power_coef 控制推力强度（loss 级，不需 reward_scale）；pi_power_decay 每 episode 衰减人工引导
    use_power_incentive: bool = False   # PLPI 已验证无效，关闭
    pi_power_coef: float = 1.0
    pi_power_decay: float = 0.999

    # SPLC: Self-Paced Load Curriculum（任务分布层干预）
    # main.py 每次 env.reset() 前按窗口化 running delivery 动态缩放
    # env.load_ratio_min/max（init_scale→final_scale 同伦延拓），delivery 稳定达标
    # 才升难度，使策略在平稳可行奖励下学会功率→吞吐→EE 因果链、跳出低功率吸引子。
    # False 时整段逻辑跳过，env.load_ratio_min/max 保持配置原值，行为与现状一致。
    use_load_curriculum: bool = False    # SPLC 已验证无效，关闭
    lc_init_scale: float = 0.5          # 初始负载缩放：ratio 乘以此系数
    lc_final_scale: float = 1.0         # 终态缩放（回到配置的 load_ratio_min/max 原值）
    lc_delivery_target: float = 0.85    # 窗口平均 delivery 达此值才升难度
    lc_step: float = 0.05               # 每次升难度的 scale 增量
    lc_window: int = 20                 # 用于判定的最近 episode 数窗口
    lc_patience: int = 3                # 连续满足次数后才升档，防抖
    lc_min_episodes_per_level: int = 30  # 每档最少停留 episode 数，保证收敛

    # HEK-Lite: Heuristic Expert Kickstart（初始化层干预，结构性创新）
    # 队列感知规则专家在真实 env 跑少量 episode 生成演示，BC 预训练把 actor
    # 参数送到高功率/高交付盆地（逃离低功率吸引子），再以衰减锚定损失微调；
    # 预热期临时关闭 load_curriculum 防止把策略拽回。区别于全部优化期
    # （reward/advantage/action/curriculum）方向，本方法只改参数初始化。
    # hek_enabled=False 时行为与现状完全一致（零干预）。
    hek_enabled: bool = False           # HEK 已验证无效，关闭
    hek_expert_episodes: int = 15       # 专家演示 episode 数
    hek_bc_epochs: int = 150            # BC 预训练轮数
    hek_bc_lr: float = 3e-4             # BC 独立 Adam 学习率
    hek_anchor_updates: int = 40        # 锚定损失生效的 PPO update 数（预热期长度）
    hek_anchor_coef: float = 0.2        # 锚定损失初始系数
    hek_anchor_decay: float = 0.95      # 锚定系数逐 update 衰减
    hek_base_power: float = 0.5         # 专家目标物理功率占比基值

    # CRPO-S: episode级原始目标切换。桶不可行时整段episode用排水盈余奖励(纯可行性目标)，
    # 桶可行时保持EE+Lagrangian现状；迟滞带防目标抖振。use_crpo_switch=False时行为与现状完全一致。
    use_crpo_switch: bool = True
    crpo_hysteresis: float = 0.03   # 切回EE模式需 delivery > threshold + 此值
    crpo_feas_scale: float = 1.0    # 可行性奖励缩放(有界[-1,1]后再乘)

    # ---- 词典序梯度投影 PPO (Lexicographic Gradient Projection) ----
    # 将 EE 策略梯度投影到 delivery 约束梯度的零空间，几何硬保证 EE 优化不损害 delivery。
    # 需配合 RCPO cost-critic 使用（main.py 会自动启用）。
    use_lexicographic_projection: bool = True
    lex_proj_lambda_scale: float = 0.2   # 对应 RCPO lambda_scale
    lex_proj_min_bucket_samples: int = 4  # 每桶最少样本数才做投影

    # ---- C3PO: 中央路径惩罚替代 PID-Lagrangian ----
    # C3PO (Milosevic et al., ICML 2025) 用移动预算 ReLU 惩罚从可行侧单调逼近约束面，
    # 无对偶变量，避免 PID-Lagrangian 在约束边界的极限环振荡。
    c3po_enabled: bool = False           # True 时冻结 Lagrangian delivery 通道
    c3po_nu_init: float = 0.05           # 惩罚系数初值
    c3po_nu_final: float = 0.5           # 惩罚系数终值（随训练进度线性增长）
    c3po_kappa: float = 1.0              # 移动预算 = kappa * 达标裕量
    c3po_margin_floor: float = 0.02      # 预算下限，防止 early stage 预算为 0 无梯度
    c3po_cost_coef: float = 0.5          # cost critic value loss 权重
    c3po_gamma: float = 0.99             # cost 折扣
    c3po_gae_lambda: float = 0.95        # cost GAE lambda
    c3po_n_points: int = 7               # 与 SDMA 评估点一一对应

    # ---- SFL 可学习性增强 PLR 采样 ----
    # 在 RolloutBuffer 采样时按 SNR 桶加权：weight ∝ deficit^a · [p(1-p)]^b，
    # 聚焦成功率中段、避开已饱和或学不动的桶。
    use_sfl_sampling: bool = True
    sfl_deficit_exponent: float = 1.0
    sfl_learnability_exponent: float = 1.0
    sfl_uniform_mix: float = 0.1         # 混合保底（避免权重坍缩）
    sfl_success_ema_decay: float = 0.99  # 每桶 delivery 成功率 EMA 衰减

@dataclass
class PPOConfig:
    """PPO训练超参"""
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    # log_std 单独高 lr（实验中 entropy 表明 log_std 收敛太慢，单开学习率加速探索强度学习）
    lr_log_std: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.008
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    n_epochs: int = 8
    batch_size: int = 64
    rollout_steps: int = 2048
    total_episodes: int = 8000

    # KL 早停与 LR 衰减
    target_kl: float = 0.02
    lr_decay_final_ratio: float = 0.2
    kl_stop_lr_decay: float = 0.9
    kl_stop_streak_threshold: int = 3

    hidden_dim: int = 256
    n_layers: int = 3

    action_dim: int = 96  # 4*K=4*24 (Task#12: K=24)

    # P1: Self-Attention Worker（替代 MLP 解决 LN 稀释 SNR/ε 信号问题）
    use_attention: bool = True
    worker_d_user: int = 13            # per-user 特征维度（v14 QoS: 12→13, 新增 lambda_load + priority）
    worker_attn_hidden: int = 128      # attention token 维度
    worker_attn_heads: int = 4         # MultiHeadAttention 头数
    worker_attn_layers: int = 2        # attention 层数

    # Mean-Field MAPPO (MVP)
    use_mean_field_critic: bool = True     # critic 加 QoS 类均值场统计量偏置
    mean_field_hidden_dim: int = 64
    use_per_user_advantage: bool = True    # 优势中加入每用户 delivery 违约项
    per_user_adv_weight: float = 1.0

    log_interval: int = 20
    save_interval: int = 100
    eval_interval: int = 500

    # SNR 课程学习（训练中动态调整 env.snr_min_db / env.snr_max_db）
    # curriculum_stages: List[dict] with keys: start_episode, end_episode, snr_min_db, snr_max_db
    curriculum_enabled: bool = False
    curriculum_stages: list = None

    # ---- CuRLTraC: 性能触发式难度退火课程 ----
    # 从易达标区（load_ratio_max=0.8、snr_min=5dB）起步，按近50ep的delivery窗口
    # 逐步退火到评估配置（load_ratio_max=1.3、snr_min=-5dB）。性能触发保证推进以
    # delivery达标为前提；回退滞回防止分布突变导致策略崩溃。
    use_curriculum: bool = False              # 启用 CuRLTraC 课程调度
    curriculum_n_stages: int = 5             # 难度档数（含起始与评估配置）
    curriculum_beta: float = 0.5             # 窗口平均 delivery ≥ 此值才推进
    curriculum_fallback_beta: float = 0.3    # 窗口平均 delivery < 此值连续计数回退
    curriculum_window: int = 50              # 判定用的最近 episode 窗口长度
    curriculum_patience: int = 20            # 连续低于 fallback_beta 的 episode 数才回退
    curriculum_snr_min_start: float = 5.0    # 起始最低 SNR (dB)
    curriculum_snr_min_end: float = -5.0     # 最终最低 SNR (dB)
    curriculum_load_ratio_max_start: float = 0.8   # 起始最大负载比
    curriculum_load_ratio_max_end: float = 1.3     # 最终最大负载比
    curriculum_min_stage_episodes: int = 50  # 每档最少停留 episode 数
    curriculum_freeze_episodes: int = 500    # 到达最终配置后至少再训练 episode 数
    curriculum_apply_snr_max: bool = True    # 是否同时把 snr_max 锁定到配置值

    # ---- CuRLTraC-ALP: 定向硬例挖掘（ALP-GMM 式任务空间主动重采样）----
    # 维护逐 SNR 桶 delivery 缺口 shortfall，按 p_b ∝ base + w_hard·shortfall_b
    # 重加权 episode 的 SNR 采样分布，把训练曝光导向未达标桶。
    hard_mining_enable: bool = True          # 是否启用定向硬例重采样
    hard_mining_w_hard: float = 2.0          # 缺口加权强度（legacy，point_w_hard 优先）
    hard_mining_ema_decay: float = 0.99      # delivery EMA 慢衰减（防震荡）
    hard_mining_floor_frac: float = 0.5      # 均匀地板 = floor_frac / n_buckets（≥1/(2K)）
    hard_mining_target: float = 1.0          # 逐桶 SDMA 目标倍数（1.0 = 对齐 SDMA 逐点 delivery）

    # ---- 逐评估点细粒度硬例挖掘 + 条件熵激励 ----
    hm_granularity: str = "point"            # "point"=7个SNR评估点, "bucket"=legacy 4桶
    point_w_hard: float = 2.0                # 逐点 shortfall 加权强度（本轮不上调）
    streak_mult: float = 1.5                 # 连续未达标轮数倍增基数（先保原值复现峰值）
    streak_cap: float = 4.0                  #  streak 倍增上限（封顶）
    fail_entropy_scale: float = 1.5          # 未达标点 episode 熵奖励放大基数 ent_base
    point_base_floor: float = 0.5            # 逐点均匀地板系数（实际=本值/n_points）

    # ---- 性能挂钩定向熵激励 ----
    entropy_performance_linked: bool = True  # 未达标点熵系数按 shortfall 动态调节
    entropy_link_coef: float = 1.0           # c≈1，shortfall 最大时熵放大 (1+c) 倍

    # 桶专家多头 Actor（按 SNR 硬路由、冻结共享主干、仅训练各专家头）
    use_bucket_experts: bool = True
    n_bucket_experts: int = 4
    bucket_snr_min: float = -5.0
    bucket_snr_max: float = 35.0
    lr_bucket_expert: float = 1e-4       # 专家头学习率（通常低于 actor lr）

    # 续训覆盖（从 iter-7 续训时降低 actor lr、收紧 clip 信任域）
    resume_lr_actor_scale: float = 1.0   # 续训时 actor lr *= 该值（建议 0.2~0.33）
    resume_clip_ratio: float = -1.0      # >0 时覆盖 clip_ratio（建议 0.05~0.1）


@dataclass
class HierarchicalConfig:
    """分层架构参数 (慢时间尺度 Meta-Controller: Self-Attention PPO)

    Meta-Controller 输出：分裂决策 + 用户分组（每用户分配到 G 个组之一）。
    """
    T_s: int = 20                    # 慢尺度决策间隔 (slots)
    meta_lr: float = 5e-4            # Meta-Controller 学习率
    meta_gamma: float = 0.95         # Meta-Controller 折扣因子
    meta_hidden_dim: int = 64        # Meta-Controller 隐藏层维度
    meta_clip_ratio: float = 0.2     # Meta PPO 裁剪范围
    meta_entropy_coef: float = 0.05  # Meta 熵系数 (高探索，慢尺度决策少)
    meta_n_heads: int = 4            # Self-Attention 头数
    meta_d_user: int = 12            # Per-user 特征维度 (v14 QoS: 10→12, 新增 lambda_load + priority)
    meta_n_epochs: int = 4           # Meta PPO 每次更新迭代数
    meta_update_threshold: int = 30  # 积累多少 meta transitions 后更新
    meta_gae_lambda: float = 0.95    # Meta GAE lambda
    meta_max_grad_norm: float = 0.5  # Meta 梯度裁剪

    # 用户分组
    n_groups: int = 4                    # 最大分组数 G (1=退化为标准RSMA, K=退化为NOMA)
    meta_group_entropy_coef: float = 0.12  # 分组策略熵系数

    # Task#38: Group-level α 架构
    # True: Meta 输出 n_groups 维连续 α (Gaussian, sigmoid 映射到 [α_min, α_max])，
    #       per-user α = group_alphas[group_ids[k]]，搜索空间从 ~13824 降至 ~28。
    # False: 原 per-user 二值分裂决策 (Bernoulli, K 维)。
    group_level_alpha: bool = True
    group_alpha_min: float = 0.1         # 映射后 α 下限
    group_alpha_max: float = 0.95        # 映射后 α 上限
    group_alpha_log_std_init: float = -0.5  # Gaussian std 初值 (exp(-0.5)≈0.61)

    # Group head 温度退火（仅作用于分组 head，不影响分裂 head）
    meta_group_tau_init: float = 2.0     # 初始温度（高探索）
    meta_group_tau_min: float = 0.5      # 最低温度（后期决策尖锐）

    # 分组平衡正则（0 = 关闭）
    meta_balance_coef: float = 0.1

    # Meta reward 缩放（防止 meta value loss 爆炸；原始 reward 量级 ±50 → ±1）
    meta_reward_scale: float = 0.02

    # Task#36: SNR 条件 α 引导奖励
    # 在 Meta reward 中加入 -alpha_guidance_scale * (alpha_mean - target_alpha)**2 软引导项，
    # 鼓励 Worker 输出的平均 split ratio α 随 SNR 变化（target = clip(0.9 - snr_db/100, 0.5, 0.95)）。
    # 仅 warmup 结束后激活，避免影响早期纯 Worker 学习信号。
    alpha_guidance_enabled: bool = True
    alpha_guidance_scale: float = 0.3
    alpha_guidance_warmup: int = 300       # episodes
    alpha_guidance_target_intercept: float = 0.9    # SNR=0dB 处的目标 α
    alpha_guidance_target_slope: float = 0.01       # 单位 dB 上的下降速率（0.9 - snr_db*0.01）
    alpha_guidance_target_min: float = 0.5
    alpha_guidance_target_max: float = 0.95

    # Plackett-Luce 解码顺序 (Worker侧)
    pl_temperature: float = 1.0          # Plackett-Luce 温度参数

@dataclass
class AblationConfig:
    """Task#11 消融实验配置：通过标志控制各组件是否启用。

    变体设计：
      - disable_split:     强制 split_mask=0（纯 SDMA，无 RSMA 分裂）
      - disable_group:     强制 group_ids=0（所有用户同组）
      - disable_sic_order: 固定解码顺序为信道增益降序（忽略 Worker 的 PL 输出）
      - disable_meta:      完全禁用 Meta-Controller（含 disable_split + disable_group）
    """
    disable_split: bool = False
    disable_group: bool = False
    disable_sic_order: bool = False
    disable_meta: bool = False


@dataclass
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    lyapunov: LyapunovConfig = field(default_factory=LyapunovConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)

    seed: int = 42
    device: str = "cpu"
    exp_name: str = "robust_snr"
