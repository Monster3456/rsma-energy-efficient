"""训练启动脚本，支持命令行参数覆盖（含 Task#11 消融标志）"""
import argparse
import sys
sys.path.insert(0, '/Users/libo/PycharmProjects/rsma-energy-efficient')

from src.training.main import train
from src.config import Config


def _validate_hm_point_targets(vals: list[float]) -> tuple[float, ...]:
    """--hm-point-targets 必须恰好 7 个值（0..30dB 逐点 delivery 目标）。"""
    if len(vals) != 7:
        raise ValueError(
            "--hm-point-targets 必须恰好 7 个值（0..30dB 逐点 delivery 目标），"
            f"got {len(vals)}: {vals}"
        )
    return tuple(vals)


def _validate_eval_guard_snr_points(vals: list[float]) -> tuple[float, ...]:
    """--eval-guard-snr-points 必须是参考表 _GUARD_REF 的网格点。"""
    supported = {0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0}
    bad = [p for p in vals if p not in supported]
    if bad:
        raise ValueError(
            f"--eval-guard-snr-points 含非参考网格点 {bad}；仅支持 "
            f"{sorted(supported)}（守卫参考表 _GUARD_REF 按这些点索引）"
        )
    return tuple(vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSMA Energy-Efficient PPO Training")
    parser.add_argument('--V', type=float, default=1.0, help='Lyapunov V parameter')
    parser.add_argument('--reward_dpp', action='store_true',
                        help='classical drift-plus-penalty reward (neural-DPP comparator)')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--episodes', type=int, default=5000, help='Total training episodes')
    parser.add_argument('--csi_mode', choices=['perfect', 'imperfect'], default='imperfect',
                        help='CSI mode')
    parser.add_argument('--exp_name', type=str, default='rsma_ee', help='Experiment name')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--rollout_steps', type=int, default=2048,
                        help='Steps per rollout')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--obs_history_window', type=int, default=None,
                        help='时序历史观测窗口长度 H（覆盖 system.obs_history_window，默认 0=关闭）')
    parser.add_argument('--obs_pressure_vector', action='store_true',
                        help='D3: 观测追加 7 维逐点压力向量（HM EMA 短差归一化）')
    parser.add_argument('--obs_prune_constant_blocks', action='store_true',
                        help='A-prune：移除观测中固定 SNR 点内零方差的 3 个常数原生块（arrivals/λ桶/Dinkelbach-η），per-user 13→10 块（可解释性取证驱动）')

    # Task#11 消融实验标志
    parser.add_argument('--disable-split', action='store_true',
                        help='强制 split_mask=0（No-Split 变体）')
    parser.add_argument('--disable-group', action='store_true',
                        help='强制 group_ids=0（No-Group 变体）')
    parser.add_argument('--disable-sic-order', action='store_true',
                        help='固定解码顺序为信道增益降序（No-SIC-Order 变体）')
    parser.add_argument('--disable-meta', action='store_true',
                        help='完全禁用 Meta-Controller（Worker-Only 变体）')

    # Task#17 PPO-Lagrangian 硬约束
    parser.add_argument('--no-lagrangian', action='store_true',
                        help='禁用 PPO-Lagrangian Delivery 约束（默认启用）')
    parser.add_argument('--delivery-threshold', type=float, default=None,
                        help='Delivery 约束阈值（默认 0.95）')
    parser.add_argument('--lagrangian-lr', type=float, default=None,
                        help='对偶乘子 λ 的学习率（默认 0.05）')
    parser.add_argument('--lagrangian-init', type=float, default=None,
                        help='对偶乘子 λ 的初值（默认 5.0）')
    parser.add_argument('--lagrangian-max', type=float, default=None,
                        help='对偶乘子 λ 的上限（默认 50.0）')
    parser.add_argument('--lagrangian-warmup', type=int, default=None,
                        help='Lagrangian warmup episodes（默认 500）')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定 checkpoint 恢复训练 (e.g. logs/full_lagrangian/checkpoints/ckpt_500.pt)')
    parser.add_argument('--save_interval', type=int, default=None,
                        help='checkpoint 保存间隔（更新数，覆盖 config.ppo.save_interval；突刺/渐进用）')
    parser.add_argument('--resume-lambda', type=float, default=None,
                        help='恢复训练时手动设置 Lagrangian λ 初值 (来源：训练日志)')
    parser.add_argument('--resume-episode-count', type=int, default=None,
                        help='恢复训练时手动设置 Lagrangian episode_count (用于跳过 warmup)')

    # Task#33: Partial resume — 仅加载 Worker (actor_critic) 权重，Meta 重新初始化
    parser.add_argument('--partial_resume', type=str, default=None,
                        help='仅加载 Worker 权重 (用于 d_user 维度变化时跨版本迁移)')

    # Meta-Controller 超参覆盖（Task#33 v5: 让 Meta 学得更快）
    parser.add_argument('--meta_lr', type=float, default=None,
                        help='Meta-Controller 学习率 (覆盖 hierarchical.meta_lr)')
    parser.add_argument('--meta_hidden_dim', type=int, default=None,
                        help='Meta-Controller 隐藏层维度 (覆盖 hierarchical.meta_hidden_dim)')
    parser.add_argument('--T_s', type=int, default=None,
                        help='慢尺度决策间隔 (覆盖 hierarchical.T_s)')

    # Task#36: SNR 条件 α 引导奖励
    parser.add_argument('--alpha_guidance_scale', type=float, default=None,
                        help='α 引导奖励权重（覆盖 hierarchical.alpha_guidance_scale）')
    parser.add_argument('--alpha_guidance_warmup', type=int, default=None,
                        help='α 引导奖励 warmup episode 数（覆盖 hierarchical.alpha_guidance_warmup）')
    parser.add_argument('--no-alpha-guidance', action='store_true',
                        help='禁用 SNR 条件 α 引导奖励（默认启用）')

    # SNR 课程学习
    parser.add_argument('--curriculum', action='store_true',
                        help='启用 SNR 课程学习（训练中动态调整 env.snr_min_db / env.snr_max_db）')
    parser.add_argument('--curriculum-stages-json', type=str, default=None,
                        help='课程 stages JSON字符串，格式: [{"start_episode":..,"end_episode":..,"snr_min_db":..,"snr_max_db":..}, ...]')

    # CuRLTraC: 性能触发式难度退火课程
    parser.add_argument('--curltrac', action='store_true',
                        help='启用 CuRLTraC 课程学习（从易达标区性能触发退火到评估配置）')

    # CuRLTraC-ALP: 定向硬例挖掘（ALP-GMM 式任务空间主动重采样）
    parser.add_argument('--hard-mining', action='store_true',
                        help='启用定向硬例挖掘（默认已启用）')
    parser.add_argument('--no-hard-mining', action='store_true',
                        help='禁用定向硬例挖掘')
    parser.add_argument('--hard-mining-w-hard', type=float, default=None,
                        help='缺口加权强度 w_hard（覆盖 ppo.hard_mining_w_hard）')
    parser.add_argument('--hard-mining-ema-decay', type=float, default=None,
                        help='delivery EMA 慢衰减系数（覆盖 ppo.hard_mining_ema_decay）')
    parser.add_argument('--hard-mining-floor-frac', type=float, default=None,
                        help='均匀地板系数 floor_frac（覆盖 ppo.hard_mining_floor_frac）')
    parser.add_argument('--hard-mining-target', type=float, default=None,
                        help='逐桶 SDMA 目标倍数（覆盖 ppo.hard_mining_target，默认 1.0）')
    parser.add_argument('--hm-point-targets', type=str, default=None,
                        help='M5.3: 逗号分隔 7 个逐点 delivery 目标（0/5/10/15/20/25/30dB），'
                             '覆盖硬例挖掘默认 SDMA 目标；不传=默认')

    # Task#38: Group-level α 架构
    parser.add_argument('--group_level_alpha', type=lambda v: v.lower() in ('1', 'true', 'yes'),
                        default=None,
                        help='Meta 输出 per-group α（默认 True），False 退回逐用户二值分裂决策')
    parser.add_argument('--group_alpha_min', type=float, default=None,
                        help='Group-level α 下限（默认 0.1）')
    parser.add_argument('--group_alpha_max', type=float, default=None,
                        help='Group-level α 上限（默认 0.95）')

    # optimize-loop 超参通道: 奖励/Lyapunov 数值旋钮
    parser.add_argument('--w_overflow', type=float, default=None,
                        help='溢出惩罚权重（覆盖 reward.w_overflow）')
    parser.add_argument('--w_outage', type=float, default=None,
                        help='中断惩罚权重（覆盖 reward.w_outage）')
    parser.add_argument('--w_power_exceed', type=float, default=None,
                        help='超功率惩罚权重（覆盖 reward.w_power_exceed）')
    parser.add_argument('--V_min', type=float, default=None,
                        help='自适应 V 下限（覆盖 lyapunov.V_min）')
    parser.add_argument('--V_max', type=float, default=None,
                        help='自适应 V 上限（覆盖 lyapunov.V_max）')
    parser.add_argument('--V_exponent', type=float, default=None,
                        help='自适应 V 指数（覆盖 lyapunov.V_exponent）')
    parser.add_argument('--blend_alpha', type=float, default=None,
                        help='队列感知混合权重指数（覆盖 lyapunov.blend_alpha）')
    parser.add_argument('--smoke', action='store_true',
                        help='冒烟测试模式：极小配置快速验证可运行性')

    # 物理层参数覆盖（天线扩容实验）
    parser.add_argument('--n_r', type=int, default=None,
                        help='基站接收天线数（覆盖 system.N_r，默认 16）；注意改动会同步改变 SDMA 基线，评估须用同口径')

    # 固定外生流量口径（文献惯例：到达不随容量自缩放）
    parser.add_argument('--fixed_lambda', type=float, default=None,
                        help='固定外生流量口径：关闭 adaptive_lambda 并把 lambda_rate 钉为该值 '
                             '（到达为外生泊松过程，不随信道容量自缩放；文献对照与取值依据见 '
                             '.trellis/tasks/08-24-temporal-worker/experiment-results.md 优化背景审计节）；'
                             '与默认自适应口径（λ∝容量）不可横比，评估须加 '
                             'evaluate_snr_sweep_temporal.py --fixed_lambda 同值')

    # reward_v2: 结构性奖励修复（Run 8，默认全关 = 官方口径）
    parser.add_argument('--reward_v2_per_user_weight', action='store_true',
                        help='Dinkelbach EE 线性项分子改逐用户加权 Σw_k·D_k（w=clip(Q/Q̄,0.5,2)），'
                             '修正深过载速率均等化分配偏向（corr(R,Q)=-0.2 诊断）')
    parser.add_argument('--reward_v2_fixed_q', type=float, default=None,
                        help='冻结 Dinkelbach q* 为该固定参考值（>0；覆盖 EMA/外层更新），'
                             '给 dkl=Σw·D−q_ref·P_eff 真实可越门槛，激活省电压力')

    # Mean-Field MAPPO toggles
    parser.add_argument('--no-softmax-power-share', action='store_true',
                        help='禁用 softmax 功率份额（恢复逐用户独立 p_total）')
    parser.add_argument('--no-mean-field-critic', action='store_true',
                        help='禁用 critic QoS 类均值场偏置')
    parser.add_argument('--no-per-user-advantage', action='store_true',
                        help='禁用 per-user delivery 违约优势增强')
    parser.add_argument('--per-user-adv-weight', type=float, default=None,
                        help='per-user 违约项权重（覆盖 ppo.per_user_adv_weight）')

    # 桶专家多头 Actor
    parser.add_argument('--no-bucket-experts', action='store_true',
                        help='禁用 SNR 桶专家多头 Actor')
    parser.add_argument('--n-bucket-experts', type=int, default=None,
                        help='专家头数量（覆盖 ppo.n_bucket_experts）')
    parser.add_argument('--fix-drain', action='store_true',
                        help='P2a：Worker d 块固定 1.0（不学习 drain 头；可解释性取证驱动，默认关）')
    parser.add_argument('--fix-order', action='store_true',
                        help='P2b：解码序固定信道增益降序（不学习 PL 头；默认关）')
    parser.add_argument('--meta-zeta', action='store_true',
                        help='Arm C：目标条件化，Meta 输出慢尺度交付目标 ζ，'
                             'Worker 观测加入 ζ_norm、DVQ 目标改为 ζ（默认关）')
    parser.add_argument('--simplified-action', action='store_true',
                        help='Arm A：精简动作模式，Worker 只输出 (p, 解码序)，'
                             'α 由 Meta 给、d_k≡1，动作 4K→2K（默认关=官方口径）')
    parser.add_argument('--n_groups', type=int, default=None,
                        help='P2c：覆盖 hierarchical.n_groups（评估 G=1 分组坍缩裁剪；默认 4）')
    parser.add_argument('--lr-bucket-expert', type=float, default=None,
                        help='专家头学习率（覆盖 ppo.lr_bucket_expert）')

    # Backpressure 先验（actor 功率头结构偏置）
    parser.add_argument('--backpressure_beta', type=float, default=None,
                        help='Worker actor 功率 logit += beta*(Q_k/Q_mean) 的先验强度'
                             '（max-weight 结构先验；默认 0=关闭。注册为 buffer 随 checkpoint '
                             '传输，评估侧自动生效，无需评估脚本加参数）')
    parser.add_argument('--backpressure_mode',
                        choices=['queue', 'queue_channel', 'mw_residual', 'mw_gated'],
                        default=None,
                        help='先验形式：queue = beta*(Q_k/Q̄-1)（纯队列加权）；'
                             'queue_channel = beta*(Q_k*ĥ_k/mean(Q·ĥ)-1)（Q×信道乘积，'
                             'max-weight 完整形式）；mw_residual = 闭式 max-weight 残差基座'
                             '（p_raw = atanh(clip(p_mw,0.01,0.95)) + head 残差，p_mw 对齐 '
                             'sdma_baseline 的 p∝Q，无视 beta；初始化即 SDMA 功率平价）；'
                             'mw_gated = 同基座乘可学习负载门 g(s)（Run 13：深过载 g→1 '
                             '保平价、欠载 g→0 释放 head 深度省电；门输入 mean/max 队列 + '
                             'mean 信道，初始 g≈0.98）。'
                             '默认 queue。mode 随 checkpoint 传输')

    # Solver-Expert Iteration（初始化层干预，ExIt/AlphaZero 模式）
    parser.add_argument('--solver-expert', action='store_true',
                        help='启用单槽 DGLCR surrogate 精修专家热启动：真实 env 逐时隙求解'
                             '收集演示 → BC 联合预训练连续头+PL 排序头 → 衰减锚定微调')
    parser.add_argument('--solver-expert-episodes', type=int, default=None,
                        help='专家演示 episode 数（默认 6；每 slot 求解约 0.3s）')
    parser.add_argument('--solver-expert-steps', type=int, default=None,
                        help='每 slot Adam 精修步数（默认 25）')
    parser.add_argument('--solver-expert-lr', type=float, default=None,
                        help='专家 Adam 学习率（默认 0.1）')
    parser.add_argument('--solver-expert-bc-epochs', type=int, default=None,
                        help='BC 预训练轮数（默认 150）')
    parser.add_argument('--solver-expert-anchor-updates', type=int, default=None,
                        help='锚定损失生效的 PPO update 数（默认 40）')
    parser.add_argument('--solver-expert-anchor-decay', type=float, default=None,
                        help='锚定系数逐 update 衰减（默认 0.95；M5 持久锚定用 0.999）')
    parser.add_argument('--solver-expert-anchor-coef', type=float, default=None,
                        help='锚定初始系数（默认 0.2）')
    parser.add_argument('--solver-expert-relabel', action='store_true',
                        help='M5-R1(ExIt-2): 首轮 BC 后 on-policy relabel 二次收集'
                             '（学生走轨迹+教师打标）+ 聚合二次 BC')
    parser.add_argument('--solver-expert-no-mix-original', action='store_true',
                        help='relabel 时不聚合原始教师轨迹演示（纯 relabel 演示）')

    # M5 SFTP: 安全微调协议（全默认关 = 官方口径零变化）
    parser.add_argument('--clip_ratio', type=float, default=None,
                        help='覆盖 PPO clip 范围（默认 0.2；微调建议 0.1）')
    parser.add_argument('--lr_log_std', type=float, default=None,
                        help='覆盖 log_std 学习率（默认 1e-3；微调建议 1e-4 防探索坍缩）')
    parser.add_argument('--actor-warmup-updates', type=int, default=None,
                        help='M5-F2: 前 N 个 update 只训 critic（actor+log_std 梯度置零）')
    parser.add_argument('--meta-freeze-episodes', type=int, default=None,
                        help='M5-F4: 前 N episode Meta 不做梯度更新（采样照常）')
    parser.add_argument('--kl-stop-lr-decay', type=float, default=None,
                        help='连续 KL 早停后的 lr 衰减乘子（默认 0.9；1.0=关闭该衰减，'
                             'M5.1 用于解除微调期 actor lr 饿死）')
    parser.add_argument('--lr-freeze-update', type=int, default=None,
                        help='M5.4: 达到该 update 数后 lr 钉死在 lr_decay_final_ratio×base'
                             '（显式冻结点；不传=关，沿用涌现式触底）')
    parser.add_argument('--eval-guard-early-stop', action='store_true',
                        help='P1 guard-plateau 早停：guard 连续 patience 次评估未刷新 best 且已训够 min_episodes 时提前终止（默认关，可省 35-40% 训练预算）')
    parser.add_argument('--eval-guard-early-stop-patience', type=int, default=None,
                        help='早停 patience（连续未刷新次数，默认 2）')
    parser.add_argument('--eval-guard-early-stop-min-episodes', type=int, default=None,
                        help='早停最小训练集数（默认 3000）')
    parser.add_argument('--eval-guard', action='store_true',
                        help='M5-F5: 训练中评估守卫（{0,5,30}dB mini-eval + 劣化回滚/lr减半）')
    parser.add_argument('--eval-guard-every', type=int, default=None,
                        help='守卫 mini-eval 间隔 episode（默认 400）')
    parser.add_argument('--eval-guard-episodes', type=int, default=None,
                        help='守卫每 SNR 点评估 episode 数（默认 4）')
    parser.add_argument('--eval-guard-margin', type=float, default=None,
                        help='守卫 avg_deliv 低于 bc 参考的容忍裕量（默认 0.02）')
    parser.add_argument('--eval-guard-max-halvings', type=int, default=None,
                        help='守卫最多回滚+lr减半次数（默认 3，超过冻结 actor）')
    # ---- M5.2: 显式信任域 + 守卫判据升级 ----
    parser.add_argument('--kl-bc-coef', type=float, default=None,
                        help='M5.2-F6: KL(π_θ‖π_BC) 信任域强度（0/不传=关；'
                             'BC 参考快照在 BC 预训练后自动注入）')
    parser.add_argument('--eval-guard-v2', action='store_true',
                        help='M5.2-F5b: 守卫 ok 判据 = avg_deliv 门 AND 逐点门全过 '
                             'AND avg_queue ≤ bc 参考×ratio（旧版只看 avg_deliv）')
    parser.add_argument('--eval-guard-queue-ratio', type=float, default=None,
                        help='M5.2-F5b(v2): avg_queue 对照 bc 参考均值的容差比（默认 1.1）')
    parser.add_argument('--eval-guard-snr-points', type=str, default=None,
                        help='M5.2-F5b: 逗号分隔守卫 SNR 点集（如 0,5,15,20,25,30；'
                             '不传=沿用旧三点 0,5,30）')

    args = parser.parse_args()

    config = Config()
    config.lyapunov.V = args.V
    config.ppo.lr_actor = args.lr_actor
    config.ppo.lr_critic = args.lr_critic
    config.seed = args.seed
    config.ppo.total_episodes = args.episodes
    config.system.csi_mode = args.csi_mode
    config.exp_name = args.exp_name
    config.device = args.device
    config.ppo.rollout_steps = args.rollout_steps
    config.ppo.batch_size = args.batch_size
    if args.obs_history_window is not None:
        config.system.obs_history_window = args.obs_history_window
    config.system.obs_prune_constant_blocks = args.obs_prune_constant_blocks
    config.system.obs_pressure_vector = args.obs_pressure_vector

    # 注入消融标志
    config.ablation.disable_split = args.disable_split
    config.ablation.disable_group = args.disable_group
    config.ablation.disable_sic_order = args.disable_sic_order
    config.ablation.disable_meta = args.disable_meta

    # Task#17 注入 Lagrangian 配置
    if args.no_lagrangian:
        config.constraint.use_lagrangian = False
    if args.delivery_threshold is not None:
        config.constraint.delivery_threshold = args.delivery_threshold
    if args.lagrangian_lr is not None:
        config.constraint.lagrangian_lr = args.lagrangian_lr
    if args.lagrangian_init is not None:
        config.constraint.lagrangian_init = args.lagrangian_init
    if args.lagrangian_max is not None:
        config.constraint.lagrangian_max = args.lagrangian_max
    if args.lagrangian_warmup is not None:
        config.constraint.lagrangian_warmup = args.lagrangian_warmup

    # Resume
    config.resume_ckpt = args.resume
    config.resume_lambda = args.resume_lambda
    config.resume_episode_count = args.resume_episode_count
    if args.save_interval is not None:
        config.ppo.save_interval = args.save_interval
    if args.fix_drain:
        config.ppo.worker_fix_drain = True
    if args.fix_order:
        config.ppo.worker_fix_order = True
    if args.simplified_action:
        config.ppo.worker_simplified_action = True
    if args.meta_zeta:
        config.hierarchical.meta_zeta_enabled = True
    if args.n_groups is not None:
        config.hierarchical.n_groups = int(args.n_groups)

    # Task#33: Partial resume
    config.partial_resume_ckpt = args.partial_resume

    # Meta hyperparam overrides
    if args.meta_lr is not None:
        config.hierarchical.meta_lr = args.meta_lr
    if args.meta_hidden_dim is not None:
        config.hierarchical.meta_hidden_dim = args.meta_hidden_dim
    if args.T_s is not None:
        config.hierarchical.T_s = args.T_s

    # Task#36: α 引导奖励覆盖
    if args.alpha_guidance_scale is not None:
        config.hierarchical.alpha_guidance_scale = args.alpha_guidance_scale
    if args.alpha_guidance_warmup is not None:
        config.hierarchical.alpha_guidance_warmup = args.alpha_guidance_warmup
    if args.no_alpha_guidance:
        config.hierarchical.alpha_guidance_enabled = False

    # SNR 课程学习
    if args.curriculum:
        config.ppo.curriculum_enabled = True
    if args.curriculum_stages_json is not None:
        import json
        config.ppo.curriculum_stages = json.loads(args.curriculum_stages_json)

    # CuRLTraC: 性能触发式难度退火课程
    if args.curltrac:
        config.ppo.use_curriculum = True

    # CuRLTraC-ALP: 定向硬例挖掘开关覆盖
    if args.hard_mining:
        config.ppo.hard_mining_enable = True
    if args.no_hard_mining:
        config.ppo.hard_mining_enable = False
    if args.hard_mining_w_hard is not None:
        config.ppo.hard_mining_w_hard = args.hard_mining_w_hard
    if args.hard_mining_ema_decay is not None:
        config.ppo.hard_mining_ema_decay = args.hard_mining_ema_decay
    if args.hard_mining_floor_frac is not None:
        config.ppo.hard_mining_floor_frac = args.hard_mining_floor_frac
    if args.hard_mining_target is not None:
        config.ppo.hard_mining_target = args.hard_mining_target
    if args.hm_point_targets:
        try:
            config.ppo.hm_point_targets = _validate_hm_point_targets(
                [float(s) for s in args.hm_point_targets.split(",") if s.strip()])
        except ValueError as e:
            parser.error(str(e))

    # Task#38: Group-level α 架构参数注入
    if args.group_level_alpha is not None:
        config.hierarchical.group_level_alpha = args.group_level_alpha
    if args.group_alpha_min is not None:
        config.hierarchical.group_alpha_min = args.group_alpha_min
    if args.group_alpha_max is not None:
        config.hierarchical.group_alpha_max = args.group_alpha_max

    # optimize-loop 超参通道: 奖励/Lyapunov 数值旋钮注入
    if args.w_overflow is not None:
        config.reward.w_overflow = args.w_overflow
    if args.w_outage is not None:
        config.reward.w_outage = args.w_outage
    if args.w_power_exceed is not None:
        config.reward.w_power_exceed = args.w_power_exceed
    if args.V_min is not None:
        config.lyapunov.V_min = args.V_min
    if args.V_max is not None:
        config.lyapunov.V_max = args.V_max
    if args.V_exponent is not None:
        config.lyapunov.V_exponent = args.V_exponent
    if args.blend_alpha is not None:
        config.lyapunov.blend_alpha = args.blend_alpha
    if args.reward_dpp:
        # neural-DPP comparator: classical drift-plus-penalty reward only
        ly = config.lyapunov
        ly.ee_mode = False
        ly.ee_weight = 0.0
        ly.ee_gate_enable = False
        ly.use_dinkelbach = False
        ly.w_deliv = 0.0
        ly.w_ee_floor = 0.0
        ly.ldptrlq_q_weight = 0.0
        ly.w_early_overflow = 0.0
        ly.w_backpressure = 0.0

    # Mean-Field MAPPO toggles
    if args.no_softmax_power_share:
        config.system.use_softmax_power_share = False
    if args.n_r is not None:
        if args.n_r < 1:
            raise ValueError(f"--n_r 必须 >= 1，got {args.n_r}")
        config.system.N_r = int(args.n_r)
    if args.fixed_lambda is not None:
        if args.fixed_lambda <= 0:
            raise ValueError(f"--fixed_lambda 必须 > 0，got {args.fixed_lambda}")
        config.system.adaptive_lambda = False
        config.system.lambda_rate = float(args.fixed_lambda)
    if args.reward_v2_per_user_weight:
        config.lyapunov.reward_v2_per_user_weight = True
    if args.reward_v2_fixed_q is not None:
        if args.reward_v2_fixed_q <= 0:
            raise ValueError(f"--reward_v2_fixed_q 必须 > 0，got {args.reward_v2_fixed_q}")
        config.lyapunov.reward_v2_fixed_q = float(args.reward_v2_fixed_q)
    if args.no_mean_field_critic:
        config.ppo.use_mean_field_critic = False
    if args.no_per_user_advantage:
        config.ppo.use_per_user_advantage = False
    if args.per_user_adv_weight is not None:
        config.ppo.per_user_adv_weight = args.per_user_adv_weight

    # 桶专家多头 Actor
    if args.no_bucket_experts:
        config.ppo.use_bucket_experts = False
    if args.n_bucket_experts is not None:
        config.ppo.n_bucket_experts = args.n_bucket_experts
    if args.lr_bucket_expert is not None:
        config.ppo.lr_bucket_expert = args.lr_bucket_expert

    # Backpressure 先验
    if args.backpressure_beta is not None:
        if args.backpressure_beta < 0:
            raise ValueError(f"--backpressure_beta 必须 >= 0，got {args.backpressure_beta}")
        config.ppo.backpressure_beta = float(args.backpressure_beta)
    if args.backpressure_mode is not None:
        if args.backpressure_beta is None:
            print(f"[Train] 警告: --backpressure_mode={args.backpressure_mode} 已指定但 "
                  f"--backpressure_beta 未给（默认 0=先验关闭）")
        config.ppo.backpressure_mode = args.backpressure_mode

    # Solver-Expert Iteration（初始化层干预）
    if args.solver_expert:
        config.constraint.solver_expert_enabled = True
    if args.solver_expert_episodes is not None:
        config.constraint.solver_expert_episodes = args.solver_expert_episodes
    if args.solver_expert_steps is not None:
        config.constraint.solver_expert_steps = args.solver_expert_steps
    if args.solver_expert_lr is not None:
        config.constraint.solver_expert_lr = args.solver_expert_lr
    if args.solver_expert_bc_epochs is not None:
        config.constraint.solver_expert_bc_epochs = args.solver_expert_bc_epochs
    if args.solver_expert_anchor_updates is not None:
        config.constraint.solver_expert_anchor_updates = args.solver_expert_anchor_updates
    if args.solver_expert_anchor_decay is not None:
        config.constraint.solver_expert_anchor_decay = args.solver_expert_anchor_decay
    if args.solver_expert_anchor_coef is not None:
        config.constraint.solver_expert_anchor_coef = args.solver_expert_anchor_coef
    if args.solver_expert_relabel:
        config.constraint.solver_expert_relabel = True
    if args.solver_expert_no_mix_original:
        config.constraint.solver_expert_mix_original = False

    # M5 SFTP: 安全微调协议
    if args.clip_ratio is not None:
        config.ppo.clip_ratio = args.clip_ratio
    if args.lr_log_std is not None:
        config.ppo.lr_log_std = args.lr_log_std
    if args.actor_warmup_updates is not None:
        config.ppo.actor_warmup_updates = args.actor_warmup_updates
    if args.meta_freeze_episodes is not None:
        config.hierarchical.meta_freeze_episodes = args.meta_freeze_episodes
    if args.kl_stop_lr_decay is not None:
        config.ppo.kl_stop_lr_decay = args.kl_stop_lr_decay
    if args.lr_freeze_update is not None:
        config.ppo.lr_freeze_update = args.lr_freeze_update
    if args.eval_guard:
        config.ppo.eval_guard_enable = True
    if args.eval_guard_every is not None:
        config.ppo.eval_guard_every = args.eval_guard_every
    if args.eval_guard_episodes is not None:
        config.ppo.eval_guard_episodes = args.eval_guard_episodes
    if args.eval_guard_margin is not None:
        config.ppo.eval_guard_margin = args.eval_guard_margin
    if args.eval_guard_max_halvings is not None:
        config.ppo.eval_guard_max_halvings = args.eval_guard_max_halvings
    # P1 早停
    config.ppo.eval_guard_early_stop = args.eval_guard_early_stop
    if args.eval_guard_early_stop_patience is not None:
        config.ppo.eval_guard_early_stop_patience = args.eval_guard_early_stop_patience
    if args.eval_guard_early_stop_min_episodes is not None:
        config.ppo.eval_guard_early_stop_min_episodes = args.eval_guard_early_stop_min_episodes
    # M5.2
    if args.kl_bc_coef is not None:
        config.ppo.kl_bc_coef = args.kl_bc_coef
    if args.eval_guard_v2:
        config.ppo.eval_guard_v2 = True
    if args.eval_guard_queue_ratio is not None:
        config.ppo.eval_guard_queue_ratio = args.eval_guard_queue_ratio
    if args.eval_guard_snr_points:
        try:
            config.ppo.eval_guard_snr_points = _validate_eval_guard_snr_points(
                [float(s) for s in args.eval_guard_snr_points.split(",") if s.strip()])
        except ValueError as e:
            parser.error(str(e))

    # 冒烟测试：极小配置覆盖（快速验证代码可运行性）
    if args.smoke:
        config.ppo.total_episodes = 1
        config.ppo.rollout_steps = 64
        config.ppo.batch_size = 16

    train(config)
