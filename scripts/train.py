"""训练启动脚本，支持命令行参数覆盖（含 Task#11 消融标志）"""
import argparse
import sys
sys.path.insert(0, '/Users/libo/PycharmProjects/rsma-energy-efficient')

from src.training.main import train
from src.config import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSMA Energy-Efficient PPO Training")
    parser.add_argument('--V', type=float, default=1.0, help='Lyapunov V parameter')
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
    parser.add_argument('--lr-bucket-expert', type=float, default=None,
                        help='专家头学习率（覆盖 ppo.lr_bucket_expert）')

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

    # Mean-Field MAPPO toggles
    if args.no_softmax_power_share:
        config.system.use_softmax_power_share = False
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

    # 冒烟测试：极小配置覆盖（快速验证代码可运行性）
    if args.smoke:
        config.ppo.total_episodes = 1
        config.ppo.rollout_steps = 64
        config.ppo.batch_size = 16

    train(config)
