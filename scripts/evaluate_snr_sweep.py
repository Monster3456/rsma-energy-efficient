"""多 SNR 鲁棒性评估脚本

在 imperfect CSI 模式下，扫描 0~30 dB 共 7 个 SNR 点，
对比 PPO-RSMA / SDMA / NOMA / EqualPower 共 4 种方法，
每个 SNR 点跑 50 个 episodes (seed=0..49)，输出 Markdown 表格。

SNR(dB) = 10*log10(P_signal / sigma2), 其中 P_signal 取 P_k_max=1.0W
  SNR(dB):     0      5     10      15     20      25     30
  sigma2:    1.0   0.316  0.1   0.0316  0.01   0.00316  0.001
"""
import argparse
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv
from src.models.actor_critic import ActorCritic, ActorCriticAttn
from src.models.meta_controller import MetaActorCritic
from scripts.baselines.sdma_baseline import SDMABaseline
from scripts.baselines.noma_baseline import NOMABaseline
from scripts.baselines.equal_power import EqualPowerBaseline


METRIC_KEYS = ['total_power', 'mean_queue', 'delivery_ratio',
               'R_sum', 'overflow_rate']

# (SNR_dB, sigma2)
SNR_POINTS = [
    (0,  1.0),
    (5,  0.3162),
    (10, 0.1),
    (15, 0.03162),
    (20, 0.01),
    (25, 0.003162),
    (30, 0.001),
]

ALGO_ORDER = ['PPO-RSMA', 'SDMA', 'NOMA', 'EqualPower']


def _empty_acc():
    return {k: [] for k in METRIC_KEYS}


def _meta_decide_eval(meta_ac, env):
    """评估时运行一次 Meta-Controller 决策（确定性模式）。"""
    user_features = env.get_user_features()  # (K, d_user)
    user_features_tensor = torch.from_numpy(user_features).float().unsqueeze(0)
    with torch.no_grad():
        primary, group_ids, _, _ = meta_ac.act(user_features_tensor, deterministic=True)
    primary_np = primary.detach().cpu().numpy().astype(np.float32)
    group_ids_np = group_ids.detach().cpu().numpy().astype(np.int64)
    env.set_split_mask(primary_np)
    env.set_group_ids(group_ids_np)


def _run_episode(action_fn, env, seed, meta_ac=None, T_s=20):
    obs, info = env.reset(seed=seed)
    # 如果有 Meta-Controller，在 episode 开始时先做一次决策
    if meta_ac is not None:
        _meta_decide_eval(meta_ac, env)
    sums = {'total_power': 0.0, 'mean_queue': 0.0,
            'R_sum': 0.0, 'overflow_rate': 0.0}
    last_delivery = 0.0
    steps = 0
    for _ in range(env.T):
        action = action_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        sums['total_power'] += info['total_power']
        sums['mean_queue'] += info['mean_queue']
        sums['R_sum'] += info['R_sum']
        sums['overflow_rate'] += info['overflow_rate']
        last_delivery = info['delivery_ratio']
        steps += 1
        # Meta-Controller 周期性决策（T_s 步间隔）
        if meta_ac is not None and steps % T_s == 0 and not (truncated or terminated):
            _meta_decide_eval(meta_ac, env)
        if truncated or terminated:
            break
    steps = max(steps, 1)
    return {
        'total_power': sums['total_power'] / steps,
        'mean_queue': sums['mean_queue'] / steps,
        'delivery_ratio': last_delivery,
        'R_sum': sums['R_sum'] / steps,
        'overflow_rate': sums['overflow_rate'] / steps,
    }


def evaluate_method(action_fn, env, n_episodes, meta_ac=None, T_s=20):
    acc = _empty_acc()
    for ep in range(n_episodes):
        m = _run_episode(action_fn, env, seed=ep, meta_ac=meta_ac, T_s=T_s)
        for k in METRIC_KEYS:
            acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def make_baseline_action_fn(baseline):
    return lambda obs, env: baseline.select_action(obs, env)


def make_ppo_action_fn(model, ckpt_obs_dim, env_obs_dim, force_alpha_zero=False, K=None):
    """构造 PPO 动作函数。

    若 checkpoint 的 obs_dim 小于环境 obs_dim（例如旧的 128 维模型），
    则截断观测末尾的 (env_obs_dim - ckpt_obs_dim) 维（如 log10_sigma2 等扩展维度）。

    PL 模式: model.act 返回 5 元素 (action_continuous, decoding_order, log_prob, value, cont_log_prob)，
              将连续动作 (3K) 和解码顺序 (K) 拼接为 4K 维 action 传给 env。
    Non-PL 模式: model.act 返回 3 元素 (action, log_prob, value)。

    Args:
        force_alpha_zero: 若为 True，强制 action[:K] = -1.0（映射后 alpha=0，纯私有流）。
        K: 用户数，force_alpha_zero 时必须提供。
    """
    keep = ckpt_obs_dim if ckpt_obs_dim is not None else env_obs_dim
    use_pl = bool(getattr(model, "use_plackett_luce", False))

    def _fn(obs, env):
        with torch.no_grad():
            obs_arr = np.asarray(obs)
            if obs_arr.shape[-1] != keep:
                obs_arr = obs_arr[..., :keep]
            obs_t = torch.from_numpy(obs_arr).float().unsqueeze(0)
            if getattr(model, "use_bucket_experts", False):
                model.set_active_bucket(float(env.current_snr_db))
            result = model.act(obs_t, deterministic=True)
            if use_pl:
                # 5 元素: (action_cont, decoding_order, log_prob, value, cont_log_prob)
                action_cont = result[0].squeeze(0).cpu().numpy()
                decoding_order = result[1].squeeze(0).cpu().numpy().astype(np.float64)
                action = np.concatenate([action_cont, decoding_order])
            else:
                # 3 元素: (action, log_prob, value)
                action = result[0].squeeze(0).cpu().numpy()
            # 强制 alpha=0: action[:K] = -1.0 → 经 (x+1)/2 映射后 alpha=0
            if force_alpha_zero and K is not None:
                action[:K] = -1.0
            return action
    return _fn


def _infer_obs_dim_from_state(state):
    """从 state_dict 推断 actor_net 第一层的输入维度（仅 MLP ActorCritic 有此 key）。"""
    for key, tensor in state.items():
        if key.endswith('actor_net.0.weight'):
            return int(tensor.shape[1])
    return None


def _infer_action_dim_from_state(state):
    """从 state_dict 推断 actor_head 输出维度（48=non-PL 3K, 64=PL 4K）。"""
    for key, tensor in state.items():
        if key.endswith('actor_head.weight'):
            return int(tensor.shape[0])
    return None


def _is_attention_checkpoint(state):
    """检测 checkpoint 是否为 ActorCriticAttn（有 user_embed / attn_layers 等专属 key）。"""
    return any(k.startswith("user_embed.") or k.startswith("attn_layers.") for k in state.keys())


def load_meta_model(checkpoint_path, env):
    """从 checkpoint 加载 Meta-Controller。

    Returns:
        (meta_ac, T_s, err): meta_ac=None 时表示加载失败，err 为错误信息。
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, 20, f"checkpoint not found: {checkpoint_path}"
    config = env.config
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        return None, 20, "checkpoint is not a dict, no meta_state_dict"
    meta_state = ckpt.get('meta_state_dict')
    if meta_state is None:
        return None, 20, "no meta_state_dict in checkpoint"

    # 从 meta_state_dict 推断参数
    K = env.K
    h_cfg = config.hierarchical
    # d_user 从 embed.0.weight shape[1] 推断
    d_user = int(meta_state['embed.0.weight'].shape[1])
    hidden_dim = int(meta_state['embed.0.weight'].shape[0])
    n_groups = int(h_cfg.n_groups)
    n_heads = int(h_cfg.meta_n_heads)
    # 检测 group_level_alpha: 有 alpha_log_std 键则为 True
    group_level_alpha = 'alpha_log_std' in meta_state
    group_alpha_min = float(getattr(h_cfg, 'group_alpha_min', 0.1))
    group_alpha_max = float(getattr(h_cfg, 'group_alpha_max', 0.95))

    T_s = max(int(h_cfg.T_s), 1)

    print(
        f"  [META] Loading MetaActorCritic (K={K}, d_user={d_user}, "
        f"hidden={hidden_dim}, n_groups={n_groups}, "
        f"group_level_alpha={group_level_alpha}, T_s={T_s})",
        flush=True,
    )

    meta_ac = MetaActorCritic(
        K=K,
        d_user=d_user,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_groups=n_groups,
        group_level_alpha=group_level_alpha,
        group_alpha_min=group_alpha_min,
        group_alpha_max=group_alpha_max,
    )
    meta_ac.load_state_dict(meta_state)
    meta_ac.eval()
    return meta_ac, T_s, None


def load_ppo_model(checkpoint_path, env):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, None, f"checkpoint not found: {checkpoint_path}"
    config = env.config
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get('worker_state_dict') or ckpt.get('model_state_dict') or ckpt
    else:
        state = ckpt

    # 检测 Attention vs MLP
    if _is_attention_checkpoint(state):
        # ActorCriticAttn: 从 user_embed.0.weight 推断
        # P1: shape[1] = d_user (e.g. 10) → inject=False
        # P2: shape[1] = d_user + n_inject (e.g. 12) → inject=True
        d_user_eff = int(state["user_embed.0.weight"].shape[1])
        hidden = int(state["user_embed.0.weight"].shape[0])
        d_global = int(state["global_proj.weight"].shape[1])
        K = env.K
        # 假设 n_inject=2 (snr+eps)，d_user_obs = d_user_eff - n_inject
        # 以 env.config.ppo.worker_d_user 为依据推断 d_user，n_inject = d_user_eff - d_user
        d_user_cfg = int(getattr(config.ppo, "worker_d_user", 10))
        if d_user_eff == d_user_cfg:
            inject_global = False
            n_inject = 0
            d_user = d_user_cfg
        else:
            d_user = d_user_cfg
            n_inject = d_user_eff - d_user
            inject_global = n_inject > 0
        ckpt_obs_dim = K * d_user + d_global
        n_attn_layers = len(set(
            k.split('.')[1] for k in state.keys() if k.startswith("attn_layers.")
        ))
        n_heads = getattr(config.ppo, "worker_attn_heads", 4)
        action_dim = 4 * K
        if ckpt_obs_dim != env.obs_dim:
            print(
                f"  [INFO] attention checkpoint obs_dim={ckpt_obs_dim} != env.obs_dim={env.obs_dim}; "
                f"截断兼容评估",
                flush=True,
            )
        # 从 checkpoint 推断 cond_dim（FiLM 输入维度）：saute_film_embed.0.weight 的 shape[1]
        # iter-3 的 Sauté 用 cond_dim=1，iter-4+ 的双约束深化用 cond_dim=6（[z_k, λ̃_1..B, μ̃]）
        # 但当前代码可能未包含 Sauté 模块（未 commit 的改动被回滚），需检查 ActorCriticAttn 是否接受 cond_dim
        import inspect
        cond_dim = 1
        if "saute_film_embed.0.weight" in state:
            cond_dim = int(state["saute_film_embed.0.weight"].shape[1])
        ac_params = list(inspect.signature(ActorCriticAttn.__init__).parameters.keys())
        has_cond_dim = "cond_dim" in ac_params
        print(
            f"  [INFO] loading ActorCriticAttn (K={K}, d_user={d_user}, "
            f"d_global={d_global}, hidden={hidden}, attn_layers={n_attn_layers}, "
            f"inject_global={inject_global}, n_inject={n_inject}, "
            f"cond_dim={cond_dim if has_cond_dim else 'N/A'})",
            flush=True,
        )
        model_kwargs = dict(
            obs_dim=ckpt_obs_dim,
            action_dim=action_dim,
            K=K,
            d_user=d_user,
            d_global=d_global,
            hidden_dim=hidden,
            n_heads=n_heads,
            n_attn_layers=n_attn_layers,
            use_plackett_luce=True,
            inject_global_to_per_user=inject_global,
            n_global_inject=n_inject,
        )
        if has_cond_dim:
            model_kwargs["cond_dim"] = cond_dim
        model = ActorCriticAttn(**model_kwargs)
        # 兼容性加载：strict=False 跳过训练新增但评估不需要的参数（如 HAPPO 的 pl_pairwise_attn.*）
        # 维度变化参数（如 cond_dim 从 1→6）已通过 cond_dim 推断匹配，此处不会触发 shape mismatch
        model.load_state_dict(state, strict=False)
        model.eval()
        return model, ckpt_obs_dim, None

    # MLP ActorCritic 路径（原有逻辑）
    ckpt_obs_dim = _infer_obs_dim_from_state(state) or env.obs_dim
    ckpt_action_dim = _infer_action_dim_from_state(state) or env.action_dim
    use_pl = (ckpt_action_dim == 4 * (env.action_dim // 3))
    if ckpt_obs_dim != env.obs_dim:
        print(
            f"  [INFO] checkpoint obs_dim={ckpt_obs_dim} != env.obs_dim={env.obs_dim}; "
            f"将截断观测前 {ckpt_obs_dim} 维进行兼容评估",
            flush=True,
        )
    if ckpt_action_dim != env.action_dim:
        print(
            f"  [INFO] checkpoint action_dim={ckpt_action_dim} != env.action_dim={env.action_dim} "
            f"(PL={use_pl}); 按 PL={use_pl} 模式构造网络",
            flush=True,
        )
    model = ActorCritic(
        obs_dim=ckpt_obs_dim,
        action_dim=ckpt_action_dim,
        hidden_dim=config.ppo.hidden_dim,
        n_layers=config.ppo.n_layers,
        use_plackett_luce=use_pl,
    )
    model.load_state_dict(state)
    model.eval()
    return model, ckpt_obs_dim, None


def build_config_for_snr(sigma2: float, csi_mode: str) -> Config:
    """构造临时 Config，覆盖 sigma2，不修改全局默认值。

    评估时强制关闭 SNR domain randomization，确保每个评估点固定在指定 sigma2，
    避免 env.reset() 重新采样导致 SNR 漂移。
    """
    cfg = Config()
    cfg.system.csi_mode = csi_mode
    cfg.system.sigma2 = float(sigma2)
    cfg.system.snr_domain_randomization = False
    return cfg


def run_one_snr(snr_db, sigma2, n_episodes, ppo_ckpt, csi_mode, force_alpha_zero=False, baselines='sdma'):
    print(f"\n=== SNR={snr_db:>2} dB  (sigma2={sigma2:.4g})  ===", flush=True)
    cfg = build_config_for_snr(sigma2, csi_mode)
    env = RSMAEnergyEnv(cfg)

    results = {}

    if not force_alpha_zero:
        selected = set(baselines.lower().split(',')) if baselines else {'sdma'}
        idx = 1
        total = len(selected) + 1  # 基线 + PPO
        if 'sdma' in selected:
            print(f"  [{idx}/{total}] SDMA ...", flush=True)
            results['SDMA'] = evaluate_method(
                make_baseline_action_fn(SDMABaseline(cfg)), env, n_episodes
            )
            idx += 1
        if 'noma' in selected:
            print(f"  [{idx}/{total}] NOMA ...", flush=True)
            results['NOMA'] = evaluate_method(
                make_baseline_action_fn(NOMABaseline(cfg)), env, n_episodes
            )
            idx += 1
        if 'equal_power' in selected:
            print(f"  [{idx}/{total}] EqualPower ...", flush=True)
            results['EqualPower'] = evaluate_method(
                make_baseline_action_fn(EqualPowerBaseline(cfg)), env, n_episodes
            )
            idx += 1

    label = 'PPO-Alpha0' if force_alpha_zero else 'PPO-RSMA'
    print(f"  [PPO] {label} ...", flush=True)
    model, ckpt_obs_dim, err = load_ppo_model(ppo_ckpt, env)
    if model is None:
        print(f"  [WARN] {err}; skip PPO", flush=True)
    else:
        if force_alpha_zero:
            # 禁用 group_level_alpha，跳过 Meta-Controller
            env.group_level_alpha = False
            env.split_mask = np.ones(env.K, dtype=np.float32)  # split_mask=1 不阻断
            print(f"  [INFO] force_alpha_zero=True: disabled group_level_alpha, "
                  f"skipping Meta-Controller", flush=True)
            meta_ac, T_s = None, 20
        else:
            # 正常模式：加载 Meta-Controller
            meta_ac, T_s, meta_err = load_meta_model(ppo_ckpt, env)
            if meta_ac is None:
                print(f"  [WARN] Meta-Controller load failed: {meta_err}; "
                      f"alpha will stay at reset default", flush=True)
        results[label] = evaluate_method(
            make_ppo_action_fn(model, ckpt_obs_dim, env.obs_dim,
                              force_alpha_zero=force_alpha_zero, K=env.K),
            env, n_episodes, meta_ac=meta_ac, T_s=T_s
        )

    algo_order = [label] if force_alpha_zero else ALGO_ORDER
    for name in algo_order:
        if name not in results:
            continue
        r = results[name]
        print(
            f"    {name:<11s}  P={r['total_power']:.3f}  Q={r['mean_queue']:.2f}  "
            f"Deliv={r['delivery_ratio']:.3f}  R_sum={r['R_sum']:.2f}  "
            f"Ovf={r['overflow_rate']:.4f}",
            flush=True,
        )
    return results


def fmt_overview_table(all_results):
    """大宽表：行=SNR，列=各算法 × 5 指标。"""
    metric_label = {
        'total_power': 'P(W)',
        'mean_queue': 'Q',
        'delivery_ratio': 'Deliv',
        'R_sum': 'Rsum',
        'overflow_rate': 'Ovf',
    }
    headers = ['SNR(dB)']
    for algo in ALGO_ORDER:
        for m in METRIC_KEYS:
            headers.append(f"{algo} {metric_label[m]}")
    sep = '| ' + ' | '.join(
        ['---' if i == 0 else '---:' for i in range(len(headers))]
    ) + ' |'

    lines = ['### 表1：0~30 dB SNR 全域对比（imperfect CSI, 50 episodes/点）', '']
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append(sep)
    for snr_db, _, res in all_results:
        row = [f"{snr_db}"]
        for algo in ALGO_ORDER:
            r = res.get(algo)
            if r is None:
                row.extend(['n/a'] * len(METRIC_KEYS))
                continue
            row.append(f"{r['total_power']:.3f}")
            row.append(f"{r['mean_queue']:.2f}")
            row.append(f"{r['delivery_ratio']:.3f}")
            row.append(f"{r['R_sum']:.2f}")
            row.append(f"{r['overflow_rate']:.4f}")
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')
    return '\n'.join(lines)


def fmt_metric_tables(all_results):
    """按指标拆分：每个指标一张表 (SNR × 算法)。"""
    titles = {
        'total_power':   '表2a：总功率 (W) vs SNR  — 越低越好',
        'mean_queue':    '表2b：平均队列 vs SNR  — 越低越好',
        'delivery_ratio':'表2c：Delivery Ratio vs SNR  — 越高越好',
        'R_sum':         '表2d：R_sum (bps/Hz) vs SNR  — 越高越好',
        'overflow_rate': '表2e：溢出率 vs SNR  — 越低越好',
    }
    fmts = {
        'total_power': '{:.4f}',
        'mean_queue': '{:.2f}',
        'delivery_ratio': '{:.4f}',
        'R_sum': '{:.3f}',
        'overflow_rate': '{:.4f}',
    }
    blocks = []
    for m in METRIC_KEYS:
        lines = [f"### {titles[m]}", '']
        head = ['SNR(dB)'] + ALGO_ORDER
        lines.append('| ' + ' | '.join(head) + ' |')
        lines.append('|---:' + '|---:' * len(ALGO_ORDER) + '|')
        for snr_db, _, res in all_results:
            row = [f"{snr_db}"]
            for algo in ALGO_ORDER:
                r = res.get(algo)
                row.append('n/a' if r is None else fmts[m].format(r[m]))
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')
        blocks.append('\n'.join(lines))
    return '\n'.join(blocks)


def fmt_advantage_summary(all_results):
    """突出 PPO-RSMA 在哪些 SNR 区间优势/劣势明显。

    使用复合指标：相对 3 个 baseline 均值，PPO 在 4 项关键指标上的改进百分比。
    """
    def pct(num, den):
        if abs(den) < 1e-12:
            return 0.0
        return num / den * 100.0

    lines = ['### 表3：PPO-RSMA 相对基线均值的综合改进 (%)', '']
    lines.append('| SNR(dB) | ΔPower (省) | ΔQueue (减) | ΔDelivery (增) | ΔR_sum (增) | 综合得分 |')
    lines.append('|---:|---:|---:|---:|---:|---:|')
    notes = []
    for snr_db, _, res in all_results:
        ppo = res.get('PPO-RSMA')
        if ppo is None:
            lines.append(f"| {snr_db} | n/a | n/a | n/a | n/a | n/a |")
            continue
        baselines = [res[n] for n in ['SDMA', 'NOMA', 'EqualPower'] if n in res]
        if not baselines:
            continue
        b_mean = {k: float(np.mean([b[k] for b in baselines])) for k in METRIC_KEYS}
        d_power = pct(b_mean['total_power'] - ppo['total_power'], b_mean['total_power'])
        d_queue = pct(b_mean['mean_queue'] - ppo['mean_queue'], b_mean['mean_queue'])
        d_deliv = pct(ppo['delivery_ratio'] - b_mean['delivery_ratio'], b_mean['delivery_ratio'])
        d_rsum  = pct(ppo['R_sum'] - b_mean['R_sum'], b_mean['R_sum'])
        score = d_power + d_queue + d_deliv + d_rsum
        lines.append(
            f"| {snr_db} | {d_power:+.2f} | {d_queue:+.2f} | {d_deliv:+.2f} | "
            f"{d_rsum:+.2f} | {score:+.2f} |"
        )
        notes.append((snr_db, score))

    if notes:
        notes_sorted = sorted(notes, key=lambda x: x[1])
        worst = notes_sorted[0]
        best = notes_sorted[-1]
        lines.append('')
        lines.append(f"- **优势最大**：SNR = {best[0]} dB（综合得分 {best[1]:+.2f}）")
        lines.append(f"- **劣势最大**：SNR = {worst[0]} dB（综合得分 {worst[1]:+.2f}）")
        lines.append('- 综合得分 = ΔPower% + ΔQueue% + ΔDelivery% + ΔR_sum%，正向越大越好')
    lines.append('')
    return '\n'.join(lines)


def _md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _diagnose_checkpoint(eval_path):
    """管线诊断：对比评估目标 checkpoint 与训练最近保存 checkpoint 的内容 hash/mtime。"""
    real_eval = os.path.realpath(eval_path)
    if not os.path.isfile(real_eval):
        print(f"[CKPT-DIAG] eval_target missing: {real_eval}", flush=True)
        return
    eval_hash = _md5(real_eval)
    eval_mtime = os.path.getmtime(real_eval)
    print(f"[CKPT-DIAG] eval_target={real_eval} md5={eval_hash} mtime={eval_mtime}", flush=True)

    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    candidates = glob.glob(os.path.join(logs_dir, '*', 'checkpoints', '*.pt'))
    if not candidates:
        print("[CKPT-DIAG] no training checkpoint found under logs/", flush=True)
        return
    latest = max(candidates, key=os.path.getmtime)
    real_latest = os.path.realpath(latest)
    latest_hash = _md5(real_latest)
    latest_mtime = os.path.getmtime(real_latest)
    print(f"[CKPT-DIAG] latest_train={real_latest} md5={latest_hash} mtime={latest_mtime}", flush=True)
    # 告警式诊断（不阻断评估）：best.pt 与 final.pt 本就不同（best-val vs 末轮），
    # 不同属正常；真正要抓的是 eval 目标为陈旧/错误 checkpoint 的情形，交由人工判读输出
    if eval_hash != latest_hash:
        print(f"[CKPT-DIAG] WARN: eval 目标与最新训练 checkpoint 不同"
              f"（best-val vs final 差异属正常；意外差异请检查管线）", flush=True)
    else:
        print("[CKPT-DIAG] OK: eval checkpoint matches latest training checkpoint", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument(
        '--checkpoint', type=str,
        default='logs/snr_robust/checkpoints/final.pt',
    )
    parser.add_argument('--csi_mode', type=str, default='imperfect',
                        choices=['perfect', 'imperfect'])
    parser.add_argument('--snr_subset', type=str, default='',
                        help='可选: 逗号分隔的 SNR(dB) 子集, 例如 "0,10,20,30"')
    parser.add_argument('--force_alpha_zero', action='store_true', default=False,
                        help='强制 alpha=0（纯私有流），跳过 Meta-Controller')
    parser.add_argument('--baselines', type=str, default='sdma',
                        help='逗号分隔的基线方法: sdma,noma,equal_power（默认仅 sdma）')
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    snr_points = SNR_POINTS
    if args.snr_subset:
        wanted = {int(x.strip()) for x in args.snr_subset.split(',') if x.strip()}
        snr_points = [p for p in SNR_POINTS if p[0] in wanted]

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), '..', ckpt_path)
    ckpt_path = os.path.normpath(ckpt_path)
    _diagnose_checkpoint(ckpt_path)
    print(f"PPO checkpoint: {ckpt_path}", flush=True)
    print(f"CSI mode      : {args.csi_mode}", flush=True)
    print(f"Episodes/SNR  : {args.n_episodes}", flush=True)
    print(f"SNR points    : {[s for s, _ in snr_points]}", flush=True)

    if args.force_alpha_zero:
        print(f"*** FORCE ALPHA=0 MODE: 禁用公共流，纯私有流评估 ***", flush=True)

    all_results = []
    for snr_db, sigma2 in snr_points:
        res = run_one_snr(snr_db, sigma2, args.n_episodes, ckpt_path, args.csi_mode,
                         force_alpha_zero=args.force_alpha_zero,
                         baselines=args.baselines)
        all_results.append((snr_db, sigma2, res))

    print("\n\n===== SNR SWEEP REPORT =====\n", flush=True)
    print(fmt_overview_table(all_results))
    print(fmt_metric_tables(all_results))
    print(fmt_advantage_summary(all_results))


if __name__ == "__main__":
    main()
