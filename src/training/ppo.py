"""PPO算法实现 (v4)

基于Clip-PPO，结合Lyapunov漂移加惩罚的奖励信号进行策略优化。

v4 关键改动：
1. 严格 KL 早停（target_kl 从 0.03 -> 0.02），超过立即终止当前 update 的剩余 epoch
2. 线性 LR 衰减：在 set_progress(p in [0,1]) 后，每个 group 的 lr 从初始线性衰减到 init * lr_decay_final_ratio
3. 连续 KL 早停（>=3 次）时，再额外把 lr 乘 kl_stop_lr_decay (0.9)
"""
import copy

import torch
import torch.nn as nn
import numpy as np

from src.config import PPOConfig


class PPOTrainer:
    """PPO训练器（v4 加强稳定性）"""

    def __init__(self, actor_critic, config: PPOConfig):
        self.ac = actor_critic
        self.config = config

        # log_std 单独 param group：让 std 学习不被 actor_net 的 lr 节奏拖累
        lr_log_std = float(getattr(config, "lr_log_std", config.lr_actor))

        # 桶专家模式：共享主干冻结（lr=0）、专家头独立小 lr、critic/log_std 照常
        self.use_bucket_experts = bool(getattr(actor_critic, "use_bucket_experts", False))
        if self.use_bucket_experts:
            lr_expert = float(getattr(config, "lr_bucket_expert", config.lr_actor))
            self.optimizer = torch.optim.Adam([
                {'params': actor_critic.shared_trunk_params(), 'lr': 0.0},
                {'params': actor_critic.expert_head_params(), 'lr': lr_expert},
                {'params': actor_critic.critic_params(), 'lr': config.lr_critic},
                {'params': actor_critic.log_std_params(), 'lr': lr_log_std},
                # mw_gate 不随共享主干冻结（Run 13 门控需学习负载开关）：
                # 追加到尾部组，避免打乱 main.py 按索引定位 actor/critic 组的约定
                {'params': actor_critic.mw_gate_params(), 'lr': lr_expert},
            ])
        else:
            self.optimizer = torch.optim.Adam([
                {'params': actor_critic.actor_params(), 'lr': config.lr_actor},
                {'params': actor_critic.critic_params(), 'lr': config.lr_critic},
                {'params': actor_critic.log_std_params(), 'lr': lr_log_std},
            ])
        # 记录每个 param_group 的"基础学习率"，用于线性衰减
        self._base_lrs = [g['lr'] for g in self.optimizer.param_groups]
        # 当前由 KL 触发的额外衰减系数（每次连续 KL 早停 *0.9）
        self._kl_decay_factor = 1.0

        self.clip_ratio = config.clip_ratio
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef
        self.max_grad_norm = config.max_grad_norm
        self.n_epochs = config.n_epochs
        self.batch_size = config.batch_size
        # v4: 严格 KL 早停阈值
        self.target_kl = float(getattr(config, "target_kl", 0.02))
        self.lr_decay_final_ratio = float(getattr(config, "lr_decay_final_ratio", 0.2))
        self.kl_stop_lr_decay = float(getattr(config, "kl_stop_lr_decay", 0.9))
        self.kl_stop_streak_threshold = int(getattr(config, "kl_stop_streak_threshold", 3))
        # M5.2-F6: KL(π_θ‖π_BC) 显式信任域强度（0=关）。BC 参考副本由
        # set_bc_reference() 注入；未注入时本项严格无操作。
        self.kl_bc_coef = float(getattr(config, "kl_bc_coef", 0.0))
        self._bc_ac = None
        # M5.4: 显式 lr 冻结点（update 计数）。冠军 Arm A 的"冻结"是 KL 衰减连乘
        # 触底 lr_decay_final_ratio 下限的涌现结果（u100 起钉死），不可控也不可消融；
        # 本旋钮把它显式化：达到该 update 数后 scale 恒等于 lr_decay_final_ratio。
        # 0 = 关（行为逐位不变）。
        self.lr_freeze_update = int(getattr(config, "lr_freeze_update", 0))
        self._lr_frozen = False

        # 进度（由外部 set_progress 注入），决定线性衰减位置
        self._progress = 0.0
        # 连续 KL 早停计数
        self._consecutive_kl_stops = 0

        # ---- RCPO 独立 cost critic（不碰 actor_critic）----
        # delivery 约束走独立 cost 通道：reward 保持纯 EE 平稳，约束由
        # policy loss 中 A_eff = A_r - λ_scaled·A_c 施加。cost critic 学习
        # 稠密 per-slot 归一化交付赤字 cost_t 的折扣回报，提供因果信用。
        obs_dim = int(getattr(self.ac, "obs_dim"))
        device = next(self.ac.parameters()).device
        self.cost_critic = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), lr=config.lr_critic
        )
        # 由 main.py 从 ConstraintConfig 注入（PPOConfig 不含这些字段）
        self.cost_critic_coef = float(getattr(config, "cost_critic_coef", 0.5))
        self.cost_gamma = float(getattr(config, "cost_gamma", 0.99))

        # ---- C3PO 中央路径惩罚 ----
        self.c3po_enabled = bool(getattr(config, "c3po_enabled", False))
        self.c3po_n_points = int(getattr(config, "c3po_n_points", 7)) if self.c3po_enabled else 0
        self.c3po_nu_init = float(getattr(config, "c3po_nu_init", 0.05))
        self.c3po_nu_final = float(getattr(config, "c3po_nu_final", 0.5))
        self.c3po_kappa = float(getattr(config, "c3po_kappa", 1.0))
        self.c3po_margin_floor = float(getattr(config, "c3po_margin_floor", 0.02))
        self.c3po_cost_coef = float(getattr(config, "c3po_cost_coef", 0.5))
        self.c3po_gamma = float(getattr(config, "c3po_gamma", 0.99))
        self.c3po_gae_lambda = float(getattr(config, "c3po_gae_lambda", 0.95))
        self._c3po_progress = 0.0
        if self.c3po_enabled:
            # 校验 cost head 已挂载到 critic，否则回退
            if not (hasattr(self.ac, "critic_cost_head") and self.ac.critic_cost_head is not None):
                self.c3po_enabled = False

        # ---- HEK-Lite 默认状态（初始化层干预；hek_enabled=False 时零介入）----
        # 超参由 main.py 从 ConstraintConfig 注入；演示数据由 bc_pretrain 缓存。
        self.hek_enabled = False
        self.hek_obs = None
        self.hek_act = None
        self.hek_orders = None
        self._hek_updates = 0
        self.hek_bc_epochs = 150
        self.hek_bc_lr = 3e-4
        self.hek_anchor_updates = 0
        self.hek_anchor_coef = 0.2
        self.hek_anchor_decay = 0.95

        # ---- M5-F2: critic 预热（SFTP 安全微调协议，默认 0 = 关闭）----
        # 前 actor_warmup_updates 个 update 只训 critic/cost-critic：
        # actor+log_std 梯度置零（Adam 零梯度 + 零动量 ⇒ 参数严格不动），
        # 消除 value 头冷启动期的优势噪声对 BC 盆地的冲击
        self.actor_warmup_updates = int(getattr(config, "actor_warmup_updates", 0))
        self._ft_update_count = 0

    def predict_cost_value(self, obs) -> torch.Tensor:
        """预测单个观测的 cost 状态价值 V_c(s)（no_grad，返回标量 tensor）。"""
        device = next(self.cost_critic.parameters()).device
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).view(1, -1)
        with torch.no_grad():
            return self.cost_critic(obs_t).squeeze(-1).squeeze(0)

    def predict_c3po_cost_values(self, obs) -> np.ndarray | None:
        """C3PO: 预测单个观测的 7 维 cost 价值（no_grad，返回 numpy 数组或 None）。"""
        if not self.c3po_enabled:
            return None
        device = next(self.ac.parameters()).device
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).view(1, -1)
        with torch.no_grad():
            vals = self.ac.evaluate_cost_values(obs_t)
            if vals is None:
                return None
            return vals.squeeze(0).cpu().numpy().astype(np.float32)

    def _compute_c3po_cost_gae(self, costs, cost_values, dones, last_values):
        """C3PO: 对 7 维逐 SNR 点 cost 序列分别做 GAE。

        Args:
            costs: (n, n_points) 每步 cost 向量
            cost_values: (n, n_points) 每步 cost value 估计
            dones: (n,) 终止标志
            last_values: (n_points,) 最后一步 bootstrap cost value
        Returns:
            (cost_advantages, cost_returns) 均为 (n, n_points)
        """
        n, n_points = costs.shape
        cost_adv = np.zeros((n, n_points), dtype=np.float32)
        gae = np.zeros(n_points, dtype=np.float32)
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - float(dones[t])
            next_values = last_values if t == n - 1 else cost_values[t + 1]
            delta = costs[t] + self.c3po_gamma * next_values * next_non_terminal - cost_values[t]
            gae = delta + self.c3po_gamma * self.c3po_gae_lambda * next_non_terminal * gae
            cost_adv[t] = gae
        cost_returns = cost_adv + cost_values.astype(np.float32)
        # 每点独立 rollout 级标准化
        for j in range(n_points):
            col = cost_adv[:, j]
            std = col.std()
            if std > 1e-8:
                cost_adv[:, j] = (col - col.mean()) / std
        return cost_adv.astype(np.float32), cost_returns.astype(np.float32)

    def _c3po_nu(self) -> float:
        """开环 schedule：ν 随训练进度从 init 线性增长到 final。"""
        p = float(np.clip(self._c3po_progress, 0.0, 1.0))
        return self.c3po_nu_init + p * (self.c3po_nu_final - self.c3po_nu_init)

    def set_progress(self, progress: float) -> None:
        """通知训练进度（0->1），更新各 param_group 的 lr。"""
        p = float(np.clip(progress, 0.0, 1.0))
        self._progress = p
        self._c3po_progress = p
        self._apply_lr()

    def _apply_lr(self) -> None:
        if self._lr_frozen:
            # M5.4: 显式冻结——scale 钉死在下限，进度退火与 KL 衰减均不再作用
            scale = self.lr_decay_final_ratio
        else:
            # 线性进度衰减
            progress_scale = (1.0 - self._progress) + self._progress * self.lr_decay_final_ratio
            # KL 触发的额外衰减保留，但合并后不得低于 lr_decay_final_ratio (1/5) 下限，
            # 避免 LR 衰减到几乎冻结导致训练失速
            scale = progress_scale * self._kl_decay_factor
            scale = max(scale, self.lr_decay_final_ratio)
        for g, base in zip(self.optimizer.param_groups, self._base_lrs):
            g['lr'] = float(base * scale)

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]['lr'])

    def _compute_expert_grad_norms(self) -> dict:
        """桶专家模式下返回每个专家头的参数梯度 L2 范数。"""
        if not self.use_bucket_experts:
            return {}
        ac = self.ac
        norms = {}
        if ac.expert_heads is None:
            return norms
        for i, head in enumerate(ac.expert_heads):
            vals = []
            for p in head.parameters():
                if p.grad is not None:
                    vals.append(float(p.grad.norm().item()))
            norms[f"expert_{i}_grad_norm"] = float(np.mean(vals)) if vals else 0.0
        return norms

    def _compute_cost_gae(self, costs, cost_values, dones, last_value):
        """对稠密 per-slot cost 序列做标准 GAE（折扣 self.cost_gamma）。

        返回 (cost_advantages, cost_returns)；A_c 已做 rollout 级标准化
        （与主 advantages 同法，见 update 内注释），全零 cost（warmup）退化为 0。
        """
        n = costs.shape[0]
        cost_adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        gae_lambda = float(getattr(self.config, "gae_lambda", 0.95))
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - float(dones[t])
            next_value = float(last_value) if t == n - 1 else float(cost_values[t + 1])
            delta = float(costs[t]) + self.cost_gamma * next_value * next_non_terminal - float(cost_values[t])
            gae = delta + self.cost_gamma * gae_lambda * next_non_terminal * gae
            cost_adv[t] = gae
        cost_returns = cost_adv + cost_values.astype(np.float32)
        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)
        return cost_adv.astype(np.float32), cost_returns.astype(np.float32)

    def bc_pretrain(self, obs_np, act_np, orders_np=None) -> dict:
        """HEK-Lite / Solver-Expert: 专家演示行为克隆预训练。

        用独立 Adam(lr=hek_bc_lr) 对演示 (obs, act[, orders]) 做 hek_bc_epochs
        轮最大似然训练：orders_np=None 时 loss = -cont_log_prob.mean()（不动
        PL 排序头）；传入 (N,K) rank-value 解码顺序时 loss = -total_log_prob
        （连续 + PL 头联合训练，env 侧 4K 动作组装约定下 rank-value 即 PL
        排列目标）。训练后 actor 落在专家盆地附近；演示数据缓存为
        self.hek_obs/hek_act/hek_orders 供 update() 衰减锚定。
        """
        device = next(self.ac.parameters()).device
        obs_t = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=device)
        act_t = torch.as_tensor(
            np.asarray(act_np, dtype=np.float32), device=device
        ).clamp(-0.999, 0.999)  # atanh 定义域保护
        if orders_np is not None:
            orders_t = torch.as_tensor(
                np.asarray(orders_np, dtype=np.int64), device=device)
        else:
            orders_t = None
        self.hek_obs = obs_t
        self.hek_act = act_t
        self.hek_orders = orders_t
        self._hek_updates = 0

        n = int(obs_t.shape[0])
        if n == 0:
            return {"bc_loss": 0.0, "bc_epochs": 0}

        epochs = int(getattr(self, "hek_bc_epochs", 150))
        lr = float(getattr(self, "hek_bc_lr", 3e-4))
        params = list(self.ac.actor_params()) + list(self.ac.log_std_params())
        opt = torch.optim.Adam(params, lr=lr)
        bs = min(self.batch_size, n)

        self.ac.train()
        last_loss = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                log_prob, _, _, cont_log_prob = self.ac.evaluate_actions(
                    obs_t[idx], act_t[idx],
                    orders_t[idx] if orders_t is not None else None
                )
                loss = -log_prob.mean() if orders_t is not None \
                    else -cont_log_prob.mean()
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                opt.step()
                last_loss = float(loss.item())
        return {"bc_loss": last_loss, "bc_epochs": epochs}

    def _anchor_grad_into(self, actor_params, hek_coef: float, n_samples: int) -> float:
        """M5-F0: 锚定梯度直接加到 actor_params 的 .grad 上（词典序投影路径专用）。

        演示批 NLL = -log π_θ(demo) 对 actor+log_std 参数求梯度，× hek_coef 累加。
        锚定是稳定器（把策略拴在教师/BC 盆地），**不参与**词典序零空间投影——
        投影只应作用于 EE-vs-delivery 的目标梯度冲突。log_std 也在参数表中，
        锚定天然抑制探索坍缩（M4 熵尖峰 −51→−89 的通道之一）。

        Returns:
            锚定 NLL 值（日志用）；演示缺失或数值异常时返回 0.0 且不改动梯度。
        """
        if getattr(self, "hek_obs", None) is None or hek_coef <= 0.0:
            return 0.0
        try:
            n_hek = int(self.hek_obs.shape[0])
            hidx = torch.randint(
                0, n_hek, (int(n_samples),), device=self.hek_obs.device
            )
            h_orders = (
                self.hek_orders[hidx]
                if getattr(self, "hek_orders", None) is not None else None
            )
            h_log_prob, _, _, h_cont_log_prob = self.ac.evaluate_actions(
                self.hek_obs[hidx], self.hek_act[hidx], h_orders
            )
            anchor_nll = -(
                h_log_prob if h_orders is not None else h_cont_log_prob
            ).mean()
            if torch.isnan(anchor_nll) or torch.isinf(anchor_nll):
                return 0.0
            params = [p for p in actor_params if p.requires_grad]
            grads = torch.autograd.grad(anchor_nll, params, allow_unused=True)
            for p, g in zip(params, grads):
                if g is not None and p.grad is not None:
                    p.grad.add_(g, alpha=float(hek_coef))
            return float(anchor_nll.item())
        except Exception:
            return 0.0  # 锚定失败不影响主更新（与原路径同语义）

    def set_bc_reference(self, state_dict) -> bool:
        """M5.2-F6: 注入冻结的 BC 策略副本作为 KL 信任域参考。

        在 BC 预训练后、PPO 微调前由 main.py 调用。副本独立于 self.ac，
        不参与优化器、不接收梯度；未调用时 kl_bc 项严格无操作。
        """
        if self.kl_bc_coef <= 0.0 or state_dict is None:
            return False
        try:
            ref = copy.deepcopy(self.ac)
            ref.load_state_dict(state_dict)
            for p in ref.parameters():
                p.requires_grad_(False)
            ref.eval()
            self._bc_ac = ref
            return True
        except Exception:
            self._bc_ac = None
            return False

    def _klbc_term(self, obs, actions, decoding_orders, bucket_indices,
                   new_log_probs) -> torch.Tensor | None:
        """M5.2-F6: 单样本 KL 估计 relu(mean(log π_BC − log π_θ))，带 π_θ 侧梯度。

        动作来自 rollout 策略，故这是 KL 的有偏但标准估计（PPO approx_kl 同款）。
        relu 的语义：π_θ 比 BC 更集中于所采动作（即在改进）时不罚；只有当 π_θ
        在该状态上漂离 BC 盆地（log π_θ < log π_BC）才产生惩罚。
        """
        if self._bc_ac is None or self.kl_bc_coef <= 0.0:
            return None
        try:
            with torch.no_grad():
                if bucket_indices is not None:
                    bc_lp, _, _, _ = self._bc_ac.evaluate_actions(
                        obs, actions, decoding_orders, bucket_indices=bucket_indices
                    )
                else:
                    bc_lp, _, _, _ = self._bc_ac.evaluate_actions(
                        obs, actions, decoding_orders
                    )
            term = torch.relu(bc_lp.detach() - new_log_probs).mean()
            if torch.isnan(term) or torch.isinf(term):
                return None
            return term
        except Exception:
            return None

    def _trust_grad_into(self, actor_params, kl_term: torch.Tensor, coef: float) -> float:
        """M5.2-F6: 把 coef × d(kl_term)/d(actor参数) 累加到 .grad（词典序路径专用）。

        与 _anchor_grad_into 同语义：信任域是稳定器，不参与词典序零空间投影。
        """
        if kl_term is None or coef <= 0.0:
            return 0.0
        try:
            params = [p for p in actor_params if p.requires_grad]
            grads = torch.autograd.grad(kl_term, params, allow_unused=True)
            for p, g in zip(params, grads):
                if g is not None and p.grad is not None:
                    p.grad.add_(g, alpha=float(coef))
            return float(kl_term.item())
        except Exception:
            return 0.0

    def _apply_lexicographic_policy_grad(
        self,
        batch: dict,
        ratio: torch.Tensor,
        entropy: torch.Tensor,
        buffer,
        actor_params: list,
        ee_advantages: torch.Tensor = None,
    ) -> tuple[float, float]:
        """词典序梯度投影：按 SNR 桶分别计算 delivery 约束梯度 g_d 与 EE 梯度 g_e，
        仅当桶存在 delivery 赤字时将 g_e 投影到 g_d 的零空间，最后设置 actor 梯度。

        Returns:
            (policy_loss_sum, entropy_loss_sum) 用于日志近似。
        """
        advantages = batch["advantages"] if ee_advantages is None else ee_advantages
        cost_advantages = batch["cost_advantages"]
        snr_db = batch["snr_db"]
        n_total = int(advantages.shape[0])
        N = float(n_total)
        device = advantages.device

        bucket_deficits = getattr(self, "_lex_bucket_deficits", None)
        min_bucket_samples = int(getattr(self.config, "lex_proj_min_bucket_samples", 4))
        entropy_coef = float(self.entropy_coef)

        snr_np = snr_db.detach().cpu().numpy()
        n_buckets = int(buffer.n_snr_buckets)
        bucket_idx = np.clip(
            ((snr_np - float(buffer.snr_min))
             / max(float(buffer.snr_max) - float(buffer.snr_min), 1e-8)
             * n_buckets).astype(int),
            0, n_buckets - 1,
        )

        def _zero_actor():
            for p in actor_params:
                if p.grad is not None:
                    p.grad.zero_()

        def _get_actor_grad():
            return {p: p.grad.clone() for p in actor_params}

        def _flatten_grad(g_dict):
            return torch.cat([g_dict[p].reshape(-1) for p in actor_params])

        def _unflatten_grad(flat_g):
            offset = 0
            out = {}
            for p in actor_params:
                numel = p.numel()
                out[p] = flat_g[offset:offset + numel].view_as(p.data)
                offset += numel
            return out

        accum = {p: torch.zeros_like(p.data) for p in actor_params}
        policy_loss_sum = 0.0
        entropy_loss_sum = 0.0
        handled_mask = np.zeros(n_total, dtype=bool)

        for b in range(n_buckets):
            mask = (bucket_idx == b)
            n_b = int(mask.sum())
            if n_b < min_bucket_samples:
                continue
            mask_t = torch.from_numpy(mask).to(device)

            # delivery 约束梯度 g_d = grad mean(ratio * cost_advantages)
            delivery_loss_b = (ratio[mask_t] * cost_advantages[mask_t]).mean()
            _zero_actor()
            delivery_loss_b.backward(retain_graph=True)
            g_d = _get_actor_grad()

            # EE 梯度 g_e = grad[-mean(ratio * advantages)] + entropy
            policy_ee_loss_b = -(ratio[mask_t] * advantages[mask_t]).mean()
            entropy_scale_b = batch["entropy_scale"][mask_t]
            entropy_term_b = -(n_b / N) * entropy_coef * (entropy_scale_b * entropy[mask_t]).mean()
            ee_loss_b = policy_ee_loss_b + entropy_term_b
            _zero_actor()
            ee_loss_b.backward(retain_graph=True)
            g_e_neg = _get_actor_grad()  # = -g_e
            g_e = {p: -g_e_neg[p] for p in actor_params}

            policy_loss_sum += float(policy_ee_loss_b.item())
            entropy_loss_sum += float(entropy_term_b.item())

            deficit_b = float(bucket_deficits[b]) if bucket_deficits is not None else 1.0
            if deficit_b > 1e-8:
                g_d_flat = _flatten_grad(g_d)
                g_e_flat = _flatten_grad(g_e)
                dot = torch.dot(g_e_flat, g_d_flat)
                if dot < 0.0:
                    scale = dot / (torch.dot(g_d_flat, g_d_flat) + 1e-12)
                    g_e_flat_proj = g_e_flat - scale * g_d_flat
                    g_e = _unflatten_grad(g_e_flat_proj)
                for p in actor_params:
                    accum[p] += g_d[p] + g_e[p]
            else:
                for p in actor_params:
                    accum[p] += g_e[p]

            handled_mask |= mask

        # fallback：未进入有效桶的样本使用未投影 EE 梯度
        fallback_mask = ~handled_mask
        if fallback_mask.any():
            n_f = int(fallback_mask.sum())
            mask_t = torch.from_numpy(fallback_mask).to(device)
            policy_ee_loss_f = -(ratio[mask_t] * advantages[mask_t]).mean()
            entropy_scale_f = batch["entropy_scale"][mask_t]
            entropy_term_f = -(n_f / N) * entropy_coef * (entropy_scale_f * entropy[mask_t]).mean()
            ee_loss_f = policy_ee_loss_f + entropy_term_f
            _zero_actor()
            ee_loss_f.backward(retain_graph=True)
            g_e_neg_f = _get_actor_grad()
            for p in actor_params:
                accum[p] += -g_e_neg_f[p]
            policy_loss_sum += float(policy_ee_loss_f.item())
            entropy_loss_sum += float(entropy_term_f.item())

        # optimizer 做梯度下降，故 param.grad = -累计方向
        for p in actor_params:
            p.grad = -accum[p]

        return policy_loss_sum, entropy_loss_sum

    def update(self, buffer, cost_data=None, lambda_scaled=0.0,
               pi_delivery=None, pi_coef=0.0, pi_threshold=0.98,
               sample_weights=None, bucket_deficits=None,
               entropy_scale=None, c3po_data=None) -> dict:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_cost_value_loss = 0.0
        total_power_incentive_loss = 0.0
        total_hek_anchor_loss = 0.0
        total_klbc_loss = 0.0
        total_c3po_loss = 0.0
        total_c3po_value_loss = 0.0
        total_entropy = 0.0
        # M5.4 诊断：按 SNR band 拆分的熵累加（low<7.5 / mid 7.5-22.5 / high≥22.5 dB），
        # 用于区分"批构成偏移"与"真 log_std 坍缩"（M5.3 暴露的测量缺陷）
        ent_band_sum = [0.0, 0.0, 0.0]
        ent_band_n = [0, 0, 0]
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        n_updates = 0
        early_stopped = False

        # ---- M5-F2: critic 预热状态（SFTP，actor_warmup_updates=0 时恒 False）----
        self._ft_update_count = int(getattr(self, "_ft_update_count", 0)) + 1
        in_warmup = (
            self.actor_warmup_updates > 0
            and self._ft_update_count <= self.actor_warmup_updates
        )

        # ---- M5.4: 显式 lr 冻结触发（lr_freeze_update=0 时恒不触发）----
        if (not self._lr_frozen and self.lr_freeze_update > 0
                and self._ft_update_count >= self.lr_freeze_update):
            self._lr_frozen = True
            self._apply_lr()

        # ---- HEK-Lite: 衰减锚定（初始化层干预的微调阶段）----
        # 仅在 hek_enabled 且演示已缓存且仍处于预热期（_hek_updates <
        # hek_anchor_updates）时生效；系数逐 update 指数衰减，到期自动归零，
        # 之后退化为标准 PPO。hek_enabled=False 时整段跳过（零介入）。
        hek_anchor_active = bool(
            getattr(self, "hek_enabled", False)
            and getattr(self, "hek_obs", None) is not None
            and int(getattr(self, "_hek_updates", 0))
            < int(getattr(self, "hek_anchor_updates", 0))
        )
        hek_coef = 0.0
        if hek_anchor_active:
            hek_coef = float(getattr(self, "hek_anchor_coef", 0.2)) * (
                float(getattr(self, "hek_anchor_decay", 0.95))
                ** int(getattr(self, "_hek_updates", 0))
            )
            if hek_coef <= 0.0:
                hek_anchor_active = False

        # ---- RCPO: cost 优势预计算（rollout 级，一次即可）----
        use_cost = cost_data is not None
        if use_cost:
            n = buffer.pos
            cost_adv, cost_returns = self._compute_cost_gae(
                costs=np.asarray(cost_data["costs"], dtype=np.float32)[:n],
                cost_values=np.asarray(cost_data["cost_values"], dtype=np.float32)[:n],
                dones=buffer.dones[:n],
                last_value=float(cost_data.get("last_cost_value", 0.0)),
            )
        else:
            cost_adv = None
            cost_returns = None

        # ---- C3PO: 7维逐SNR点 cost 优势预计算 ----
        use_c3po = self.c3po_enabled and c3po_data is not None
        if use_c3po:
            n = buffer.pos
            c3po_costs = np.asarray(c3po_data["costs"], dtype=np.float32)[:n]
            c3po_values = np.asarray(c3po_data["cost_values"], dtype=np.float32)[:n]
            c3po_last = np.asarray(c3po_data.get("last_cost_values", np.zeros(self.c3po_n_points)), dtype=np.float32)
            c3po_cost_adv, c3po_cost_returns = self._compute_c3po_cost_gae(
                costs=c3po_costs,
                cost_values=c3po_values,
                dones=buffer.dones[:n],
                last_values=c3po_last,
            )
            c3po_budgets = np.asarray(c3po_data.get("budgets", np.zeros(self.c3po_n_points)), dtype=np.float32)
        else:
            c3po_cost_adv = None
            c3po_cost_returns = None
            c3po_budgets = None

        # ---- 词典序梯度投影：需要 cost_data 提供独立的 delivery 约束梯度 ----
        use_lex = bool(getattr(self.config, "use_lexicographic_projection", False)) and use_cost
        self._lex_bucket_deficits = bucket_deficits

        def _iter_batches():
            """统一 mini-batch 切片：支持 SFL 加权采样、RCPO cost、PLPI、PL decoding_order，
            并始终携带 snr_db 供按桶投影使用。"""
            if sample_weights is not None and len(sample_weights) > 0:
                weights = np.asarray(sample_weights, dtype=np.float64)[:buffer.pos]
                w_sum = float(weights.sum())
                if w_sum <= 1e-12:
                    weights = np.ones(buffer.pos, dtype=np.float64)
                    w_sum = float(buffer.pos)
                probs = weights / w_sum
                n_samples = buffer.pos
                for start in range(0, n_samples, self.batch_size):
                    end = min(start + self.batch_size, n_samples)
                    bi = np.random.choice(n_samples, size=end - start, replace=True, p=probs)
                    yield _build_batch(bi, cost_adv, cost_returns, pi_delivery)
            else:
                indices = np.random.permutation(buffer.pos)
                for start in range(0, buffer.pos, self.batch_size):
                    bi = indices[start:start + self.batch_size]
                    yield _build_batch(bi, cost_adv, cost_returns, pi_delivery)

        def _build_batch(bi, cost_adv, cost_returns, pi_delivery):
            batch = {
                "obs": torch.tensor(buffer.observations[bi], dtype=torch.float32, device=buffer.device),
                "actions": torch.tensor(buffer.actions[bi], dtype=torch.float32, device=buffer.device),
                "old_log_probs": torch.tensor(buffer.log_probs[bi], dtype=torch.float32, device=buffer.device),
                "old_cont_log_probs": torch.tensor(buffer.cont_log_probs[bi], dtype=torch.float32, device=buffer.device),
                "returns": torch.tensor(buffer.returns[bi], dtype=torch.float32, device=buffer.device),
                "advantages": torch.tensor(buffer.advantages[bi], dtype=torch.float32, device=buffer.device),
                "snr_db": torch.tensor(buffer.snr_dbs[bi], dtype=torch.float32, device=buffer.device),
            }
            if buffer.per_user_advantages is not None:
                batch["per_user_advantages"] = torch.tensor(
                    buffer.per_user_advantages[bi], dtype=torch.float32, device=buffer.device
                )
            if use_cost:
                batch["cost_advantages"] = torch.tensor(cost_adv[bi], dtype=torch.float32, device=buffer.device)
                batch["cost_returns"] = torch.tensor(cost_returns[bi], dtype=torch.float32, device=buffer.device)
            if use_c3po:
                batch["c3po_cost_advantages"] = torch.tensor(c3po_cost_adv[bi], dtype=torch.float32, device=buffer.device)
                batch["c3po_cost_returns"] = torch.tensor(c3po_cost_returns[bi], dtype=torch.float32, device=buffer.device)
                batch["c3po_budgets"] = torch.tensor(c3po_budgets, dtype=torch.float32, device=buffer.device)
            if pi_delivery is not None:
                n = buffer.pos
                pi_arr = np.asarray(pi_delivery[:n], dtype=np.float32)
                batch["pi_delivery"] = torch.tensor(pi_arr[bi], dtype=torch.float32, device=buffer.device)
            if buffer.decoding_orders is not None:
                batch["decoding_orders"] = torch.tensor(
                    buffer.decoding_orders[bi], dtype=torch.long, device=buffer.device
                )
            if entropy_scale is not None:
                batch["entropy_scale"] = torch.tensor(
                    entropy_scale[bi], dtype=torch.float32, device=buffer.device
                )
            else:
                batch["entropy_scale"] = torch.ones(
                    len(bi), dtype=torch.float32, device=buffer.device
                )
            return batch

        for epoch in range(self.n_epochs):
            epoch_kl = 0.0
            epoch_batches = 0
            kl_breached = False

            for batch in _iter_batches():
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                old_cont_log_probs = batch.get("old_cont_log_probs", None)
                returns = batch["returns"]
                advantages = batch["advantages"]
                decoding_orders = batch.get("decoding_orders", None)

                # Mean-Field MAPPO: 若存在 per-user advantages，取平均作为有效优势
                if "per_user_advantages" in batch:
                    adv_eff = batch["per_user_advantages"].mean(dim=1)
                else:
                    adv_eff = advantages

                # 注: advantages 已在 RolloutBuffer.compute_returns_and_advantages 做过 rollout 级标准化
                # 这里不再做 batch 级二次标准化（会在小样本上引入噪声、扭曲信号）

                # 桶专家硬路由：按 SNR 计算每个样本的桶索引
                bucket_indices = None
                if self.use_bucket_experts:
                    snr_np = batch["snr_db"].detach().cpu().numpy()
                    span = max(float(self.ac.bucket_snr_max) - float(self.ac.bucket_snr_min), 1e-8)
                    bucket_indices = np.clip(
                        ((snr_np - float(self.ac.bucket_snr_min)) / span * int(self.ac.n_experts)).astype(int),
                        0, int(self.ac.n_experts) - 1,
                    )

                if self.use_bucket_experts and bucket_indices is not None:
                    new_log_probs, values, entropy, new_cont_log_probs = self.ac.evaluate_actions(
                        obs, actions, decoding_orders,
                        bucket_indices=torch.from_numpy(bucket_indices).long().to(obs.device),
                    )
                else:
                    new_log_probs, values, entropy, new_cont_log_probs = self.ac.evaluate_actions(
                        obs, actions, decoding_orders
                    )

                # M5.2-F6: KL-to-BC 信任域项（未启用/未注入参考时为 None，严格无操作）
                klbc_term = self._klbc_term(
                    obs, actions, decoding_orders,
                    torch.from_numpy(bucket_indices).long().to(obs.device)
                    if (self.use_bucket_experts and bucket_indices is not None) else None,
                    new_log_probs,
                )

                # M5.4 诊断：按 SNR band 累加逐样本熵（no_grad，不参与梯度）
                with torch.no_grad():
                    snr_b = batch["snr_db"].detach().cpu().numpy()
                    ent_b = entropy.detach().cpu().numpy()
                    for k, m in enumerate((
                        snr_b < 7.5,
                        (snr_b >= 7.5) & (snr_b < 22.5),
                        snr_b >= 22.5,
                    )):
                        if m.any():
                            ent_band_sum[k] += float(ent_b[m].sum())
                            ent_band_n[k] += int(m.sum())

                # policy loss 用总 log_prob（连续 + PL）算 ratio，保证 PL 部分梯度正常反传
                log_ratio = new_log_probs - old_log_probs
                log_ratio = log_ratio.clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)

                if use_lex:
                    # ---------- 词典序梯度投影路径 ----------
                    # 1) value + cost critic 反向（只更新 critic，保留计算图）
                    value_loss = nn.functional.mse_loss(values, returns)
                    critic_loss = self.value_coef * value_loss
                    cost_value_loss = None
                    if use_cost:
                        cost_values_pred = self.cost_critic(obs).squeeze(-1)
                        cost_value_loss = nn.functional.mse_loss(
                            cost_values_pred, batch["cost_returns"]
                        )
                        critic_loss = critic_loss + self.cost_critic_coef * cost_value_loss

                    self.optimizer.zero_grad()
                    if use_cost:
                        self.cost_critic_optimizer.zero_grad()
                    critic_loss.backward(retain_graph=True)

                    # 2) actor 词典序投影梯度（按 SNR 桶）
                    actor_params = list(self.ac.actor_params()) + list(self.ac.log_std_params())
                    if in_warmup:
                        # M5-F2: 预热期只训 critic——actor 梯度不计算不设置
                        # （grad=None ⇒ Adam 跳过该参数组，参数与动量严格不动）
                        policy_loss_sum, entropy_loss_sum = 0.0, 0.0
                    else:
                        policy_loss_sum, entropy_loss_sum = self._apply_lexicographic_policy_grad(
                            batch, ratio, entropy, buffer, actor_params, ee_advantages=adv_eff
                        )
                        # M5-F0: 锚定梯度注入 lex 路径（M4 时此处缺失 →
                        # BC 盆地自 update 1 起无保护漂移，见 M5 预注册法医结论）
                        if hek_anchor_active:
                            total_hek_anchor_loss += self._anchor_grad_into(
                                actor_params, hek_coef, obs.shape[0]
                            )
                        # M5.2-F6: KL-to-BC 信任域梯度注入（同锚定语义，不参与投影）
                        if klbc_term is not None:
                            total_klbc_loss += self._trust_grad_into(
                                actor_params, klbc_term, self.kl_bc_coef
                            )

                    # 3) 裁剪 + 步进
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if use_cost:
                        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                        self.cost_critic_optimizer.step()

                    policy_loss_item = float(policy_loss_sum)
                    entropy_item = float(-entropy_loss_sum / max(float(self.entropy_coef), 1e-8))
                else:
                    # ---------- 原 PPO 路径 ----------
                    # RCPO 优势分解：约束从 reward 减法迁移到优势通道，
                    # A_eff = A_r - λ_scaled·A_c；非 cost 模式 A_eff = A_r（原行为）
                    if use_cost:
                        adv_eff = adv_eff - float(lambda_scaled) * batch["cost_advantages"]

                    value_loss = nn.functional.mse_loss(values, returns)
                    batch_entropy_scale = batch["entropy_scale"]
                    entropy_loss = -(batch_entropy_scale * entropy).mean()
                    if in_warmup:
                        # M5-F2: 预热期只训 critic（policy/entropy 断开，anchor/PI 同理）
                        loss = self.value_coef * value_loss
                    else:
                        surr1 = ratio * adv_eff
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_eff
                        policy_loss = -torch.min(surr1, surr2).mean()
                        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                        # M5.2-F6: KL-to-BC 信任域惩罚（relu 语义：只罚漂离 BC，不罚改进）
                        if klbc_term is not None:
                            loss = loss + self.kl_bc_coef * klbc_term
                            total_klbc_loss += float(klbc_term.item())

                    # PLPI: delivery-gated power incentive（直达梯度 bypass 噪声 throughput 梯度）
                    pi_loss = None
                    if not in_warmup and pi_coef > 0 and "pi_delivery" in batch:
                        dr = float(batch["pi_delivery"].mean().item())
                        if dr < pi_threshold:
                            deficit = (pi_threshold - dr) / max(pi_threshold, 1e-8)
                            K = actions.shape[1] // 3
                            p_actions = actions[:, K:2 * K]  # p_total 块（tanh 空间 [-1,1]）
                            mean_p = p_actions.mean()
                            # 负号：最小化 → 推高 mean_p（tanh→1 → 高功率）
                            pi_loss = -pi_coef * deficit * mean_p
                            loss = loss + pi_loss
                            total_power_incentive_loss += pi_loss.item()

                    # RCPO: cost critic 回归稠密 cost 折扣回报
                    cost_value_loss = None
                    if use_cost:
                        cost_values_pred = self.cost_critic(obs).squeeze(-1)
                        cost_value_loss = nn.functional.mse_loss(cost_values_pred, batch["cost_returns"])
                        loss = loss + self.cost_critic_coef * cost_value_loss

                    # C3PO: 7维逐SNR点 cost critic + 移动预算中央路径惩罚
                    c3po_value_loss = None
                    c3po_penalty = None
                    if use_c3po:
                        c3po_cost_values_pred = self.ac.evaluate_cost_values(obs)
                        c3po_value_loss = nn.functional.mse_loss(
                            c3po_cost_values_pred, batch["c3po_cost_returns"]
                        )
                        budgets = batch["c3po_budgets"].unsqueeze(0)  # (1, n_points)
                        c3po_penalty = self._c3po_nu() * torch.relu(
                            c3po_cost_values_pred - budgets
                        ).mean()
                        loss = loss + self.c3po_cost_coef * c3po_value_loss + c3po_penalty

                    # HEK-Lite: 衰减锚定损失（把 actor 锚在专家演示的高功率/
                    # 高交付盆地，防止 PPO 初期被拽回低功率吸引子）
                    if hek_anchor_active and not in_warmup:
                        try:
                            n_hek = int(self.hek_obs.shape[0])
                            hidx = torch.randint(
                                0, n_hek, (obs.shape[0],), device=self.hek_obs.device
                            )
                            h_orders = None
                            if getattr(self, "hek_orders", None) is not None:
                                h_orders = self.hek_orders[hidx]
                            h_log_prob, _, _, h_cont_log_prob = self.ac.evaluate_actions(
                                self.hek_obs[hidx], self.hek_act[hidx], h_orders
                            )
                            anchor_loss = -(h_log_prob if h_orders is not None
                                            else h_cont_log_prob).mean()
                            if not (torch.isnan(anchor_loss) or torch.isinf(anchor_loss)):
                                loss = loss + hek_coef * anchor_loss
                                total_hek_anchor_loss += float(anchor_loss.item())
                        except Exception:
                            pass  # 锚定失败不影响主 PPO 更新

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    self.optimizer.zero_grad()
                    if use_cost:
                        self.cost_critic_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if use_cost:
                        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                        self.cost_critic_optimizer.step()

                    policy_loss_item = 0.0 if in_warmup else float(policy_loss.item())
                    entropy_item = float((-entropy_loss).item())

                # KL 估计 / 早停 只用连续动作部分（PL 排列分布的 log_prob 量级大、方差高，
                # 会使 approx_kl 虚高导致每个 update 都早停 → policy 实际无法更新）
                with torch.no_grad():
                    if old_cont_log_probs is not None:
                        cont_log_ratio = (new_cont_log_probs - old_cont_log_probs).clamp(-20.0, 20.0)
                        cont_ratio = torch.exp(cont_log_ratio)
                        approx_kl = ((cont_ratio - 1) - cont_log_ratio).mean().item()
                        clip_fraction = ((cont_ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                    else:
                        # Fallback: 兼容无 cont_log_probs 的旧 buffer
                        clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()

                total_policy_loss += policy_loss_item
                total_value_loss += value_loss.item()
                if cost_value_loss is not None:
                    total_cost_value_loss += cost_value_loss.item()
                if c3po_value_loss is not None:
                    total_c3po_value_loss += c3po_value_loss.item()
                if c3po_penalty is not None:
                    total_c3po_loss += c3po_penalty.item()
                total_entropy += entropy_item
                total_clip_fraction += clip_fraction
                total_approx_kl += approx_kl
                n_updates += 1

                epoch_kl += approx_kl
                epoch_batches += 1

                # v4: 严格 KL 早停：单 batch KL > 1.5 * target 才触发提前结束当前 epoch
                # （单 batch 阈值放宽避免噪声触发，epoch 平均阈值仍为 target_kl）
                if approx_kl > 1.5 * self.target_kl:
                    kl_breached = True
                    break

            # epoch 结束：按平均 KL 决定是否早停后续 epoch
            mean_epoch_kl = epoch_kl / max(epoch_batches, 1)
            if kl_breached or mean_epoch_kl > self.target_kl:
                early_stopped = True
                break

        # 连续 KL 早停统计与 LR 额外衰减
        if early_stopped:
            self._consecutive_kl_stops += 1
            if self._consecutive_kl_stops >= self.kl_stop_streak_threshold:
                self._kl_decay_factor *= self.kl_stop_lr_decay
                self._consecutive_kl_stops = 0
                self._apply_lr()
        else:
            self._consecutive_kl_stops = 0

        # HEK-Lite: 每次 update 递增计数（驱动锚定系数衰减与预热期结束判定）
        self._hek_updates = int(getattr(self, "_hek_updates", 0)) + 1

        n_updates = max(n_updates, 1)
        # M5.4 诊断：band 熵均值 + log_std 均值（后者与采样分布无关，是真坍缩判据）
        ent_bands = [
            (ent_band_sum[k] / ent_band_n[k]) if ent_band_n[k] else 0.0
            for k in range(3)
        ]
        with torch.no_grad():
            log_std_mean = float(torch.cat(
                [p.detach().flatten() for p in self.ac.log_std_params()]).mean())
        # 桶专家模式下 group[0] 是冻结主干(lr=0)，专家头在 group[1]
        lr_actor_report = float(self.optimizer.param_groups[1]['lr']) if self.use_bucket_experts else float(self.optimizer.param_groups[0]['lr'])
        result = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'cost_value_loss': total_cost_value_loss / n_updates,
            'c3po_value_loss': total_c3po_value_loss / n_updates,
            'c3po_penalty': total_c3po_loss / n_updates,
            'power_incentive_loss': total_power_incentive_loss / n_updates,
            'hek_anchor_loss': total_hek_anchor_loss / n_updates,
            'hek_coef': float(hek_coef) if hek_anchor_active else 0.0,
            'kl_bc_loss': total_klbc_loss / n_updates,
            'kl_bc_coef': float(self.kl_bc_coef) if self._bc_ac is not None else 0.0,
            'entropy': total_entropy / n_updates,
            'entropy_lo': ent_bands[0],
            'entropy_mid': ent_bands[1],
            'entropy_hi': ent_bands[2],
            'log_std_mean': log_std_mean,
            'lr_frozen': float(self._lr_frozen),
            'clip_fraction': total_clip_fraction / n_updates,
            'approx_kl': total_approx_kl / n_updates,
            'early_stopped': float(early_stopped),
            'lr_actor': lr_actor_report,
            'kl_decay_factor': float(self._kl_decay_factor),
            'in_warmup': float(in_warmup),
        }
        result.update(self._compute_expert_grad_norms())
        return result


__all__ = ["PPOTrainer"]
