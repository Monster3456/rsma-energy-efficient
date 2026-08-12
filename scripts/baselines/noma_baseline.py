"""NOMA基线：非正交多址（全公共流）

作为对比方案，使用功率域NOMA，所有功率分配给公共流，alpha_k=1。
等价于上行NOMA，所有用户信号都作为"公共"部分被SIC解码。
SIC解码顺序按信道增益排序。

这个基线展示：全公共流虽然利于SIC消除，但缺乏私有流的空间复用增益。
"""
import numpy as np

from src.config import Config


class NOMABaseline:
    """NOMA基线：alpha_k = 1（全部功率给公共流）

    等价于上行NOMA，所有用户信号都作为"公共"部分被SIC解码。
    SIC解码顺序按信道增益 ||H_k||^2 降序（强用户先解码）。
    """

    def __init__(self, config: Config):
        self.config = config
        self.K = config.system.K
        self.P_k_max = config.system.P_k_max
        # 公平功率预算：P_budget = K * P_k_max（与 PPO 同绪拿到 16W 上限）
        self.P_budget = self.K * self.P_k_max

    def select_action(self, obs: np.ndarray, env) -> np.ndarray:
        """根据当前状态选择动作

        策略：
        - alpha_k = 1.0（全公共流）
        - p_total_k：信道逆水注 p_k ∝ 1/||h_k||^2
          弱信道用户分配更多功率，确保公平性
        - d_ratio_k = 1.0（尽量多发）

        Returns: action (3K,) in [-1, 1] range
        """
        K = self.K

        # --- alpha: 全为1 (全公共流) ---
        # action映射: alpha = (a+1)/2, 要得到1.0 -> a = 1.0
        alpha_action = np.full(K, 1.0)

        # --- p_total: 信道逆水注 ---
        # 获取信道增益
        channel_gains = env._channel_obs.get(
            "channel_gains", np.ones(K)
        )
        # 逆信道增益（弱用户分配更多功率）
        inv_gains = 1.0 / (channel_gains + 1e-8)
        inv_sum = np.sum(inv_gains) + 1e-8
        p_k = self.P_budget * inv_gains / inv_sum
        p_k = np.clip(p_k, 0.0, self.P_k_max)
        # 归一化到 [0, 1]
        p_ratio = p_k / self.P_k_max
        # 映射到 [-1, 1]
        p_action = 2.0 * p_ratio - 1.0

        # --- d_ratio: 尽量多发 ---
        d_action = np.full(K, 1.0)

        action = np.concatenate([alpha_action, p_action, d_action])
        return action.astype(np.float32)

    def evaluate(self, env, n_episodes: int = 100) -> dict:
        """运行n_episodes评估

        Returns: {
            'mean_power': float,
            'mean_queue': float,
            'delivery_ratio': float,
            'queue_stable': bool,
        }
        """
        powers, queues, deliveries = [], [], []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            ep_power, ep_queue = 0.0, 0.0
            for t in range(env.T):
                action = self.select_action(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_power += info['total_power']
                ep_queue += info['mean_queue']
                if truncated:
                    break
            powers.append(ep_power / env.T)
            queues.append(ep_queue / env.T)
            deliveries.append(info['delivery_ratio'])

        mean_queue = float(np.mean(queues))
        return {
            'mean_power': float(np.mean(powers)),
            'mean_queue': mean_queue,
            'delivery_ratio': float(np.mean(deliveries)),
            'queue_stable': mean_queue < env.Q_max * 0.8,
        }
