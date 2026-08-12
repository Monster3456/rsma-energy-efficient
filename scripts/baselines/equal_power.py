"""等功率基线：所有用户平均分配功率

作为最简单的对比方案，所有用户使用固定策略：
- alpha_k = 0.5（公共/私有各一半）
- p_total_k = P_max / 2（固定半功率）
- d_ratio_k = 1.0（尽量多发）

这是最简单的基线，不做任何自适应优化。
展示：不考虑信道和队列状态的固定策略有多大损失。
"""
import numpy as np

from src.config import Config


class EqualPowerBaseline:
    """等功率基线：所有用户固定使用 P_max/2

    alpha_k = 0.5（公共/私有各一半）
    p_total_k = P_max / 2（固定半功率）
    d_ratio_k = 1.0（尽量多发）
    """

    def __init__(self, config: Config):
        self.config = config
        self.K = config.system.K
        self.P_k_max = config.system.P_k_max

    def select_action(self, obs: np.ndarray, env) -> np.ndarray:
        """返回固定动作

        策略完全静态，不依赖任何状态信息：
        - alpha = 0.5 -> action = 0.0
        - p_total = P_max/2 -> ratio = 0.5 -> action = 0.0
        - d_ratio = 1.0 -> action = 1.0

        Returns: action (3K,) in [-1, 1] range
        """
        K = self.K

        # alpha = 0.5: (a+1)/2 = 0.5 -> a = 0.0
        alpha_action = np.full(K, 0.0)

        # p_total = P_max/2: ratio = 0.5 -> a = 2*0.5 - 1 = 0.0
        p_action = np.full(K, 0.0)

        # d_ratio = 1.0: a = 1.0
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
