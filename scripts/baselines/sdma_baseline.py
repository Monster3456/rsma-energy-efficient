"""SDMA基线：纯空分复用（无速率分割）

作为对比方案，所有功率分配给私有流，alpha_k=0。
等价于传统上行MIMO，无RSMA分层。
功率策略：基于Lyapunov漂移的启发式（队列长的用户分配更多功率）。

这个基线展示：没有公共流时，队列稳定性需要更高功率。
"""
import numpy as np

from src.config import Config


class SDMABaseline:
    """SDMA基线：alpha_k = 0（全部功率给私有流）

    等价于传统上行MIMO，无RSMA分层。
    功率策略：基于Lyapunov漂移的启发式（队列加权功率分配）。
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
        - alpha_k = 0（无公共流）
        - p_total_k：基于Lyapunov漂移的启发式
          p_k = min(P_max, P_budget * Q_k / sum(Q_j))
          队列越长的用户分配越多功率
        - d_ratio_k = 1.0（尽量多发）

        Returns: action (3K,) in [-1, 1] range
        """
        K = self.K

        # 从环境获取队列长度
        queue = env.traffic.queue_lengths.copy()

        # --- alpha: 全为0 (无公共流) ---
        # action映射: alpha = (a+1)/2, 要得到0 -> a = -1
        alpha_action = np.full(K, -1.0)

        # --- p_total: 基于队列加权的Lyapunov启发式 ---
        q_sum = np.sum(queue) + 1e-8
        p_k = self.P_budget * queue / q_sum  # 队列越长功率越大
        p_k = np.clip(p_k, 0.0, self.P_k_max)
        # 归一化到 [0, 1] (相对于 P_k_max)
        p_ratio = p_k / self.P_k_max
        # 映射到 [-1, 1]: a = 2*ratio - 1
        p_action = 2.0 * p_ratio - 1.0

        # --- d_ratio: 尽量多发 = 1.0 ---
        # action映射: d = (a+1)/2, 要得到1.0 -> a = 1.0
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
