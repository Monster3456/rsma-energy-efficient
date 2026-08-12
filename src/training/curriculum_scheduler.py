"""SNR 课程学习调度器

在训练过程中根据 episode 进度动态调整环境的 SNR 采样范围
（env.snr_min_db / env.snr_max_db），实现从弱信号到全范围的渐进训练。

stages 中的 episode 计数与训练循环中的 episode_count 对齐，
若 resume 训练，则需要使用绝对 episode 数（包含已训练部分）。
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CurriculumStage:
    """课程学习阶段：[start_episode, end_episode) 内固定 SNR 采样范围"""
    start_episode: int
    end_episode: int
    snr_min_db: float
    snr_max_db: float


class SNRCurriculumScheduler:
    """SNR 课程调度器：在训练过程中动态调整 env 的 SNR 采样范围。

    使用方式：
        scheduler = SNRCurriculumScheduler(stages)
        # 每个 episode 开始前
        scheduler.update_env(env, episode_count)
    """

    def __init__(self, stages: List[CurriculumStage]):
        if not stages:
            raise ValueError("CurriculumScheduler 需要至少一个 stage")
        self.stages = sorted(stages, key=lambda s: s.start_episode)
        self._current_stage_idx = 0

    def get_snr_range(self, episode: int) -> Tuple[float, float]:
        """根据 episode 返回 (snr_min_db, snr_max_db)。

        若 episode 在所有 stage 区间之外（如未到第一个 stage 的 start
        或超出最后一个 stage 的 end），返回最近 stage 的范围。
        """
        # episode 早于第一个 stage：使用第一个 stage
        if episode < self.stages[0].start_episode:
            s = self.stages[0]
            return s.snr_min_db, s.snr_max_db
        for stage in self.stages:
            if stage.start_episode <= episode < stage.end_episode:
                return stage.snr_min_db, stage.snr_max_db
        # 超过最后一个 stage：保持最后阶段的范围
        last = self.stages[-1]
        return last.snr_min_db, last.snr_max_db

    def get_stage_index(self, episode: int) -> int:
        """返回当前 episode 所处 stage 的索引（用于日志/统计）"""
        if episode < self.stages[0].start_episode:
            return 0
        for idx, stage in enumerate(self.stages):
            if stage.start_episode <= episode < stage.end_episode:
                return idx
        return len(self.stages) - 1

    def update_env(self, env, episode: int) -> dict:
        """更新 env 的 SNR 采样范围，返回当前状态字典（用于日志）。"""
        snr_min, snr_max = self.get_snr_range(episode)
        env.snr_min_db = snr_min
        env.snr_max_db = snr_max
        stage_idx = self.get_stage_index(episode)
        return {
            'episode': episode,
            'snr_min_db': snr_min,
            'snr_max_db': snr_max,
            'stage_idx': stage_idx,
        }
