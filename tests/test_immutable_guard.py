"""不可变路径校验单元测试

覆盖 optimize_loop 代码创新通道在"施工后、接受与提交之前"的不可变路径门禁：
1. find_immutable_violations 纯函数（精确匹配/目录前缀/边界不误伤/空输入）
2. 触碰评估脚本等不可变文件的变更 → 本轮被拒绝、工作区回滚、状态记录拒绝原因
3. 白名单内的正常变更不受影响

不改动现有冒烟测试（run_smoke_test）与回滚机制（_restore_from_backup），
测试通过 monkeypatch 隔离 git/qodercli 等外部依赖。
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import optimize_loop as ol  # noqa: E402

# fast 子集：不可变护栏逻辑校验，轻量且确定性
pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
IMMUTABLE_PATHS = [
    "src/env/channel.py",
    "src/env/rate_calc.py",
    "scripts/evaluate_snr_sweep.py",
]


class _FakeCompleted:
    """subprocess.run 的假返回对象"""

    def __init__(self):
        self.returncode = 0
        self.stdout = "ok"
        self.stderr = ""


class _FakeSubprocess:
    """替换 ol.subprocess：记录所有命令；qodercli 施工时按 side_effect 落盘改动"""

    def __init__(self, on_build=None):
        self.calls = []
        self.on_build = on_build  # 收到 qodercli 命令时调用（模拟施工写文件）

    def run(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if cmd and cmd[0] == "qodercli" and self.on_build:
            self.on_build()
        return _FakeCompleted()


def _base_config(mutable_files):
    """最小可用 config：白名单放行被测文件，forbidden 置空以隔离旧护栏，
    证明拦截来自不可变路径校验本身"""
    return {
        "termination": {
            "snr_points": [],
            "delivery_sdma_ratio": 1.0,
            "queue_sdma_ratio": 1.1,
            "ee_advantage_min": 1.05,
        },
        "code_guard": {
            "mutable_files": mutable_files,
            "forbidden_files": [],
            "rsma_env_physical_markers": [],
            "immutable_paths": list(IMMUTABLE_PATHS),
        },
        "qodercli": {"command": "qodercli", "timeout": 10},
    }


def _base_state():
    return {
        "iteration": 7,
        "current_direction": {"desc": "测试创新方向", "channel": "code_innovation"},
        "failed_directions": [],
        "best_checkpoint": None,
        "current_checkpoint": None,
    }


def _patch_snapshots(monkeypatch, before, after):
    """按调用次序返回快照：第1次=施工前，之后全部=施工后（含回滚阶段）"""
    seq = [before, after]

    def fake_snapshot():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(ol, "_snapshot_worktree", fake_snapshot)


def _run_apply(monkeypatch, tmp_path, state, config, changed_file, tamper_fn,
               smoke_result=(True, "冒烟测试通过")):
    """通用打桩后调用 apply_code_innovation，返回 (ok, reason, fake_sub, smoke_calls)"""
    monkeypatch.setattr(ol, "PROJECT_ROOT", tmp_path)
    fake_sub = _FakeSubprocess(on_build=tamper_fn)
    monkeypatch.setattr(ol, "subprocess", fake_sub)
    _patch_snapshots(monkeypatch, {}, {changed_file: " M"})
    smoke_calls = []

    def fake_smoke(*args, **kwargs):
        smoke_calls.append(1)
        return smoke_result

    monkeypatch.setattr(ol, "run_smoke_test", fake_smoke)
    ok, reason = ol.apply_code_innovation(state, config, {"per_snr": {}}, dry_run=False)
    return ok, reason, fake_sub, smoke_calls


# ---------------------------------------------------------------------- #
# 1. find_immutable_violations 纯函数
# ---------------------------------------------------------------------- #
class TestFindImmutableViolations:

    def test_exact_match_eval_script(self):
        got = ol.find_immutable_violations(
            ["scripts/evaluate_snr_sweep.py"], IMMUTABLE_PATHS)
        assert got == ["scripts/evaluate_snr_sweep.py"]

    def test_exact_match_physics_model(self):
        got = ol.find_immutable_violations(
            ["src/training/ppo.py", "src/env/channel.py"], IMMUTABLE_PATHS)
        assert got == ["src/env/channel.py"]

    def test_file_under_immutable_dir(self):
        got = ol.find_immutable_violations(
            ["src/env/physics/new_model.py"], ["src/env/physics/"])
        assert got == ["src/env/physics/new_model.py"]

    def test_similar_prefix_not_matched(self):
        # 文件名仅前缀相似但非同一文件/非子路径，不得误伤
        got = ol.find_immutable_violations(
            ["scripts/evaluate_snr_sweep_v2.py", "src/env/channel_utils.py"],
            IMMUTABLE_PATHS)
        assert got == []

    def test_whitelisted_change_not_matched(self):
        got = ol.find_immutable_violations(
            ["src/training/ppo.py", "src/models/actor_critic.py", "scripts/train.py"],
            IMMUTABLE_PATHS)
        assert got == []

    def test_empty_inputs(self):
        assert ol.find_immutable_violations([], IMMUTABLE_PATHS) == []
        assert ol.find_immutable_violations(["src/training/ppo.py"], []) == []

    def test_trailing_slash_entry(self):
        got = ol.find_immutable_violations(
            ["docs/spec/x.md"], ["docs/spec/", "docs/spec"])
        assert got == ["docs/spec/x.md"]


# ---------------------------------------------------------------------- #
# 2. 触碰不可变路径 → 拒绝本轮 + 回滚 + 状态记录
# ---------------------------------------------------------------------- #
class TestImmutableGateRejects:

    @pytest.mark.parametrize("violated", [
        "scripts/evaluate_snr_sweep.py",  # 评估脚本
        "src/env/channel.py",             # 环境物理模型
    ])
    def test_touch_immutable_rejected_rolled_back_and_recorded(
            self, monkeypatch, tmp_path, violated):
        # 沙盒内准备被触碰的不可变文件
        target = tmp_path / violated
        target.parent.mkdir(parents=True, exist_ok=True)
        original = "# 原始内容（不可变）\n"
        target.write_text(original)
        tampered = "# 被施工篡改的内容\n"

        state = _base_state()
        # 白名单包含该文件：证明拦截来自不可变校验层，而非旧白名单护栏
        config = _base_config(mutable_files=[violated])

        ok, reason, fake_sub, smoke_calls = _run_apply(
            monkeypatch, tmp_path, state, config,
            changed_file=violated,
            tamper_fn=lambda: target.write_text(tampered),
        )

        # 本轮被拒绝
        assert ok is False
        assert reason.startswith("IMMUTABLE:")
        assert violated in reason
        # 工作区已回滚：文件恢复施工前内容
        assert target.read_text() == original
        # 状态文件记录拒绝原因
        rejections = state.get("immutable_rejections", [])
        assert len(rejections) == 1
        assert rejections[0]["violated_files"] == [violated]
        assert rejections[0]["iter"] == 7
        assert violated in rejections[0]["reason"]
        # 拒绝发生在接受与提交之前：未跑冒烟、未执行任何 git commit
        assert smoke_calls == []
        assert not any("commit" in c for c in fake_sub.calls)


# ---------------------------------------------------------------------- #
# 3. 白名单内正常变更不受影响
# ---------------------------------------------------------------------- #
class TestWhitelistedChangeUnaffected:

    def test_whitelisted_change_passes_gate(self, monkeypatch, tmp_path):
        ppo = tmp_path / "src" / "training" / "ppo.py"
        ppo.parent.mkdir(parents=True, exist_ok=True)
        ppo.write_text("# ppo 原始实现\n")
        modified = "# ppo 创新改动\n"

        state = _base_state()
        config = _base_config(mutable_files=["src/training/"])

        ok, reason, fake_sub, smoke_calls = _run_apply(
            monkeypatch, tmp_path, state, config,
            changed_file="src/training/ppo.py",
            tamper_fn=lambda: ppo.write_text(modified),
        )

        assert ok is True
        assert "施工成功" in reason
        # 白名单改动保留（未被回滚），冒烟测试照常执行一次
        assert ppo.read_text() == modified
        assert smoke_calls == [1]
        assert "immutable_rejections" not in state
