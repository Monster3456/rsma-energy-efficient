# 增广拉格朗日二次罚 (APPO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SNR 分桶 PPO-Lagrangian 中引入增广拉格朗日二次罚 `λ·c + (ρ/2)·c²`，ρ 按违约未缩减比例乘性增长，打破约束边界梯度平台。

**Architecture:** 仅修改 `src/config.py`（新增超参）与 `src/training/lagrangian.py`（核心算法）；`src/training/main.py` 只做最小 wiring 把新配置传入 `DeliveryConstraint`。保持模型维度与动作空间不变。

**Tech Stack:** Python 3, NumPy, 现有 PPO-RSMA 训练框架。

---

## File Mapping

- `src/config.py`: 在 `ConstraintConfig` 增加 `al_enable`, `al_rho_init`, `al_rho_max`, `al_tau`。
- `src/training/lagrangian.py`: 在 `DeliveryConstraint` 中维护每桶 ρ，实现二次罚与 ρ 增长律。
- `src/training/main.py`: 实例化 `DeliveryConstraint` 时传入新增参数（可选，因为已有默认值，但为让 config.py 生效需 wiring）。

---

## Task 1: Add augmented Lagrangian hyperparameters to config

**Files:**
- Modify: `src/config.py:226-333` (`ConstraintConfig`)

- [ ] **Step 1: Insert AL config fields**

```python
    # ---- Augmented Lagrangian (APPO-style quadratic penalty) ----
    # penalty_b = λ_b * c_b + (ρ_b/2) * c_b^2,  c_b = max(0, threshold - delivery)
    # ρ_b 按违约未缩减比例乘性增长: 若当前 violation > 0 且未下降, ρ *= al_tau
    al_enable: bool = True
    al_rho_init: float = 1.0
    al_rho_max: float = 100.0
    al_tau: float = 1.5
```

- [ ] **Step 2: Validate syntax**

Run: `python3 -m py_compile src/config.py`
Expected: no output (success)

---

## Task 2: Implement augmented Lagrangian penalty in DeliveryConstraint

**Files:**
- Modify: `src/training/lagrangian.py:21-248`

- [ ] **Step 1: Extend __init__ with AL state**

Add parameters to `__init__` signature:
```python
        al_enable: bool = False,
        al_rho_init: float = 1.0,
        al_rho_max: float = 100.0,
        al_tau: float = 1.5,
```

Store:
```python
        self.al_enable = bool(al_enable)
        self.al_rho_max = float(al_rho_max)
        self.al_tau = float(al_tau)
        n = self.n_snr_buckets
        self.rho_val = np.full(n, float(al_rho_init), dtype=np.float64)
```

- [ ] **Step 2: Replace compute_penalty with AL quadratic term when enabled**

In `compute_penalty`:
```python
        violation = max(0.0, self.threshold - float(delivery_ratio))
        if snr_db is None:
            lam = float(np.mean(self.lambda_val))
            rho = float(np.mean(self.rho_val))
        else:
            b = self._bucket(snr_db)
            lam = float(self.lambda_val[b])
            rho = float(self.rho_val[b])
        if self.al_enable:
            return lam * violation + 0.5 * rho * violation * violation
        # fallback: existing exponential barrier
        kappa = 8.0
        return lam * (float(np.exp(kappa * violation)) - 1.0) / kappa
```

- [ ] **Step 3: Update rho per bucket in _update_bucket**

After computing `violation = self.threshold - control_signal` and before lambda update:
```python
        prev_violation = self.prev_violation[b]
        # 增广拉格朗日：若违约未缩减，ρ 乘性增长
        if self.al_enable and violation > 0.0 and violation >= prev_violation:
            self.rho_val[b] = float(np.clip(self.rho_val[b] * self.al_tau, 0.0, self.al_rho_max))
```

- [ ] **Step 4: Include rho in stats**

Add to `get_stats`:
```python
            "rho": float(np.mean(self.rho_val)),
            "rho_per_bucket": [float(x) for x in self.rho_val],
            "al_enable": self.al_enable,
```

- [ ] **Step 5: Validate syntax**

Run: `python3 -m py_compile src/training/lagrangian.py`
Expected: no output (success)

---

## Task 3: Wire new config into main.py

**Files:**
- Modify: `src/training/main.py:402-416`

- [ ] **Step 1: Pass AL params to DeliveryConstraint**

```python
        delivery_constraint = DeliveryConstraint(
            threshold=constraint_cfg.delivery_threshold,
            lr=constraint_cfg.lagrangian_lr,
            init_lambda=constraint_cfg.lagrangian_init,
            lambda_max=constraint_cfg.lagrangian_max,
            warmup_episodes=constraint_cfg.lagrangian_warmup,
            lambda_decay=getattr(constraint_cfg, 'lagrangian_decay', 0.999),
            success_ema_decay=getattr(constraint_cfg, 'sfl_success_ema_decay', 0.99),
            al_enable=getattr(constraint_cfg, 'al_enable', False),
            al_rho_init=getattr(constraint_cfg, 'al_rho_init', 1.0),
            al_rho_max=getattr(constraint_cfg, 'al_rho_max', 100.0),
            al_tau=getattr(constraint_cfg, 'al_tau', 1.5),
        )
```

Update print statement to include `al_enable` and `rho_init/max/tau`.

- [ ] **Step 2: Validate syntax**

Run: `python3 -m py_compile src/training/main.py`
Expected: no output (success)

---

## Task 4: Smoke test

**Files:** none

- [ ] **Step 1: Run 1-episode smoke test**

Run: `python3 scripts/train.py --smoke --device cpu --csi_mode imperfect`
Expected: completes 1 episode without error, prints final delivery/queue/EE

- [ ] **Step 2: Fix any runtime errors**

If smoke fails, diagnose and patch; retry up to 3 times.

---

## Self-Review

1. **Spec coverage**: AL quadratic penalty → Task 2 Step 2; ρ growth → Task 2 Step 3; config exposure → Task 1; wiring → Task 3; smoke → Task 4.
2. **Placeholder scan**: all code blocks contain literal Python; no TBD/TODO.
3. **Type consistency**: `rho_val` is `np.ndarray` of float64, matching `lambda_val`; `al_enable` is bool throughout.
