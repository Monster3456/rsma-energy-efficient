# ============================================================
# RSMA Energy-Efficient 统一命令入口
# 四类常用操作：setup / test / smoke / eval
# 固定 Python 版本要求：3.14（与 requirements.txt 依赖验证环境一致）
# 用法见 `make help` 与 CLAUDE.md「开发命令」一节
# ============================================================

# 固定 Python 版本要求（major.minor），冷环境校验不通过即中止
REQUIRED_PYTHON := 3.14

# 宿主解释器（仅用于创建 venv），可经 PYTHON=xxx 覆盖
PYTHON ?= python3

VENV      := .venv
PY        := $(VENV)/bin/python
PIP       := $(PY) -m pip
DEP_STAMP := $(VENV)/.deps.stamp

# 评估参数（可覆盖：make eval CKPT=... N_EPISODES=...）
CKPT       ?= logs/snr_robust/checkpoints/final.pt
N_EPISODES ?= 50

.PHONY: help check-python setup test smoke eval clean

help:  ## 显示本命令面
	@echo "RSMA Energy-Efficient 统一命令入口（要求 Python $(REQUIRED_PYTHON)）"
	@echo ""
	@echo "  make setup   创建 .venv 并安装依赖（幂等，requirements.txt 变更时自动重装）"
	@echo "  make test    快速单元测试（pytest tests/）"
	@echo "  make smoke   冒烟训练（--smoke 极小配置，验证全流程可运行性）"
	@echo "  make eval    SNR 扫描评估（可覆盖 CKPT= / N_EPISODES=）"
	@echo "  make clean   删除 .venv（不删除 logs/ 训练产物）"
	@echo ""
	@echo "冷环境一条命令示例：make smoke"

check-python:  ## 校验宿主 Python 版本
	@$(PYTHON) -c 'import sys; v=".".join(map(str, sys.version_info[:2])); assert v == "$(REQUIRED_PYTHON)", f"错误：需要 Python $(REQUIRED_PYTHON)，当前为 {v}。请安装匹配版本后重试。"'

$(PY):
	@$(PYTHON) -c 'import sys; v=".".join(map(str, sys.version_info[:2])); assert v == "$(REQUIRED_PYTHON)", f"错误：需要 Python $(REQUIRED_PYTHON)，当前为 {v}。请安装匹配版本后重试。"'
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

$(DEP_STAMP): $(PY) requirements.txt
	$(PIP) install -r requirements.txt
	@touch $(DEP_STAMP)

setup: check-python $(DEP_STAMP)  ## 创建虚拟环境并安装依赖

test: setup  ## 快速单元测试
	$(PY) -m pytest tests/ -v

smoke: setup  ## 冒烟训练（等价于 python scripts/train.py --smoke）
	$(PY) scripts/train.py --smoke --exp_name make_smoke

eval: setup  ## SNR 扫描评估
	@test -f $(CKPT) || { echo "错误：checkpoint 不存在: $(CKPT)（可用 make eval CKPT=<路径> 指定）"; exit 1; }
	$(PY) scripts/evaluate_snr_sweep.py --checkpoint $(CKPT) --n_episodes $(N_EPISODES)

clean:  ## 删除虚拟环境
	rm -rf $(VENV)
