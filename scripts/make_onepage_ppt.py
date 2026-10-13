"""One-page PPT (single 16:9 slide) summarizing the RSMA-EE paper.

Quadrants: problem background / optimization formulation / core difficulties /
proposed algorithm, plus a results footer. Content mirrors docs/paper/main.tex
(Section II model, Section III method, Section V results).

Usage:
  python scripts/make_onepage_ppt.py [--out docs/paper/one_pager.pptx]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

FONT = "Microsoft YaHei"
NAVY = RGBColor(0x1F, 0x4E, 0x79)
BLUE = RGBColor(0x2E, 0x75, 0xB6)
RED = RGBColor(0xC0, 0x00, 0x00)
GREEN = RGBColor(0x54, 0x82, 0x35)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF6, 0xFB)

QUADRANTS = [
    (
        NAVY,
        "问题背景｜过载 RSMA 上行",
        [
            "过载上行：活跃用户 K=24 超过接收天线 Nr=16，空间复用饱和、干扰受限",
            "RSMA：消息分裂为公共流+私有流，接收端 MMSE-SIC，把部分干扰变为可解码速率",
            "现实损伤：不完美 CSI（误差随负载 K 增大）、Poisson 排队到达、SIC 硬件残差",
            "运行硬要求：7 个 SNR 工作点（0–30 dB）上交付率/队列相对 SDMA 基线同时达标",
        ],
    ),
    (
        BLUE,
        "优化建模｜逐点约束下的长期能效最大化",
        [
            "逐槽决策变量：分组 g、公共流功率比 α、功率 p、队列服务比 d、SIC 解码序 π",
            "max  liminf (1/T) Σ_t EE[t]，  EE = R_sum / (Σ_k p_k/η_PA + P_circ)",
            "s.t.  deliv(s) ≥ deliv_SDMA(s)，queue(s) ≤ 1.1·queue_SDMA(s)，∀ s ∈ S（7 点）",
            "      0 ≤ p_k ≤ P_k,max；队列均值有界；π ∈ S_K（合法排列）",
            "R_k = log2(1+SINR_c) + log2(1+SINR_p)；SINR 含未解码干扰 + SIC 残差 β² + CSI 误差 ε²",
        ],
    ),
    (
        RED,
        "核心难点",
        [
            "组合不可微：π 共 K! 种排列，速率对解码序不可微 → 梯度无法直接作用",
            "逐点硬约束：单一策略须 7 点同时满足、不可逐点分解；“平均好”即判失败",
            "过载 × 不完美 CSI：估计误差 σ²/(K·p_pilot) 随 K 增大，SIC 残差沿解码链传播",
            "分数目标 × 队列耦合：EE 为比式且与队列状态耦合 → 需 Dinkelbach + Lyapunov",
            "交付门控使梯度饱和；效应量 ~0.005 淹没于种子噪声，单种子对比不可信",
        ],
    ),
    (
        GREEN,
        "我们的算法｜分层 PPO + 求解器专家蒸馏",
        [
            "双时间尺度分层 PPO：Meta（每 20 槽：用户分组+组级分裂）→ Worker SA-GAT",
            "Worker 逐槽输出：功率份额、队列排泄比、Plackett-Luce SIC 序（Gumbel top-K）",
            "DGLCR 奖励：Dinkelbach 线性化 EE + max-weight 吞吐 + CVaR 交付虚拟队列罚；",
            "   Lagrangian 对偶 + 词典序梯度投影保障硬约束",
            "求解器专家：可微 MMSE-SIC 影子引擎，逐槽 25 步 Adam 精炼功率 logits + 序候选",
            "蒸馏部署：8000 episode BC → SFTP 步冻微调 → guard 逐点选点；推理单前向传播",
        ],
    ),
]

FOOTER = (
    "结果：7/7 逐点门全过 ｜ 交付 0.9496±0.0047（5 seeds） ｜ 30 dB：0.888 vs SDMA 0.664 / NOMA 0.581 / EP 0.601 "
    "｜ 发射功率约减半 ｜ 蒸馏成本 ≈1.2%（专家上界 0.9574 → 策略 0.9519）"
)


def _set_text(tf, lines, size, color, bold=False, space_after=2):
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        run = p.add_run()
        run.text = line
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color


def build(out: str) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # title
    box = slide.shapes.add_textbox(Inches(0.35), Inches(0.18), Inches(12.63), Inches(0.95))
    tf = box.text_frame
    _set_text(
        tf,
        ["过载 RSMA 上行能效在线调度：逐点硬约束下的分层强化学习与求解器专家蒸馏"],
        24,
        NAVY,
        bold=True,
        space_after=4,
    )
    p = tf.add_paragraph()
    run = p.add_run()
    run.text = "K=24 > Nr=16 过载 MU-MIMO 上行 · 不完美 CSI · 每槽联合决策 (功率, 流分裂, SIC 序) · 能效 = R_sum / P_consume"
    run.font.name = FONT
    run.font.size = Pt(12)
    run.font.color.rgb = GREY

    # 2x2 quadrants
    x0, y0 = 0.35, 1.25
    w, h = 6.24, 2.62
    gap = 0.15
    for idx, (color, header, bullets) in enumerate(QUADRANTS):
        x = x0 + (idx % 2) * (w + gap)
        y = y0 + (idx // 2) * (h + gap)
        head = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(0.34))
        head.fill.solid()
        head.fill.fore_color.rgb = color
        head.line.fill.background()
        htf = head.text_frame
        htf.margin_top = Pt(2)
        htf.margin_bottom = Pt(2)
        _set_text(htf, [header], 13, RGBColor(0xFF, 0xFF, 0xFF), bold=True, space_after=0)

        body = slide.shapes.add_textbox(Inches(x + 0.08), Inches(y + 0.38), Inches(w - 0.16), Inches(h - 0.42))
        btf = body.text_frame
        _set_text(btf, ["• " + b for b in bullets], 10.5, RGBColor(0x26, 0x26, 0x26), space_after=3)

    # footer
    foot = slide.shapes.add_shape(1, Inches(0.35), Inches(6.98), Inches(12.63), Inches(0.36))
    foot.fill.solid()
    foot.fill.fore_color.rgb = LIGHT
    foot.line.color.rgb = NAVY
    ftf = foot.text_frame
    ftf.margin_top = Pt(3)
    ftf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _set_text(ftf, [FOOTER], 11, NAVY, bold=True, space_after=0)
    ftf.paragraphs[0].alignment = PP_ALIGN.CENTER

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/paper/one_pager.pptx")
    args = ap.parse_args(argv)
    build(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
