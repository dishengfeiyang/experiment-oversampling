"""Exp 6: 理论最优 vs 基线的风险对比（横轴刻度改进版）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import set_style, savefig, COLORS

set_style()

df = pd.read_csv("results/exp6/raw.csv")

summary = df.groupby(["m_max", "method"])["balanced_excess_risk"].agg(
    risk_mean="mean", risk_std="std", n="count"
).reset_index()
summary["risk_se"] = summary["risk_std"] / np.sqrt(summary["n"])

# 横轴刻度：手动指定位置和标签
M_TICKS = [0, 50, 100, 200, 500, 1000, 2000]
M_LABELS = ['0', '50', '100', '200', '500', '1k', '2k']

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

# 左图：risk vs m
ax = axes[0]
method_styles = [
    ("real_only",                    COLORS['real_only'],   's', 'real_only'),
    ("naive_balancing_0.5",          COLORS['naive'],       '^', 'naive_balancing_0.5'),
    ("fixed_s_0.1_optimal_m",        COLORS['fixed_s'],     'D', 'fixed_s_0.1_optimal_m'),
    ("theoretical_optimal_forced_m", COLORS['theoretical'], 'o', 'theoretical_optimal_forced_m'),
]
for method, color, marker, label in method_styles:
    sub = summary[summary["method"] == method].sort_values("m_max")
    ax.errorbar(sub["m_max"], sub["risk_mean"], yerr=1.96 * sub["risk_se"],
                color=color, marker=marker, linewidth=1.8, markersize=6,
                capsize=3, label=label)

ax.set_xscale('symlog', linthresh=50, linscale=0.5)
ax.set_xticks(M_TICKS)
ax.set_xticklabels(M_LABELS)
ax.set_xlabel(r'Synthetic volume $m_{\max}$')
ax.set_ylabel(r'Balanced excess risk')
ax.set_title(r'Risk vs synthetic volume')
ax.grid(True, alpha=0.25, linestyle='--')
ax.legend(fontsize=8, loc='lower left', framealpha=0.95)

# 右图：改善量
ax = axes[1]
pivot = summary.pivot(index="m_max", columns="method", values="risk_mean")
baselines = [
    ("real_only",             COLORS['real_only'], 's', 'vs real_only'),
    ("naive_balancing_0.5",   COLORS['naive'],     '^', 'vs naive_balancing_0.5'),
    ("fixed_s_0.1_optimal_m", COLORS['fixed_s'],   'D', 'vs fixed_s_0.1_optimal_m'),
]
for baseline, color, marker, label in baselines:
    delta = pivot[baseline] - pivot["theoretical_optimal_forced_m"]
    ax.plot(delta.index, delta.values, color=color, marker=marker,
            linewidth=1.8, markersize=6, label=label)

ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax.set_xscale('symlog', linthresh=50, linscale=0.5)
ax.set_xticks(M_TICKS)
ax.set_xticklabels(M_LABELS)
ax.set_xlabel(r'Synthetic volume $m_{\max}$')
ax.set_ylabel(r'Risk improvement (baseline $-$ theoretical)')
ax.set_title(r'Improvement over baselines')
ax.grid(True, alpha=0.25, linestyle='--')
ax.legend(fontsize=8, loc='upper left', framealpha=0.95)

fig.tight_layout()
savefig(fig, "results/figures/exp6_comparison.png")