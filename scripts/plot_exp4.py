"""Exp 4: s*/k_n 和 risk 随 epsilon 的变化（含退化标注）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import set_style, savefig, COLORS

set_style()

df = pd.read_csv("results/exp4/raw.csv")

summary_s = df[df["method"] == "theoretical_optimal_forced_m"].groupby("epsilon").agg(
    s_mean=("s_n", "mean"),
    s_std=("s_n", "std"),
    n=("s_n", "count"),
).reset_index()
summary_s["s_over_k"] = summary_s["s_mean"] / 1000.0
summary_s["s_se"] = summary_s["s_std"] / np.sqrt(summary_s["n"]) / 1000.0

summary_r = df.groupby(["epsilon", "method"])["balanced_excess_risk"].agg(
    risk_mean="mean", risk_std="std", n="count"
).reset_index()
summary_r["risk_se"] = summary_r["risk_std"] / np.sqrt(summary_r["n"])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

# 左图：s*/k_n vs epsilon
ax = axes[0]
ax.errorbar(summary_s["epsilon"], summary_s["s_over_k"],
            yerr=1.96 * summary_s["s_se"],
            color=COLORS['theoretical'], marker='o', linewidth=1.8,
            markersize=6, capsize=3)
ax.set_xscale('symlog', linthresh=0.01)
ax.set_xlabel(r'Error floor $\varepsilon_n$')
ax.set_ylabel(r'$s^* / k_n$')
ax.set_title(r'Split fraction decreases with $\varepsilon_n$')
ax.grid(True, alpha=0.25, linestyle='--')

# 加一个注释：epsilon 大时理论最优接近 0
ax.annotate(r'$s^*/k_n \to 0$ as $\varepsilon_n$ grows',
            xy=(1.0, summary_s["s_over_k"].iloc[-1]),
            xytext=(0.1, 0.28),
            fontsize=10, color=COLORS['theoretical'],
            arrowprops=dict(arrowstyle='->', color=COLORS['theoretical'], lw=1.2))

# 右图：risk vs epsilon
ax = axes[1]
method_order = [
    ("real_only",                    COLORS['real_only'],   's', 'real_only'),
    ("naive_balancing_0.5",          COLORS['naive'],       '^', 'naive_balancing_0.5'),
    ("theoretical_optimal",          COLORS['fixed_s'],     'D', 'theoretical_optimal'),
    ("theoretical_optimal_forced_m", COLORS['theoretical'], 'o', 'theoretical_optimal_forced_m'),
]
for method, color, marker, label in method_order:
    sub = summary_r[summary_r["method"] == method]
    ax.errorbar(sub["epsilon"], sub["risk_mean"], yerr=1.96 * sub["risk_se"],
                color=color, marker=marker, linewidth=1.8, markersize=6,
                capsize=3, label=label)

# 退化标注
ax.annotate('theoretical_optimal\ndegenerates to real_only',
            xy=(0.2, -0.10), xytext=(0.01, -0.135),
            fontsize=9, color=COLORS['fixed_s'],
            arrowprops=dict(arrowstyle='->', color=COLORS['fixed_s'], lw=1.0),
            ha='left')

ax.set_xscale('symlog', linthresh=0.01)
ax.set_xlabel(r'Error floor $\varepsilon_n$')
ax.set_ylabel(r'Balanced excess risk')
ax.set_title(r'Realized risk vs $\varepsilon_n$')
ax.grid(True, alpha=0.25, linestyle='--')
ax.legend(fontsize=8, loc='lower right', framealpha=0.95)

fig.tight_layout()
savefig(fig, "results/figures/exp4_epsilon.png")