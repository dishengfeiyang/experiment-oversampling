"""Exp 3: s*/k_n vs beta 相变图"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import set_style, savefig, COLORS

set_style()

df = pd.read_csv("results/exp3/raw.csv")
sub = df[df["method"] == "theoretical_optimal_forced_m"]

summary = sub.groupby("beta").agg(
    s_mean=("s_n", "mean"),
    s_std=("s_n", "std"),
    n=("s_n", "count"),
).reset_index()
summary["s_over_k"] = summary["s_mean"] / 1000.0
summary["s_se"] = summary["s_std"] / np.sqrt(summary["n"]) / 1000.0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# 左图：s*/k_n vs beta
rho = 1.0
beta_arr = np.linspace(0.05, 1.3, 200)
theory_tau = np.minimum(1.0, beta_arr * (1 + rho) / (1 + beta_arr))

ax = axes[0]
ax.errorbar(summary["beta"], summary["s_over_k"], yerr=1.96 * summary["s_se"],
            color=COLORS['theoretical'], marker='o', label='Experiment')
ax.plot(beta_arr, theory_tau, '--', color=COLORS['theory_line'],
        label=r'Theory $\tau_\beta$')
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.7)
ax.axvspan(0.05, 0.5, alpha=0.08, color=COLORS['shade'])
ax.axvspan(0.5, 1.3, alpha=0.08, color=COLORS['naive'])
ax.text(0.3, 0.85, r'$\beta < 1/2$', fontsize=11, ha='center')
ax.text(0.85, 0.85, r'$\beta > 1/2$', fontsize=11, ha='center')
ax.set_xlabel(r'Learning exponent $\beta$')
ax.set_ylabel(r'$s^* / k_n$')
ax.set_title(r'Phase transition at $\beta = 1/2$')
ax.legend(loc='lower left')
ax.set_ylim(-0.05, 1.05)

# 右图：log s* vs log k_n
# 重新整理数据，按 beta 分组
grouped = sub.groupby("beta")["s_n"].mean()
beta_vals = sorted(grouped.index)
# 这里需要重跑不同 k_n，但 exp3 只跑了一个 k_n，所以只画一个点
# 改为画 s* 随 beta 的对数，并叠加两条理论线
ax = axes[1]
ax.semilogy(summary["beta"], summary["s_mean"], marker='o',
            color=COLORS['theoretical'], label=r'$s^*$ (experiment)')
# 叠加 k_n^{-beta} 的参考线（归一化）
ref = 1000.0 * (1000.0 ** (-beta_arr))
ref = ref / ref[0] * summary["s_mean"].iloc[0]
ax.semilogy(beta_arr, ref, '--', color=COLORS['fixed_s'], label=r'$k_n^{-\beta}$ ref')
ax.set_xlabel(r'Learning exponent $\beta$')
ax.set_ylabel(r'$s^*$')
ax.set_title(r'Log-scale behavior of $s^*$')
ax.legend()

fig.tight_layout()
savefig(fig, "results/figures/exp3_phase_transition.png")