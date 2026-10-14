"""Exp 1: 绘制 J(s, m) vs s 的凸性图（y 轴截断版）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import set_style, savefig, COLORS

set_style()

df = pd.read_csv("results/exp1/J_curve.csv")
s = df["s"].values
J = df["J"].values

s_star = s[np.argmin(J)]
J_star = J.min()
J0 = J[0]

# 只画 s >= 2，避免 s=1 处的尖峰压扁其余曲线
mask = s >= 2
s_plot = s[mask]
J_plot = J[mask]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(s_plot, J_plot, color=COLORS['theoretical'], marker='o', markersize=5,
        label=r'$J_n(s, m)$ at $m = 200$')
ax.axhline(J0, color=COLORS['real_only'], linestyle='--', linewidth=1.4,
           label=r'$J_n(0,0)$ (real-only)')
ax.axvline(s_star, color=COLORS['theory_line'], linestyle=':', linewidth=1.4)
ax.scatter([s_star], [J_star], color=COLORS['theory_line'], s=120, zorder=5,
           edgecolor='white', linewidth=1.5,
           label=f'Discrete optimum $s^* = {s_star}$')
ax.annotate(f'$J^* = {J_star:.4f}$',
            xy=(s_star, J_star), xytext=(s_star + 8, J_star - 0.008),
            fontsize=10, color=COLORS['theory_line'],
            arrowprops=dict(arrowstyle='->', color=COLORS['theory_line'], lw=1.2))

ax.set_xlabel(r'Number of generator-fitting observations $s$')
ax.set_ylabel(r'Allocation objective $J_n(s, m)$')
ax.set_title(r'Strict convexity and discrete optimum ($m = 200$ fixed)')
ax.set_ylim(0.155, 0.185)
ax.legend(loc='upper right', framealpha=0.95)
savefig(fig, "results/figures/exp1_convexity.png")