"""Exp 2: 双对数尺度图（理论虚线区分版）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import set_style, savefig, COLORS

set_style()

df = pd.read_csv("results/exp2/raw.csv")
df = df[df["method"] == "theoretical_optimal_forced_m"]

# 每条 beta 用独立的颜色和 marker，理论虚线用独立的 style
BETA_STYLES = {
    0.5: {'color': '#1f77b4', 'marker': 'o', 'dash': (0, (5, 5)),      'label': r'$\beta = 0.5$'},
    0.8: {'color': '#ff7f0e', 'marker': 's', 'dash': (0, (3, 3, 1, 3)),'label': r'$\beta = 0.8$'},
    1.2: {'color': '#2ca02c', 'marker': 'D', 'dash': (0, (1, 2)),      'label': r'$\beta = 1.2$'},
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

for eps_mode, ax in zip(["zero", "sqrt_k"], axes):
    sub = df[df["epsilon_mode"] == eps_mode]

    for beta, style in BETA_STYLES.items():
        s_beta = sub[sub["beta"] == beta].groupby("k_n")["s_n"].mean()
        if len(s_beta) == 0:
            continue
        k_arr = np.array(s_beta.index, dtype=float)
        s_arr = s_beta.values

        ax.loglog(k_arr, s_arr, marker=style['marker'], color=style['color'],
                  linewidth=1.8, markersize=6, label=style['label'])

        # 理论斜率参考线
        if eps_mode == "zero":
            slope = 1.0 / (2.0 * (beta + 1.0))
        else:
            slope = 1.0 / (beta + 1.0)
        ref = s_arr[0] * (k_arr / k_arr[0]) ** slope
        ax.loglog(k_arr, ref, linestyle=style['dash'], color=style['color'],
                  linewidth=1.2, alpha=0.65)

    ax.set_xlabel(r'Minority count $k_n$')
    ax.set_ylabel(r'Optimal split $s_n$')
    title = r'$\varepsilon_n = 0$' if eps_mode == "zero" else r'$\varepsilon_n = 10\sqrt{k_n}$'
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9, loc='upper left')

# 加一个说明：虚线是理论斜率
fig.text(0.5, -0.02,
         'Dashed lines show theoretical reference slopes '
         r'$s_n \propto k_n^{1/[2(\beta+1)]}$ (left) and $s_n \propto k_n^{1/(\beta+1)}$ (right).',
         ha='center', fontsize=9, color='#444444')

fig.suptitle(r'Asymptotic scale of $s_n$ at fixed $m = 200$', fontsize=13, y=1.02)
fig.tight_layout()
savefig(fig, "results/figures/exp2_asymptotic.png")