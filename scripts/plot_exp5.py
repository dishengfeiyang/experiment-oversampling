"""Exp 5: s*/k_n 随 k_n 的转折"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import set_style, savefig, COLORS

set_style()

df = pd.read_csv("results/exp5/raw.csv")
sub = df[df["method"] == "theoretical_optimal_forced_m"]

fig, ax = plt.subplots(figsize=(6.5, 4.5))

for eps_mode, color, marker in [
    ("zero", COLORS['theoretical'], 'o'),
    ("n_inv_sqrt", COLORS['fixed_s'], 'D'),
]:
    s = sub[sub["epsilon_mode"] == eps_mode].groupby("k_n").agg(
        s_mean=("s_n", "mean"),
        s_std=("s_n", "std"),
        n=("s_n", "count"),
    ).reset_index()
    s["s_over_k"] = s["s_mean"] / s["k_n"]
    s["s_se"] = s["s_std"] / np.sqrt(s["n"]) / s["k_n"]

    label = r'$\varepsilon_n = 0$' if eps_mode == "zero" else r'$\varepsilon_n = n^{-1/2}$'
    ax.errorbar(s["k_n"], s["s_over_k"], yerr=1.96 * s["s_se"],
                color=color, marker=marker, label=label)

ax.axvline(1450, color='gray', linestyle=':', alpha=0.7,
           label=r'Theoretical $u_n \approx 1450$')
ax.set_xscale('log')
ax.set_xlabel(r'Minority count $k_n$')
ax.set_ylabel(r'$s^* / k_n$')
ax.set_title(r'Majority-scale generation ($m = 0.05 n$)')
ax.legend(loc='lower left')
ax.set_ylim(-0.05, 1.05)

savefig(fig, "results/figures/exp5_majority_scale.png")