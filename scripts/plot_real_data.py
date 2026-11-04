"""真实数据实验结果绘图 (Mammography, 同预算对比)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from plot_utils import set_style, savefig, COLORS

set_style()

OUT_DIR = Path("results/real_data")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============ 图 1：学习曲线估计 ============
if (OUT_DIR / "learning_curves.csv").exists():
    lc_df = pd.read_csv(OUT_DIR / "learning_curves.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in lc_df.iterrows():
        try:
            s_used = ast.literal_eval(row["s_grid_used"])
            tv_used = ast.literal_eval(row["tv_estimates"])
            if not isinstance(s_used, list):
                s_used = [s_used]
                tv_used = [tv_used]
        except Exception:
            continue
        if len(s_used) == 0:
            continue
        ax.scatter(s_used, tv_used, alpha=0.3, s=15, color=COLORS["real_only"])
        if len(s_used) >= 2:
            s_fit = np.linspace(min(s_used), max(s_used), 50)
            # 列名为 D, beta, eps
            tv_fit = row["D"] * s_fit ** (-row["beta"]) + row["eps"]
            ax.plot(s_fit, tv_fit, alpha=0.2, linewidth=0.8, color=COLORS["theoretical"])

    ax.set_xlabel("Generator-fitting size $s$")
    ax.set_ylabel("Estimated TV distance $\\widehat{\\Delta}(s)$")
    # 【已删除】ax.set_title("Learning-curve estimation on Mammography") 
    # 标题由 LaTeX 提供
    savefig(fig, str(FIG_DIR / "real_learning_curve.png"))

# ============ 图 2：各方法少数类召回率 ============
if (OUT_DIR / "raw.csv").exists():
    raw = pd.read_csv(OUT_DIR / "raw.csv")
    summary = raw.groupby("method").agg(
        rec_min_mean=("minority_recall", "mean"),
        rec_min_std=("minority_recall", "std"),
        n=("minority_recall", "count"),
    ).reset_index()
    summary["se"] = summary["rec_min_std"] / np.sqrt(summary["n"])

    method_order = ["real_only", "naive_balancing_0.5", "fixed_s_0.1_optimal_m", "theoretical_optimal"]
    method_styles = {
        "real_only": COLORS["real_only"],
        "naive_balancing_0.5": COLORS["naive"],
        "fixed_s_0.1_optimal_m": COLORS["fixed_s"],
        "theoretical_optimal": COLORS["theoretical"],
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs, ys, errs, colors = [], [], [], []
    for m in method_order:
        row = summary[summary["method"] == m]
        if len(row) == 0:
            continue
        xs.append(m)
        ys.append(row["rec_min_mean"].iloc[0])
        errs.append(1.96 * row["se"].iloc[0])
        colors.append(method_styles[m])

    bars = ax.bar(xs, ys, yerr=errs, capsize=5, color=colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Minority recall")
    # 【已删除】ax.set_title("Minority recall on Mammography (mean $\\pm$ 95% CI)")
    # 标题由 LaTeX 提供
    plt.xticks(rotation=15, ha="right")
    savefig(fig, str(FIG_DIR / "real_minority_recall.png"))

# ============ 图 3：BRD 对比 ============
if (OUT_DIR / "raw.csv").exists():
    summary_brd = raw.groupby("method").agg(
        brd_mean=("balanced_excess_risk", "mean"),
        brd_std=("balanced_excess_risk", "std"),
        n=("balanced_excess_risk", "count"),
    ).reset_index()
    summary_brd["se"] = summary_brd["brd_std"] / np.sqrt(summary_brd["n"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs, ys, errs, colors = [], [], [], []
    for m in method_order:
        row = summary_brd[summary_brd["method"] == m]
        if len(row) == 0:
            continue
        xs.append(m)
        ys.append(row["brd_mean"].iloc[0])
        errs.append(1.96 * row["se"].iloc[0])
        colors.append(method_styles[m])

    bars = ax.bar(xs, ys, yerr=errs, capsize=5, color=colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Balanced risk difference (BRD)")
    # 【已删除】ax.set_title("Balanced risk difference on Mammography (mean $\\pm$ 95% CI)")
    # 标题由 LaTeX 提供
    plt.xticks(rotation=15, ha="right")
    savefig(fig, str(FIG_DIR / "real_brd.png"))

print(f"Figures saved to {FIG_DIR}")