"""结果可视化"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_phase_transition(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, eps in zip(axes, sorted(df["epsilon"].unique())):
        sub = df[df["epsilon"] == eps]
        for method in sub["method"].unique():
            data = sub[sub["method"] == method].groupby("beta")["balanced_excess_risk"].mean()
            ax.plot(data.index, data.values, marker="o", label=method, markersize=4)
        ax.set_title(f"eps = {eps}")
        ax.set_xlabel("beta")
        ax.set_ylabel("Balanced Excess Risk")
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "phase_transition.png", dpi=150)
    plt.close()


def plot_recall(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, eps in zip(axes, sorted(df["epsilon"].unique())):
        sub = df[df["epsilon"] == eps]
        for method in sub["method"].unique():
            data = sub[sub["method"] == method].groupby("beta")["minority_recall"].mean()
            ax.plot(data.index, data.values, marker="o", label=method, markersize=4)
        ax.set_title(f"eps = {eps}")
        ax.set_xlabel("beta")
        ax.set_ylabel("Minority Recall")
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "minority_recall.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("results/raw/simulation_results.csv")
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plot_phase_transition(df, "results/figures")
    plot_recall(df, "results/figures")
    print("Figures saved to results/figures/")
