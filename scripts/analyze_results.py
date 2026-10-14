import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RAW_PATH = "results/raw/simulation_results.csv"
OUT_DIR = "results"
FIG_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"结果文件不存在: {RAW_PATH}")
if os.path.getsize(RAW_PATH) == 0:
    raise RuntimeError(f"结果文件为空: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)
print("总行数:", len(df))
if len(df) == 0:
    raise RuntimeError("结果文件只有表头，没有数据行。")

risk_col = "balanced_excess_risk"
rec_min_col = "minority_recall"
rec_maj_col = "majority_recall"
f1_col = "macro_f1"
div_col = "discrepancy"
s_col = "s_n"
m_col = "m_n"
method_col = "method"
beta_col = "beta"
eps_col = "epsilon"
rep_col = "rep"

# n 用于归一化 m
n = df["n"].iloc[0] if "n" in df.columns else 100000.0

# 动态分组列：method, beta, epsilon，加上 k_n, m_max_mode, m_max_factor
group_cols = [method_col, beta_col, eps_col]
for extra in ["k_n", "m_max_mode", "m_max_factor"]:
    if extra in df.columns:
        group_cols.append(extra)

print("分组列:", group_cols)

summary = df.groupby(group_cols).agg(
    risk_mean=(risk_col, "mean"),
    risk_std=(risk_col, "std"),
    risk_se=(risk_col, lambda x: x.std() / np.sqrt(len(x))),
    rec_min_mean=(rec_min_col, "mean"),
    rec_maj_mean=(rec_maj_col, "mean"),
    f1_mean=(f1_col, "mean"),
    div_mean=(div_col, "mean"),
    s_mean=(s_col, "mean"),
    m_mean=(m_col, "mean"),
    n_rep=(risk_col, "count"),
).reset_index()

summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
print("已保存 results/summary.csv")

# 最强基线：按 (beta, epsilon, k_n, m_max_*) 分组，找非 theoretical 方法中 risk 最小的
baseline_df = summary[summary[method_col] != "theoretical_optimal"].copy()
baseline_keys = [c for c in group_cols if c != method_col]
idx = baseline_df.groupby(baseline_keys)["risk_mean"].idxmin()
strongest = baseline_df.loc[idx].rename(
    columns={method_col: "strongest_baseline", "risk_mean": "baseline_risk_mean"}
)
strongest.to_csv(
    os.path.join(OUT_DIR, "strongest_baseline_by_config.csv"), index=False
)
print("已保存 results/strongest_baseline_by_config.csv")

# 配对 t 检验：theoretical_optimal vs 其他
theory_names = [m for m in df[method_col].unique() if "theoretical" in str(m).lower()]
if theory_names:
    base = theory_names[0]
    piv_index = [beta_col, eps_col, rep_col]
    for extra in ["k_n", "m_max_mode", "m_max_factor"]:
        if extra in df.columns:
            piv_index.insert(0, extra)
    piv = df.pivot_table(index=piv_index, columns=method_col, values=risk_col)
    pairs = []
    for other in piv.columns:
        if other == base:
            continue
        tmp = piv[[base, other]].dropna()
        if len(tmp) > 1:
            t, p = stats.ttest_rel(tmp[base], tmp[other])
            pairs.append({
                "base": base,
                "other": other,
                "mean_diff": (tmp[base] - tmp[other]).mean(),
                "t": t,
                "p": p,
                "n": len(tmp),
            })
    pd.DataFrame(pairs).to_csv(
        os.path.join(OUT_DIR, "pairwise_ttests.csv"), index=False
    )
    print("已保存 results/pairwise_ttests.csv")

# 图 1：risk vs epsilon，按 (beta, k_n) 分面
for beta, sub_beta in summary.groupby(beta_col):
    if "k_n" in sub_beta.columns:
        k_n_values = sorted(sub_beta["k_n"].unique())
    else:
        k_n_values = [None]

    for k_n_val in k_n_values:
        if k_n_val is not None:
            sub = sub_beta[sub_beta["k_n"] == k_n_val]
        else:
            sub = sub_beta

        plt.figure(figsize=(10, 6))
        for method, d in sub.groupby(method_col):
            d = d.sort_values(eps_col)
            plt.errorbar(d[eps_col], d["risk_mean"], yerr=d["risk_se"],
                         marker='o', capsize=3, label=str(method))
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("Balanced excess risk")
        title = f"Risk vs epsilon (beta={beta}"
        if k_n_val is not None:
            title += f", k_n={k_n_val}"
        title += ")"
        plt.title(title)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        if k_n_val is not None:
            fname = f"risk_vs_eps_beta{beta}_k{k_n_val}.png"
        else:
            fname = f"risk_vs_eps_beta{beta}.png"
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=200)
        plt.close()

# 图 2：s*/k_n vs epsilon，按 (beta, k_n) 分面
opt = summary[summary[method_col].astype(str).str.contains("theoretical", case=False)]
if len(opt):
    for (beta, k_n_val), sub in opt.groupby([beta_col, "k_n"]) if "k_n" in opt.columns else opt.groupby(beta_col):
        if isinstance(k_n_val, pd.Series):
            k_n_val = None
        sub = sub.sort_values(eps_col)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(sub[eps_col], sub["s_mean"] / float(k_n_val) if k_n_val is not None else sub["s_mean"], marker='o')
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$s^*/k_n$")
        plt.subplot(1, 2, 2)
        plt.plot(sub[eps_col], sub["m_mean"] / n, marker='o')
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$m^*/n$")
        plt.tight_layout()
        if k_n_val is not None:
            fname = f"optimal_allocation_beta{beta}_k{k_n_val}.png"
        else:
            fname = f"optimal_allocation_beta{beta}.png"
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=200)
        plt.close()

print("完成：results/summary.csv, results/pairwise_ttests.csv, results/figures/")