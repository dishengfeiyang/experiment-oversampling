"""按 epsilon 分组分析配对差异，并动态选择最强基线"""
import os
import pandas as pd
from scipy import stats

RAW_PATH = "results/raw/simulation_results.csv"
OUT_PATH = "results/pairwise_by_epsilon.csv"
STRONGEST_PATH = "results/strongest_baseline_by_config.csv"
VS_STRONGEST_PATH = "results/pairwise_vs_strongest.csv"

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(
        f"结果文件不存在: {RAW_PATH}\n"
        f"请先运行 python scripts/run_simulation.py"
    )
if os.path.getsize(RAW_PATH) == 0:
    raise RuntimeError(
        f"结果文件为空: {RAW_PATH}\n"
        f"说明 run_simulation.py 没有写入任何记录。"
    )

df = pd.read_csv(RAW_PATH)
print("总行数:", len(df))
if len(df) == 0:
    raise RuntimeError("结果文件只有表头，没有数据行。")

risk_col = "balanced_excess_risk"
method_col = "method"
beta_col = "beta"
eps_col = "epsilon"
rep_col = "rep"

theory = "theoretical_optimal"
if theory not in df[method_col].unique():
    raise ValueError(f"{theory} 不在结果中")

others = [m for m in df[method_col].unique() if m != theory]

# 动态分组列：k_n, m_max_mode, m_max_factor
extra_group_cols = []
for extra in ["k_n", "m_max_mode", "m_max_factor"]:
    if extra in df.columns:
        extra_group_cols.append(extra)

print("额外分组列:", extra_group_cols)

# ---------- 1. 每个 (beta, epsilon, [extra]) 下的最强基线 ----------
group_cols = [beta_col, eps_col] + extra_group_cols

summary = (
    df.groupby([method_col] + group_cols)[risk_col]
    .mean()
    .reset_index()
)

baseline_df = summary[summary[method_col] != theory].copy()

if len(baseline_df) == 0:
    raise RuntimeError("没有非 theoretical_optimal 的方法可用作基线。")

idx = baseline_df.groupby(group_cols)[risk_col].idxmin()
strongest = baseline_df.loc[idx].rename(
    columns={method_col: "strongest_baseline", risk_col: "baseline_risk"}
)
strongest.to_csv(STRONGEST_PATH, index=False)
print(f"\n已保存 {STRONGEST_PATH}")
print(strongest.to_string(index=False))

# ---------- 2. theoretical_optimal vs 所有其他方法 ----------
rows = []

for eps in sorted(df[eps_col].unique()):
    sub = df[df[eps_col] == eps]
    index_cols = [beta_col, rep_col] + extra_group_cols
    piv = sub.pivot_table(
        index=index_cols, columns=method_col, values=risk_col
    )
    for other in others:
        if other not in piv.columns:
            continue
        tmp = piv[[theory, other]].dropna()
        if len(tmp) < 2:
            continue
        t, p = stats.ttest_rel(tmp[theory], tmp[other])
        row = {
            "epsilon": eps,
            "other": other,
            "mean_diff": (tmp[theory] - tmp[other]).mean(),
            "t": t,
            "p": p,
            "n": len(tmp),
        }
        for extra in extra_group_cols:
            row[extra] = tmp.index.get_level_values(extra)[0] \
                if extra in tmp.index.names else None
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(OUT_PATH, index=False)
print(f"\n已保存 {OUT_PATH}")
print(out.to_string(index=False))

# ---------- 3. theoretical_optimal vs 最强基线 ----------
strongest_rows = []

for _, row in strongest.iterrows():
    beta = row[beta_col]
    eps = row[eps_col]
    baseline = row["strongest_baseline"]

    mask = (df[beta_col] == beta) & (df[eps_col] == eps)
    for extra in extra_group_cols:
        mask &= (df[extra] == row[extra])
    sub = df[mask]

    index_cols = [rep_col] + extra_group_cols
    piv = sub.pivot_table(
        index=index_cols, columns=method_col, values=risk_col
    )
    if theory not in piv.columns or baseline not in piv.columns:
        continue
    tmp = piv[[theory, baseline]].dropna()
    if len(tmp) < 2:
        continue
    t, p = stats.ttest_rel(tmp[theory], tmp[baseline])

    res_row = {
        "beta": beta,
        "epsilon": eps,
        "baseline": baseline,
        "mean_diff": (tmp[theory] - tmp[baseline]).mean(),
        "t": t,
        "p": p,
        "n": len(tmp),
    }
    for extra in extra_group_cols:
        res_row[extra] = row[extra]
    strongest_rows.append(res_row)

strongest_out = pd.DataFrame(strongest_rows)
strongest_out.to_csv(VS_STRONGEST_PATH, index=False)
print(f"\n已保存 {VS_STRONGEST_PATH}")
print(strongest_out.to_string(index=False))

# ---------- 4. 核心判据打印 ----------
print()
print("=" * 70)
print("核心判据：epsilon=0 子集下 theoretical_optimal vs real_only")
print("=" * 70)

key = out[(out["epsilon"] == 0.0) & (out["other"] == "real_only")]
if len(key) > 0:
    if extra_group_cols:
        for _, row in key.iterrows():
            tag = " ".join(
                f"{c}={row[c]}" for c in extra_group_cols
            )
            verdict = (
                "✅ 成立"
                if (row["mean_diff"] < 0 and row["p"] < 0.05)
                else "❌ 不成立"
            )
            print(
                f"[{tag}] mean_diff={row['mean_diff']:.5f} "
                f"p={row['p']:.5f} n={int(row['n'])} -> {verdict}"
            )
    else:
        row = key.iloc[0]
        verdict = (
            "✅ 成立"
            if (row["mean_diff"] < 0 and row["p"] < 0.05)
            else "❌ 不成立"
        )
        print(f"mean_diff = {row['mean_diff']:.5f}")
        print(f"p-value   = {row['p']:.5f}")
        print(f"n         = {int(row['n'])}")
        print(f"判定      = {verdict}")
else:
    print("未找到 epsilon=0 且 other=real_only 的记录。")

# ---------- 5. 与最强基线比较的汇总打印 ----------
print()
print("=" * 70)
print("核心判据：theoretical_optimal vs 最强基线")
print("=" * 70)

if len(strongest_out) > 0:
    for _, row in strongest_out.iterrows():
        tag = f"beta={row['beta']}, eps={row['epsilon']}"
        for extra in extra_group_cols:
            tag += f", {extra}={row[extra]}"
        verdict = (
            "✅ 优于最强基线"
            if (row["mean_diff"] < 0 and row["p"] < 0.05)
            else "❌ 未优于最强基线"
        )
        print(
            f"[{tag}] baseline={row['baseline']}, "
            f"mean_diff={row['mean_diff']:.5f}, "
            f"p={row['p']:.5f} -> {verdict}"
        )
else:
    print("无最强基线比较记录。")