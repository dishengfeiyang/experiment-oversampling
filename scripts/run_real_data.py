"""真实数据实验：Mammography + GMM 生成器"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA

from src.data.real_data import load_mammography, split_real_data
from src.estimation.learning_curve import estimate_learning_curve
from src.experiment.real_runner import run_real_method, theoretical_allocation


def main():
    with open("configs/real_data.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_repeats = cfg["experiment"]["n_repeats"]
    seed_base = cfg["experiment"]["seed_base"]
    gmm_components = cfg["generator"].get("n_components", 1)
    m_max_factor = cfg["experiment"]["m_max_factor"]
    minority_subsample = cfg["experiment"].get("minority_subsample", None)

    clf_cfg = {
        "C": cfg["classifier"]["C"],
        "max_iter": cfg["classifier"]["max_iter"],
        "class_weight": cfg["classifier"]["class_weight"],
    }

    records = []
    learning_curves = []
    
    X, y = load_mammography(path="data/mammography.csv")
    
    # PCA 降维
    pca = PCA(n_components=3, random_state=42)
    X = pca.fit_transform(X)

    for rep in tqdm(range(n_repeats), desc="repeats"):
        seed = seed_base + rep

        splits = split_real_data(
            X, y,
            test_size=cfg["experiment"]["test_size"],
            fidelity_ratio=cfg["experiment"].get("fidelity_ratio", 0.25),
            eval_ratio=cfg["experiment"].get("eval_ratio", 0.25),
            minority_subsample=minority_subsample,
            seed=seed,
        )
        k_n = splits.k_n
        n = splits.n
        m_max = int(m_max_factor * k_n)

        # 估计学习曲线
        s_grid = [max(3, int(len(splits.X_min_fidelity) * r)) for r in cfg["estimation"]["s_ratios"]]
        s_grid = [s for s in s_grid if s <= len(splits.X_min_fidelity)]
        if len(s_grid) < 2:
            s_grid = [3, max(4, len(splits.X_min_fidelity))]

        curve = estimate_learning_curve(
            splits.X_min_fidelity, splits.X_min_eval, s_grid,
            n_syn_per_s=cfg["estimation"]["n_syn_per_s"],
            seed=seed,
        )
        curve["rep"] = rep
        learning_curves.append(curve)

        beta_hat = curve["beta"]
        D_hat = curve["D"]
        eps_hat = curve["eps"]

        # 关键修改：强制 m = m_max，只优化 s
        s_opt, m_opt = theoretical_allocation(
            k_n=k_n, n=n, beta=beta_hat, D=D_hat, eps=eps_hat, m_max=m_max,
            force_m_max=True
        )

        # 为了公平，fixed_s_0.1 也使用同样的 m_max
        method_configs = [
            ("real_only", 0, 0, True),
            ("naive_balancing_0.5", int(0.5 * k_n), m_max, True),
            ("fixed_s_0.1_optimal_m", int(0.1 * k_n), m_max, True),
            ("theoretical_optimal", s_opt, m_opt, True),
        ]

        for name, s_n, m_n, use_real in method_configs:
            res = run_real_method(
                name, s_n, m_n, use_real, splits, clf_cfg, seed,
                gmm_components=gmm_components,
            )
            if res is not None:
                records.append({
                    "method": res.method,
                    "s_n": res.s_n,
                    "m_n": res.m_n,
                    "balanced_excess_risk": res.balanced_excess_risk,
                    "minority_recall": res.minority_recall,
                    "majority_recall": res.majority_recall,
                    "macro_f1": res.macro_f1,
                    "rep": rep,
                    "beta_hat": beta_hat,
                    "D_hat": D_hat,
                    "eps_hat": eps_hat,
                    "k_n": k_n,
                    "n": n,
                })

    df = pd.DataFrame(records)
    out_dir = Path("results/real_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "raw.csv", index=False)

    lc_df = pd.DataFrame(learning_curves)
    lc_df.to_csv(out_dir / "learning_curves.csv", index=False)

    summary = df.groupby("method").agg(
        risk_mean=("balanced_excess_risk", "mean"),
        rec_min_mean=("minority_recall", "mean"),
        rec_maj_mean=("majority_recall", "mean"),
        f1_mean=("macro_f1", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()