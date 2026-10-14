"""Exp 1: 计算 J_n(s, m) 在固定 m 下随 s 的变化，验证严格凸性"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.generator import DataConfig
from src.allocation.objective import compute_J, compute_c_n

def main():
    with open("configs/exp1_convexity.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n = cfg["data"]["n"]
    k_n = cfg["data"]["k_n"]
    D = cfg["synthesizer"]["D"]
    c_n = cfg["allocation"]["c_n"] or compute_c_n(
        alpha=cfg["allocation"]["alpha"],
        M=cfg["allocation"]["M"],
        B=cfg["allocation"]["B"],
    )
    m_fixed = cfg["exp1"]["m_fixed"]
    beta = cfg["exp1"]["beta"]
    epsilon = cfg["exp1"]["epsilon"]
    s_grid = cfg["exp1"]["s_grid"]

    # 权重 balanced
    w0, w1 = 0.5, 0.5

    rows = []
    for s in s_grid:
        if s == 0:
            m = 0
        else:
            m = m_fixed
        J = compute_J(s, m, k_n, n, beta, epsilon, D, c_n, w0, w1)
        rows.append({"s": s, "m": m, "J": J})

    df = pd.DataFrame(rows)
    out_dir = Path("results/exp1")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "J_curve.csv", index=False)
    print(f"Saved {out_dir / 'J_curve.csv'}")
    print(df.to_string(index=False))

    # 找连续最优和离散最优
    J_values = df["J"].values
    s_values = df["s"].values
    s_star = s_values[np.argmin(J_values)]
    print(f"\n离散最优 s* = {s_star}, J* = {J_values.min():.6f}")
    print(f"J(0,0) = {J_values[0]:.6f}")

if __name__ == "__main__":
    main()