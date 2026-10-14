"""Exp 4: m = rho k_n, beta = 0.8, 扫描 epsilon"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.data.generator import DataConfig, generate_gaussian_data
from src.data.split import SplitConfig, nested_split
from src.methods.registry import build_methods_from_grid
from src.experiment.runner import run_single_method
from src.utils.seed import set_seed
from src.allocation.objective import compute_c_n

def main():
    with open("configs/exp4_epsilon.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n = cfg["data"]["n"]
    k_n = cfg["data"]["k_n"]
    D = cfg["synthesizer"]["D"]
    rho = cfg["exp4"]["rho"]
    m_max = int(rho * k_n)
    beta = cfg["exp4"]["beta"]
    c_n = compute_c_n(
        alpha=cfg["allocation"]["alpha"],
        M=cfg["allocation"]["M"],
        B=cfg["allocation"]["B"],
    )

    split_cfg = SplitConfig(
        test_size=cfg["experiment"]["test_size"],
        val_size=cfg["experiment"]["val_size"],
        fidelity_ratio=cfg["experiment"]["fidelity_ratio"],
    )

    method_specs = [
        {"name": "real_only", "type": "real_only"},
        {"name": "theoretical_optimal", "type": "theoretical_optimal"},
        {"name": "theoretical_optimal_forced_m", "type": "theoretical_optimal",
         "params": {"force_m_max": True}},
        {"name": "naive_balancing_0.5", "type": "naive_balancing",
         "params": {"tau": 0.5}},
    ]

    all_records = []
    for epsilon in tqdm(cfg["exp4"]["epsilon_grid"], desc="epsilon"):
        for rep in range(cfg["experiment"]["n_repeats"]):
            seed = cfg["experiment"]["seed_base"] + rep
            set_seed(seed)
            data_cfg = DataConfig(n=n, k_n=k_n, d=cfg["data"]["d"], mu=cfg["data"]["mu"])
            X, y, mu = generate_gaussian_data(data_cfg, seed)
            splits = nested_split(X, y, split_cfg, seed)
            splits.mu_true = mu

            synth_cfg = {"type": "gaussian", "beta": beta, "epsilon": epsilon, "D": D}
            clf_cfg = {
                "C": cfg["classifier"]["C"],
                "max_iter": cfg["classifier"]["max_iter"],
                "class_weight": cfg["classifier"]["class_weight"],
            }

            context = {"beta": beta, "epsilon": epsilon, "D": D, "c_n": c_n}
            methods = build_methods_from_grid(method_specs, context)

            for method in methods:
                res = run_single_method(method, splits, synth_cfg, clf_cfg,
                                         k_n, n, seed, m_max=m_max, c_n=c_n)
                if res is not None:
                    res.update({"beta": beta, "epsilon": epsilon, "rep": rep, "k_n": k_n})
                    all_records.append(res)

    df = pd.DataFrame(all_records)
    out_dir = Path("results/exp4")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "raw.csv", index=False)
    print(f"Saved {out_dir / 'raw.csv'}, {len(df)} rows")

if __name__ == "__main__":
    main()