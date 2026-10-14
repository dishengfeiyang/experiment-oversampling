"""主实验脚本"""
import sys
import os
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


def _as_list(x):
    if isinstance(x, list):
        return x
    return [x]


def main():
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("configs/experiment_grid.yaml", encoding="utf-8") as f:
        grid = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg["data"])
    split_cfg = SplitConfig(
        test_size=cfg["experiment"]["test_size"],
        val_size=cfg["experiment"]["val_size"],
        fidelity_ratio=cfg["experiment"]["fidelity_ratio"],
    )

    n = data_cfg.n
    k_n = data_cfg.k_n
    D = cfg["synthesizer"]["D"]
    n_repeats = cfg["experiment"]["n_repeats"]
    seed_base = cfg["experiment"]["seed_base"]

    m_max_mode = grid.get("m_max_mode", "k_n")
    m_max_factors = _as_list(grid.get("m_max_factor", 2.0))
    beta_grid = _as_list(grid["beta_grid"])
    epsilon_grid = _as_list(grid["epsilon_grid"])

    alloc_cfg = cfg.get("allocation", {}) or {}
    c_n_cfg = alloc_cfg.get("c_n", None)
    alpha = alloc_cfg.get("alpha", 0.05)
    M = alloc_cfg.get("M", 13)
    B = alloc_cfg.get("B", 1.0)

    if c_n_cfg is None:
        c_n = compute_c_n(alpha=alpha, M=M, B=B)
        print(f"[c_n] config 未指定，自动计算 c_n = {c_n:.6f} "
              f"(alpha={alpha}, M={M}, B={B})")
    else:
        c_n = float(c_n_cfg)
        print(f"[c_n] 使用 config 指定值 c_n = {c_n:.6f}")

    print(f"[CONFIG] n={n}, k_n={k_n}, m_max_mode={m_max_mode}, "
          f"m_max_factor={m_max_factors}")

    method_specs = grid["methods"]
    all_records = []

    for m_max_factor in m_max_factors:
        if m_max_mode == "n":
            m_max = int(m_max_factor * n)
        elif m_max_mode == "absolute":
            m_max = int(m_max_factor)
        else:  # k_n
            m_max = int(m_max_factor * k_n)

        for beta in beta_grid:
            for epsilon in epsilon_grid:
                print(
                    f"\n=== beta={beta}, eps={epsilon}, "
                    f"m_max_mode={m_max_mode}, m_max_factor={m_max_factor}, "
                    f"m_max={m_max}, c_n={c_n:.4f} ==="
                )

                context = {
                    "beta": beta,
                    "epsilon": epsilon,
                    "D": D,
                    "c_n": c_n,
                }
                methods = build_methods_from_grid(method_specs, context)
                print(f"方法数量: {len(methods)}")

                for rep in tqdm(range(n_repeats)):
                    seed = seed_base + rep
                    set_seed(seed)

                    X, y, mu = generate_gaussian_data(data_cfg, seed)
                    splits = nested_split(X, y, split_cfg, seed)
                    splits.mu_true = mu

                    synth_cfg = {
                        "type": "gaussian",
                        "beta": beta,
                        "epsilon": epsilon,
                        "D": D,
                    }
                    clf_cfg = {
                        "C": cfg["classifier"]["C"],
                        "max_iter": cfg["classifier"]["max_iter"],
                        "class_weight": cfg["classifier"].get(
                            "class_weight", "balanced"
                        ),
                    }

                    for method in methods:
                        res = run_single_method(
                            method, splits, synth_cfg, clf_cfg,
                            k_n, n, seed,
                            m_max=m_max, c_n=c_n,
                        )
                        if res is not None:
                            res["beta"] = beta
                            res["epsilon"] = epsilon
                            res["rep"] = rep
                            res["k_n"] = int(k_n)
                            res["n"] = int(n)
                            res["m_max"] = int(m_max)
                            res["m_max_factor"] = m_max_factor
                            res["m_max_mode"] = m_max_mode
                            all_records.append(res)

    df = pd.DataFrame(all_records)
    out_path = Path(cfg["output"]["results_dir"]) / "simulation_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Total records: {len(df)}")


if __name__ == "__main__":
    main()