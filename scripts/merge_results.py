"""合并多个 k_n 下的 simulation_results_*.csv 到同一个文件"""
import pandas as pd
from pathlib import Path

K_LIST = [50, 100, 200, 500, 1000, 2000]
RAW_DIR = Path("results/raw")

dfs = []
for k in K_LIST:
    fpath = RAW_DIR / f"simulation_results_k{k}.csv"
    if not fpath.exists():
        print(f"[WARN] 文件不存在，跳过: {fpath}")
        continue
    df = pd.read_csv(fpath)
    df["k_n"] = k
    dfs.append(df)
    print(f"[OK] 读取 {fpath}, {len(df)} 行")

if not dfs:
    raise RuntimeError("没有找到任何 simulation_results_k*.csv 文件")

merged = pd.concat(dfs, ignore_index=True)
out_path = RAW_DIR / "simulation_results.csv"
merged.to_csv(out_path, index=False)
print(f"\n已合并到 {out_path}")
print(f"总行数: {len(merged)}")
print(f"k_n 值: {sorted(merged['k_n'].unique())}")