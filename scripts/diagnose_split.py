"""详细诊断 nested_split 后少数类样本去向"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
from src.data.generator import DataConfig, generate_gaussian_data
from src.data.split import SplitConfig, nested_split
from src.utils.seed import set_seed

with open("configs/default.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("=" * 60)
print("default.yaml 实际内容")
print("=" * 60)
print(yaml.dump(cfg, allow_unicode=True))

data_cfg = DataConfig(**cfg["data"])
print("=" * 60)
print("DataConfig 实际字段")
print("=" * 60)
for k, v in vars(data_cfg).items():
    print(f"  {k} = {v}")

split_cfg = SplitConfig(
    test_size=cfg["experiment"]["test_size"],
    val_size=cfg["experiment"]["val_size"],
    fidelity_ratio=cfg["experiment"]["fidelity_ratio"],
)
print("=" * 60)
print("SplitConfig 实际字段")
print("=" * 60)
for k, v in vars(split_cfg).items():
    print(f"  {k} = {v}")

set_seed(42)
X, y, mu = generate_gaussian_data(data_cfg, 42)
print("=" * 60)
print("生成数据")
print("=" * 60)
print(f"  X.shape      = {X.shape}")
print(f"  y 唯一值      = {np.unique(y)}")
print(f"  y=1 数量      = {int((y == 1).sum())}")
print(f"  y=0 数量      = {int((y == 0).sum())}")

splits = nested_split(X, y, split_cfg, 42)
print("=" * 60)
print("nested_split 结果")
print("=" * 60)
for a in ["X_maj_train", "X_min_train", "X_maj_val", "X_min_val",
          "X_test", "y_test"]:
    if hasattr(splits, a):
        v = getattr(splits, a)
        try:
            print(f"  {a}.shape   = {v.shape}")
        except AttributeError:
            print(f"  {a}         = {v}")
    else:
        print(f"  {a} 不存在")