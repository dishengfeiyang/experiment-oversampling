"""嵌套划分"""
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class SplitConfig:
    test_size: int = 5000
    val_size: int = 2000
    fidelity_ratio: float = 0.1


@dataclass
class DataSplits:
    X_test: np.ndarray
    y_test: np.ndarray
    X_min_train: np.ndarray
    X_maj_train: np.ndarray
    X_min_fidelity: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    mu_true: np.ndarray = None


def nested_split(X, y, cfg: SplitConfig, seed: int) -> DataSplits:
    """嵌套划分：测试集 / 验证集 / 生成器保真度集 / 分类器训练集"""
    rng = np.random.default_rng(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=cfg.val_size, stratify=y_train,
        random_state=seed + 1
    )

    X_min = X_train[y_train == 1]
    X_maj = X_train[y_train == 0]

    n_min = len(X_min)

    if n_min == 0:
        # 极端情况：训练集里没有少数类，直接返回空保真度集
        X_min_fid = X_min
        X_min_clf = X_min
    else:
        # 给分类器至少留 min_keep 个真实少数类
        # 至少留 1 个，最多留一半
        min_keep = max(1, min(n_min // 2, 10))

        # 根据 fidelity_ratio 计算期望的保真度集大小
        n_fid_desired = int(n_min * cfg.fidelity_ratio)

        # 下限：至少 1 个（生成器要能 fit）
        # 上限：n_min - min_keep，保证分类器训练集非空
        n_fid = max(1, n_fid_desired)
        n_fid = min(n_fid, n_min - min_keep)
        n_fid = max(1, n_fid)  # 二次保护，避免 n_min - min_keep < 1

        idx = rng.permutation(n_min)
        X_min_fid = X_min[idx[:n_fid]]
        X_min_clf = X_min[idx[n_fid:]]

    return DataSplits(
        X_test=X_test, y_test=y_test,
        X_min_train=X_min_clf, X_maj_train=X_maj,
        X_min_fidelity=X_min_fid,
        X_val=X_val, y_val=y_val,
    )