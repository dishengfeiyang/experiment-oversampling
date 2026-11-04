"""加载真实不平衡数据集并进行分层划分"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class RealDataSplits:
    X_maj_train: np.ndarray
    X_min_train: np.ndarray
    X_min_fidelity: np.ndarray
    X_min_eval: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    k_n: int
    n: int


def load_mammography(path="data/mammography.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据文件: {path}")
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        df = pd.read_csv(path, header=None, dtype=str)

    for col in df.columns:
        df[col] = df[col].astype(str).str.replace("'", "").str.replace('"', "").str.strip()

    label_col = None
    for c in df.columns:
        if "severity" in str(c).lower() or "target" in str(c).lower() or "class" in str(c).lower():
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]

    X_cols = [c for c in df.columns if c != label_col]
    X = df[X_cols].replace("?", np.nan).fillna(0).astype(float).values
    y = df[label_col].astype(float).values.astype(int)

    unique = np.unique(y)
    if set(unique) == {-1, 1}:
        y = (y == 1).astype(int)
    elif set(unique) == {1, 2}:
        y = (y == 2).astype(int)
    elif not set(unique).issubset({0, 1}):
        counts = {v: int((y == v).sum()) for v in unique}
        minority_label = min(counts, key=counts.get)
        y = (y == minority_label).astype(int)

    print(f"[data] Loaded {path}: X={X.shape}, y={y.shape}, minority={int((y == 1).sum())}")
    return X, y


def split_real_data(X, y, test_size=0.2, fidelity_ratio=0.15, eval_ratio=0.15, 
                    minority_subsample=None, seed=42):
    """支持少数类子采样的分层划分"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # 少数类子采样
    if minority_subsample is not None:
        rng = np.random.default_rng(seed)
        min_idx = np.where(y_train == 1)[0]
        maj_idx = np.where(y_train == 0)[0]
        rng.shuffle(min_idx)
        keep = max(20, int(len(min_idx) * minority_subsample))
        X_train = np.vstack([X_train[maj_idx], X_train[min_idx[:keep]]])
        y_train = np.concatenate([np.zeros(len(maj_idx), dtype=int), np.ones(keep, dtype=int)])
        print(f"[data] Subsample minority: keep={keep}, majority={len(maj_idx)}")

    X_min = X_train[y_train == 1]
    X_maj = X_train[y_train == 0]

    n_min = len(X_min)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_min)

    n_fid = max(3, int(n_min * fidelity_ratio))
    n_eval = max(3, int(n_min * eval_ratio))
    
    while n_fid + n_eval > n_min - 5:
        if n_fid > 3 and n_fid >= n_eval:
            n_fid -= 1
        elif n_eval > 3:
            n_eval -= 1
        else:
            break

    X_min_fid = X_min[idx[:n_fid]]
    X_min_eval = X_min[idx[n_fid:n_fid + n_eval]]
    X_min_clf = X_min[idx[n_fid + n_eval:]]

    return RealDataSplits(
        X_maj_train=X_maj,
        X_min_train=X_min_clf,
        X_min_fidelity=X_min_fid,
        X_min_eval=X_min_eval,
        X_test=X_test,
        y_test=y_test,
        k_n=len(X_min_clf),
        n=len(X_maj) + len(X_min_clf),
    )