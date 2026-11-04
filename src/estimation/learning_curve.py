"""通过分类器估计生成器与真实分布的 TV 距离，并拟合 D s^{-beta} + eps"""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from ..synthesizer.gmm import GMMSynthesizer


def estimate_tv_distance(X_real, X_syn, seed=42):
    """
    用随机森林区分真实样本和合成样本，用 AUC 估计 TV 距离：
        TV ≈ 2 * AUC - 1。
    """
    if len(X_real) < 10 or len(X_syn) < 10:
        return np.nan
    X = np.vstack([X_real, X_syn])
    y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_syn))])
    # 增大树深到 5，能捕捉 3 维空间中的非线性边界
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
    try:
        probs = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, probs)
        # 截断到 [0, 1] 防止数值误差
        tv = max(0.0, min(1.0, 2 * auc - 1))
        return tv
    except Exception:
        return np.nan


def _power_law(s, D, beta, eps):
    return D * np.power(s, -beta) + eps


def fit_learning_curve(s_values, tv_values, beta_bounds=(0.05, 3.0)):
    """
    拟合 delta(s) = D s^{-beta} + eps。
    """
    s_arr = np.asarray(s_values, dtype=float)
    tv_arr = np.asarray(tv_values, dtype=float)
    mask = ~np.isnan(tv_arr)
    s_arr, tv_arr = s_arr[mask], tv_arr[mask]
    
    if len(s_arr) < 4:
        return 1.0, 1.0, max(0.0, float(np.nanmean(tv_arr)) if len(tv_arr) else 0.1)

    eps0 = max(0.01, float(tv_arr.min()) * 0.8)
    D0 = max(0.1, float(tv_arr.max() - eps0))
    p0 = [D0, 1.0, eps0]
    
    # === 方法 B：eps 下界设为 0.01，避免被推到 0 ===
    bounds = ([1e-4, beta_bounds[0], 0.01], [1e4, beta_bounds[1], 1.0])
    
    try:
        popt, _ = curve_fit(_power_law, s_arr, tv_arr, p0=p0, bounds=bounds, maxfev=10000)
        D_hat, beta_hat, eps_hat = popt
        return float(beta_hat), float(D_hat), float(eps_hat)
    except Exception:
        return 1.0, float(D0), float(eps0)


def estimate_learning_curve(X_min_fid, X_min_eval, s_grid, n_syn_per_s=500, seed=42):
    """
    拟合 GMM 用 X_min_fid，评估 TV 距离用独立的 X_min_eval。
    """
    rng = np.random.default_rng(seed)
    s_list, tv_list = [], []
    max_s = len(X_min_fid)

    for s in s_grid:
        s_use = min(s, max_s)
        if s_use < 3:
            continue
        idx = rng.permutation(max_s)[:s_use]
        X_sub = X_min_fid[idx]

        syn = GMMSynthesizer(random_state=seed)
        try:
            syn.fit(X_sub, seed=seed)
            X_syn = syn.sample(n_syn_per_s, seed=seed)
        except Exception:
            continue

        tv = estimate_tv_distance(X_min_eval, X_syn, seed=seed)
        if np.isnan(tv):
            continue
        s_list.append(s_use)
        tv_list.append(tv)

    # === 方法 A：对 0 做线性插值平滑 ===
    if len(tv_list) >= 3:
        tv_arr = np.array(tv_list, dtype=float)
        nonzero_idx = np.where(tv_arr > 0)[0]
        if len(nonzero_idx) >= 2:
            for i in range(len(tv_arr)):
                if tv_arr[i] == 0:
                    left = nonzero_idx[nonzero_idx < i]
                    right = nonzero_idx[nonzero_idx > i]
                    if len(left) > 0 and len(right) > 0:
                        l, r = left[-1], right[0]
                        w = (i - l) / (r - l)
                        tv_arr[i] = (1 - w) * tv_arr[l] + w * tv_arr[r]
                    elif len(left) > 0:
                        tv_arr[i] = tv_arr[left[-1]] * 0.5
                    elif len(right) > 0:
                        tv_arr[i] = tv_arr[right[0]] * 0.5
            tv_list = tv_arr.tolist()

    # 强制单调不增（TV 距离不应随 s 增大而上升）
    if len(tv_list) >= 2:
        tv_arr = np.array(tv_list)
        tv_arr = np.minimum.accumulate(tv_arr)
        tv_list = tv_arr.tolist()

    beta_hat, D_hat, eps_hat = fit_learning_curve(s_list, tv_list)
    return {
        "beta": beta_hat,
        "D": D_hat,
        "eps": eps_hat,
        "s_grid_used": s_list,
        "tv_estimates": tv_list,
    }