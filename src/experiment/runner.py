"""单次实验运行器"""
import numpy as np
from ..synthesizer.registry import build_synthesizer
from ..classifiers.logistic import train_classifier
from ..metrics.balanced_risk import balanced_excess_risk, balanced_log_loss
from ..metrics.recall import minority_recall, majority_recall, macro_f1


def run_single_method(method, data_splits, synthesizer_cfg, classifier_cfg,
                      k_n, n, seed, m_max=None, c_n=None):
    """对单个方法、单次重复运行实验"""
    # 若未传入 c_n，按理论风险界默认计算
    if c_n is None:
        alpha = 0.05
        M = 13
        B = 1.0
        c_n = 2.0 * B * np.sqrt(np.log(2.0 * M / alpha) / 2.0)

    # 尝试传递 c_n；若方法暂不支持，回退旧调用
    try:
        alloc = method.allocate(k_n=k_n, n=n, m_max=m_max, c_n=c_n)
    except TypeError:
        alloc = method.allocate(k_n=k_n, n=n, m_max=m_max)

    s_n, m_n = alloc.s_n, alloc.m_n

    X_min_pool = data_splits.X_min_train
    if s_n > len(X_min_pool):
        s_n = len(X_min_pool)
    if m_max is not None and m_n > m_max:
        m_n = m_max

    rng = np.random.default_rng(seed + 2000)

    if s_n > 0:
        idx = rng.permutation(len(X_min_pool))
        gen_idx = idx[:s_n]
        clf_idx = idx[s_n:]
        X_gen = X_min_pool[gen_idx]
        X_clf_real = (
            X_min_pool[clf_idx]
            if alloc.use_real_minority
            else np.empty((0, X_min_pool.shape[1]))
        )
    else:
        X_gen = None
        X_clf_real = (
            X_min_pool
            if alloc.use_real_minority
            else np.empty((0, X_min_pool.shape[1]))
        )

    # 生成合成样本
    if s_n > 0 and m_n > 0:
        synth = build_synthesizer(
            synthesizer_cfg["type"],
            mu_true=(
                data_splits.mu_true
                if data_splits.mu_true is not None
                else np.zeros(X_min_pool.shape[1])
            ),
            beta=synthesizer_cfg["beta"],
            epsilon=synthesizer_cfg["epsilon"],
            D=synthesizer_cfg["D"],
        )
        synth.fit(X_gen, seed=seed + 3000)
        X_syn = synth.sample(m_n, seed=seed + 4000)
        discrepancy = synth.discrepancy
    else:
        X_syn = np.empty((0, X_min_pool.shape[1]))
        discrepancy = np.nan

    # 组装训练集
    X_parts = [data_splits.X_maj_train]
    y_parts = [np.zeros(len(data_splits.X_maj_train))]
    if len(X_clf_real) > 0:
        X_parts.append(X_clf_real)
        y_parts.append(np.ones(len(X_clf_real)))
    if len(X_syn) > 0:
        X_parts.append(X_syn)
        y_parts.append(np.ones(len(X_syn)))

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    if len(np.unique(y_train)) < 2:
        return None

    clf = train_classifier(X_train, y_train, **classifier_cfg)

    return {
        "method": method.name,
        "s_n": int(s_n),
        "m_n": int(m_n),
        "s_ratio": s_n / k_n,
        "m_ratio": m_n / max(1, n - 2 * k_n),
        "discrepancy": float(discrepancy),
        "balanced_excess_risk": float(
            balanced_excess_risk(clf, data_splits.X_test, data_splits.y_test)
        ),
        "balanced_log_loss": float(
            balanced_log_loss(clf, data_splits.X_test, data_splits.y_test)
        ),
        "minority_recall": float(
            minority_recall(clf, data_splits.X_test, data_splits.y_test)
        ),
        "majority_recall": float(
            majority_recall(clf, data_splits.X_test, data_splits.y_test)
        ),
        "macro_f1": float(
            macro_f1(clf, data_splits.X_test, data_splits.y_test)
        ),
    }