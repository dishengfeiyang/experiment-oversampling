"""真实数据实验运行器"""
import numpy as np
from dataclasses import dataclass

from ..synthesizer.gmm import GMMSynthesizer
from ..classifiers.logistic import train_classifier
from ..metrics.balanced_risk import balanced_excess_risk
from ..metrics.recall import minority_recall, majority_recall, macro_f1
from ..allocation.objective import compute_c_n
from ..allocation.optimization import find_theoretical_optimal


@dataclass
class RealResult:
    method: str
    s_n: int
    m_n: int
    balanced_excess_risk: float
    minority_recall: float
    majority_recall: float
    macro_f1: float


def _train_and_eval(X_train, y_train, X_test, y_test, clf_cfg):
    if len(np.unique(y_train)) < 2:
        return None
    clf = train_classifier(X_train, y_train, **clf_cfg)
    return {
        "balanced_excess_risk": float(balanced_excess_risk(clf, X_test, y_test)),
        "minority_recall": float(minority_recall(clf, X_test, y_test)),
        "majority_recall": float(majority_recall(clf, X_test, y_test)),
        "macro_f1": float(macro_f1(clf, X_test, y_test)),
    }


def _assemble(X_maj, X_real_min, X_syn):
    X_parts = [X_maj]
    y_parts = [np.zeros(len(X_maj), dtype=int)]
    if X_real_min is not None and len(X_real_min) > 0:
        X_parts.append(X_real_min)
        y_parts.append(np.ones(len(X_real_min), dtype=int))
    if X_syn is not None and len(X_syn) > 0:
        X_parts.append(X_syn)
        y_parts.append(np.ones(len(X_syn), dtype=int))
    return np.vstack(X_parts), np.concatenate(y_parts)


def run_real_method(method_name, s_n, m_n, use_real_minority, splits,
                    clf_cfg, seed, gmm_components=None):
    rng = np.random.default_rng(seed + 1000)
    X_min_pool = splits.X_min_train
    k_n = len(X_min_pool)

    s_n = min(max(s_n, 0), k_n)
    if s_n > 0:
        idx = rng.permutation(k_n)
        gen_idx = idx[:s_n]
        clf_idx = idx[s_n:]
        X_gen = X_min_pool[gen_idx]
        X_clf_real = X_min_pool[clf_idx] if use_real_minority else np.empty((0, X_min_pool.shape[1]))
    else:
        X_gen = None
        X_clf_real = X_min_pool if use_real_minority else np.empty((0, X_min_pool.shape[1]))

    if s_n > 0 and m_n > 0:
        syn = GMMSynthesizer(n_components=gmm_components, random_state=seed)
        try:
            syn.fit(X_gen, seed=seed)
            X_syn = syn.sample(m_n, seed=seed)
        except Exception as e:
            print(f"[WARN] GMM failed for s={s_n}: {e}")
            X_syn = np.empty((0, X_min_pool.shape[1]))
    else:
        X_syn = np.empty((0, X_min_pool.shape[1]))

    X_train, y_train = _assemble(splits.X_maj_train, X_clf_real, X_syn)
    metrics = _train_and_eval(X_train, y_train, splits.X_test, splits.y_test, clf_cfg)
    if metrics is None:
        return None
    return RealResult(
        method=method_name,
        s_n=int(s_n),
        m_n=int(m_n),
        **metrics,
    )


def theoretical_allocation(k_n, n, beta, D, eps, m_max, clf_weights=(0.5, 0.5),
                           force_m_max=False):
    """
    计算理论最优。
    如果 force_m_max=True，则固定 m = m_max，只优化 s。
    这对应论文中“在相同合成预算下比较比例”的设定。
    """
    c_n = compute_c_n(alpha=0.05, M=13, B=1.0)
    
    if force_m_max:
        from ..allocation.optimization import find_optimal_s_fixed_m
        s_opt, _ = find_optimal_s_fixed_m(
            k_n=k_n, n=n, m=m_max,
            beta=beta, epsilon=eps, D=D,
            c_n=c_n, w0=clf_weights[0], w1=clf_weights[1]
        )
        m_opt = m_max
    else:
        s_opt, m_opt, _ = find_theoretical_optimal(
            k_n=k_n, n=n,
            beta=beta, epsilon=eps, D=D,
            c_n=c_n, w0=clf_weights[0], w1=clf_weights[1],
            m_max=m_max,
        )
    return s_opt, m_opt