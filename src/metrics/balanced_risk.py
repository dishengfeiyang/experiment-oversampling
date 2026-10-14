"""平衡风险指标"""
import numpy as np
from sklearn.metrics import log_loss


def balanced_log_loss(clf, X, y):
    """平衡对数损失（辅助指标，反映校准质量）"""
    proba = clf.predict_proba(X)
    mask0 = (y == 0)
    mask1 = (y == 1)
    loss0 = log_loss(y[mask0], proba[mask0], labels=[0, 1])
    loss1 = log_loss(y[mask1], proba[mask1], labels=[0, 1])
    return 0.5 * loss0 + 0.5 * loss1


def balanced_error(clf, X, y):
    """
    基于 0-1 损失的平衡错误率：
        R_bal = 0.5 * P(pred=1 | y=0) + 0.5 * P(pred=0 | y=1)
    """
    y_pred = clf.predict(X)
    err0 = np.mean(y_pred[y == 0] != 0) if np.sum(y == 0) > 0 else 0.0
    err1 = np.mean(y_pred[y == 1] != 1) if np.sum(y == 1) > 0 else 0.0
    return 0.5 * err0 + 0.5 * err1


def balanced_excess_risk(clf, X_test, y_test):
    """
    平衡超额风险 = R_bal(clf) - R_bal(baseline)

    基线：恒定预测多数类（少数类召回 = 0）。
        R_bal(baseline) = 0.5 * 0 + 0.5 * 1 = 0.5

    因此：
        - real_only          → excess ≈ 0
        - 任意非平凡分类器   → excess ∈ [-0.5, 0)
        - 完美分类器         → excess = -0.5
    数值越负越好。
    """
    R_clf = balanced_error(clf, X_test, y_test)
    R_baseline = 0.5
    return R_clf - R_baseline