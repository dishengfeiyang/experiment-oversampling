from sklearn.linear_model import LogisticRegression


def train_classifier(X, y, C=1.0, max_iter=1000, class_weight='balanced', **kwargs):
    """
    训练逻辑回归分类器。

    默认 class_weight='balanced'，在不平衡数据下给少数类更高权重，
    避免分类器退化为恒定多数类。
    """
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        **kwargs
    )
    clf.fit(X, y)
    return clf