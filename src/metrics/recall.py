"""召回率和 F1"""
from sklearn.metrics import recall_score, f1_score


def minority_recall(clf, X, y):
    y_pred = clf.predict(X)
    return recall_score(y, y_pred, pos_label=1, zero_division=0)


def majority_recall(clf, X, y):
    y_pred = clf.predict(X)
    return recall_score(y, y_pred, pos_label=0, zero_division=0)


def macro_f1(clf, X, y):
    y_pred = clf.predict(X)
    return f1_score(y, y_pred, average="macro", zero_division=0)