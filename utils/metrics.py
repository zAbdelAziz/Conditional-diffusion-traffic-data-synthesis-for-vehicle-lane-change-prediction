from numpy import ndarray, zeros, diag, trace, mean, sqrt, int64, float64

def confusion_matrix(y_true: ndarray, y_pred: ndarray, num_classes: int):
    cm = zeros((num_classes, num_classes), dtype=int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_from_cm(cm: ndarray, eps: float = 1e-12):
    # cm[i,j]: true=i predicted=j
    tp = diag(cm).astype(float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    support = cm.sum(axis=1).astype(int64)
    return precision, recall, f1, support


def accuracy_from_cm(cm: ndarray):
    return float(trace(cm) / max(cm.sum(), 1))


def balanced_accuracy_from_cm(cm: ndarray, eps: float = 1e-12):
    # mean recall across classes
    recall = diag(cm) / (cm.sum(axis=1) + eps)
    return float(mean(recall))


def macro_f1_from_cm(cm: ndarray, eps: float = 1e-12):
    _, _, f1, _ = precision_recall_f1_from_cm(cm, eps=eps)
    return float(mean(f1))


def weighted_f1_from_cm(cm: ndarray, eps: float = 1e-12):
    _, _, f1, support = precision_recall_f1_from_cm(cm, eps=eps)
    w = support / max(support.sum(), 1)
    return float(sum(f1 * w))


def mcc_from_cm(cm: ndarray, eps: float = 1e-12):
    # Multiclass MCC (Gorodkin)
    t_sum = cm.sum(axis=1).astype(float64)
    p_sum = cm.sum(axis=0).astype(float64)
    n = cm.sum().astype(float64)
    c = trace(cm).astype(float64)

    s = sum(p_sum * t_sum)
    numerator = c * n - s
    denom = sqrt((n**2 - sum(p_sum**2)) * (n**2 - sum(t_sum**2)) + eps)
    return float(numerator / (denom + eps))
