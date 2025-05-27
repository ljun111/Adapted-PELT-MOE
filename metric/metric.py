import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

def precision(truelabels, predictedlabels):
    """
    Calculate precision.

    Args:
    truelabels (list): List of true labels.
    predictedlabels (list): List of predicted labels.

    Returns:
    float: The precision score.
    """
    true_positive = sum(1 for t, p in zip(truelabels, predictedlabels) if t == 1 and p == 1)
    false_positive = sum(1 for t, p in zip(truelabels, predictedlabels) if t == 0 and p == 1)
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0


def recall(truelabels, predictedlabels):
    """
    Calculate recall.

    Args:
    truelabels (list): List of true labels.
    predictedlabels (list): List of predicted labels.

    Returns:
    float: The recall score.
    """
    true_positive = sum(1 for t, p in zip(truelabels, predictedlabels) if t == 1 and p == 1)
    false_negative = sum(1 for t, p in zip(truelabels, predictedlabels) if t == 1 and p == 0)
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0


def f1_score(truelabels, predictedlabels):
    """
    Calculate F1 score.

    Args:
    truelabels (list): List of true labels.
    predictedlabels (list): List of predicted labels.

    Returns:
    float: The F1 score.
    """
    prec = precision(truelabels, predictedlabels)
    rec = recall(truelabels, predictedlabels)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def calculate_accuracy(changepoints, true_changepoints):
    """
    Calculate the accuracy of whether the values at corresponding indices between
    changepoints and true_changepoints differ by no more than 5.

    Args:
        changepoints (list): List of predicted change points
        true_changepoints (list): List of true change points

    Returns:
        float: Accuracy score (between 0 and 1)
    """
    changepoints = changepoints[1:]
    if len(changepoints) != len(true_changepoints):
        raise ValueError("两个列表长度必须相等")

    correct = 0
    total = len(changepoints)

    for pred, true in zip(changepoints, true_changepoints):
        if abs(pred - true) <= 5:
            correct += 1

    accuracy = correct / total
    return accuracy


def calculate_auroc(y_true, y_scores):
    """
    Calculate AUROC (Area Under the ROC Curve)

    Args:
        y_true: Array of true labels (0 or 1)
        y_scores: Array of predicted scores/probabilities

    Returns:
        auroc: AUROC value
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUROC需要二分类问题")
    return roc_auc_score(y_true, y_scores)


def calculate_auprc(y_true, y_scores):
    """
    Calculate AUPRC (Area Under the Precision-Recall Curve)

    Args:
        y_true: Array of true labels (0 or 1)
        y_scores: Array of predicted scores/probabilities

    Returns:
        auprc: AUPRC value
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUPRC需要二分类问题")
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)  # 注意参数顺序是recall在前


def adjusted_f1_score(true_labels,pred_labels):
    """
    Calculate adjusted F1 score where predicted labels within ±tolerance range
    of true labels are considered correct.

    Args:
        true_labels (List[int]): List of true labels
        pred_labels (List[int]): List of predicted labels
        tolerance (int): Maximum allowed index difference (default: 5)

    Returns:
        float: Adjusted F1 score (between 0 and 1)
    """
    anomaly_state = False
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    for i in range(len(true_labels)):
        if true_labels[i] == 1 and pred_labels[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if true_labels[j] == 0:
                    break
                else:
                    if pred_labels[j] == 0:
                        pred_labels[j] = 1
            for j in range(i, len(true_labels)):
                if true_labels[j] == 0:
                    break
                else:
                    if pred_labels[j] == 0:
                        pred_labels[j] = 1
        elif true_labels[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred_labels[i] = 1

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f_score, support = precision_recall_fscore_support(
        true_labels, pred_labels.astype(bool), average='binary')
    return accuracy, precision, recall, f_score