"""
Helper functions that implement various performance metrics in numpy
"""
import numpy as np


def mse(y_true, y_preds):
    """Computes the mean squared error"""
    return np.mean((y_true - y_preds)**2)

def accuracy_score(y_true, y_preds):
    """Computes accuracy of true an predicted labels. Assumes :y_true: and :y_pred: are one-hot encoded vectors."""
    y_true_classes = np.argmax(y_true, axis=1, keepdims=True)
    y_pred_classes = np.argmax(y_preds, axis=1, keepdims=True)

    accuracy_score = np.sum(y_true_classes == y_pred_classes) / y_true_classes.shape[0]
    return accuracy_score


def precision_score(y_true, y_preds):
    """Computes the average precision for each class, weighted by the number of samples in each class"""
    y_true_classes = np.argmax(y_true, axis=1, keepdims=True)
    y_pred_classes = np.argmax(y_preds, axis=1, keepdims=True)

    n_samples = y_true.shape[0]
    n_classes = np.unique(y_true_classes).shape[0]

    w = np.array([np.sum(y_true_classes == i) / n_samples for i in range(n_classes)]).reshape(-1, 1)
    precision = np.zeros((n_classes, 1))
    for i in range(n_classes):
       tp = np.sum(np.logical_and(y_pred_classes == i, y_pred_classes == y_true_classes))
       fp = np.sum(np.logical_and(y_pred_classes == i, y_pred_classes != y_true_classes))
       precision[i] = tp / (tp + fp)

    return np.sum(w * precision)


def recall_score(y_true, y_preds):
    """Computes the average recall for each class, weighted by the number of samples in each class"""
    y_true_classes = np.argmax(y_true, axis=1, keepdims=True)
    y_pred_classes = np.argmax(y_preds, axis=1, keepdims=True)

    n_samples = y_true.shape[0]
    n_classes = np.unique(y_true_classes).shape[0]

    w = np.array([np.sum(y_true_classes == i) / n_samples for i in range(n_classes)]).reshape(-1, 1)
    recall = np.zeros((n_classes, 1))
    for i in range(n_classes):
        tp = np.sum(np.logical_and(y_pred_classes == i, y_pred_classes == y_true_classes))
        fn = np.sum(np.logical_and(y_pred_classes != i, y_true_classes == i))
        recall[i] = tp / (tp + fn)

    return np.sum(w * recall)


def f1_score(y_true, y_preds):
    """Computes the average recall for each class, weighted by the number of samples in each class"""
    y_true_classes = np.argmax(y_true, axis=1, keepdims=True)
    y_pred_classes = np.argmax(y_preds, axis=1, keepdims=True)

    n_samples = y_true.shape[0]
    n_classes = np.unique(y_true_classes).shape[0]

    w = np.array([np.sum(y_true_classes == i) / n_samples for i in range(n_classes)]).reshape(-1, 1)
    precision = np.zeros((n_classes, 1))
    recall = np.zeros((n_classes, 1))
    for i in range(n_classes):
        tp = np.sum(np.logical_and(y_pred_classes == i, y_pred_classes == y_true_classes))
        fp = np.sum(np.logical_and(y_pred_classes == i, y_pred_classes != y_true_classes))
        fn = np.sum(np.logical_and(y_pred_classes != i, y_true_classes == i))
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)

    f1_score = 2 * (precision * recall) / (precision + recall)

    return np.sum(w * f1_score)
