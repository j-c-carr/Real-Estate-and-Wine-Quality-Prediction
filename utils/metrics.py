import numpy as np


def mse(y_true, y_preds):
    """Computes the mean squared error"""
    return np.mean((y_true - y_preds)**2)

def class_metrics(y_true, y_preds, class_avg=False):
    """NOTE: INCOMPLETE
    Computes the mean squared error for each class. Assumes that y_true is one-hot encoded vector."""

    y_bin_pred = (y_preds >= y_preds.max(axis=1, keepdims=True)).astype(int)
    tp = np.zeros(y_true.shape[1])
    fp = np.zeros(y_true.shape[1])
    tn = np.zeros(y_true.shape[1])
    fn = np.zeros(y_true.shape[1])

    # Calculate the tp, fp, tn, fn for each class
    # src: https://stackoverflow.com/questions/68157408/using-numpy-to-test-for-false-positives-and-false-negatives
    positive = 1
    negative = 0
    for i in range(y_true.shape[1]):
        tp[i] = np.sum(np.logical_and(y_bin_pred[:, i] == positive, y_true[:, i] == positive))
        tn[i] = np.sum(np.logical_and(y_bin_pred[:, i] == negative, y_true[:, i] == negative))
        fp[i] = np.sum(np.logical_and(y_bin_pred[:, i] == positive, y_true[:, i] == negative))
        fn[i] = np.sum(np.logical_and(y_bin_pred[:, i] == negative, y_true[:, i] == positive))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # compute the average accuracy, precision, recall and f1 score
    if class_avg:
        # get the fraction of samples for each class
        w = np.array([np.sum(y_true[:, i]) / y_true.shape[0] for i in range(3)])
        avg_accuracy = np.sum(accuracy * w)
        avg_precision = np.sum(precision* w)
        avg_recall = np.sum(recall * w)
        avg_f1 = np.sum(f1_score* w)
        return avg_accuracy, avg_precision, avg_recall, avg_f1

    return accuracy, precision, recall, f1_score
