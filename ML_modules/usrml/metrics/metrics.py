import numpy as np


def rmse(y, y_pred):
    """
    Calculate Root Mean Squared Error for regression problem.

    Params
    ______
    y : array-like
        Observed labels array
    y_pred : array-like
        Estimated labels array

    Returns
    ______
    score : float
        RMSE score
    """
    assert y.shape == y_pred.shape, f'Input arrays must be of the same shape. y is {y.shape}, y_pred is {y_pred.shape}'
    score = np.sqrt(np.sum((y - y_pred) ** 2) / len(y))
    return score


def mse(y, y_pred):
    """
    Calculate Mean Square Error for regression problem.

    Params
    ______
    y : array-like
        Observed labels array
    y_pred : array-like
        Estimated labels array

    Returns
    ______
    score : float
        MSE score
    """
    assert y.shape == y_pred.shape, f'Input arrays must be of the same shape. y is {y.shape}, y_pred is {y_pred.shape}'
    score = np.sum((y - y_pred) ** 2) / len(y)
    return score


def confusion_matrix(y, y_pred):
    """
    Calculate confusion matrix for classification problem

    Params
    ______
    y : array-like
        Observed labels array
    y_pred : array-like
        Estimated labels array

    Returns
    ______
    confusion : numpy array
        Confusion scores, where rows - observed, columns - estimated labels
    """
    unique = np.unique(y)
    confusion = np.zeros((len(unique), len(unique)))
    # y_pred_bin = np.zeros((len(y), len(unique)))
    # for k in unique:
    #     y_pred_bin[:, k] = (y_pred == k)
    # for k in unique:
    #     confusion[k] = np.sum(y_pred_bin[y == k, :], axis=0)
    # mine might actually be faster but take up more memory
    for o, p in zip(y, y_pred):
        confusion[o, p] += 1
    return confusion


def accuracy_score(y, y_pred):
    """
    Calculate accuracy for classification problem

    Params
    ______
    y : array-like
        Observed labels array
    y_pred : array-like
        Estimated labels array

    Returns
    ______
    accuracy : float
        Accuracy score
    """
    accuracy = np.sum(y == y_pred) / len(y)
    return accuracy


def recall(y, y_pred):
    """
    Calculate recall for classification problem

    Params
    ______
    y : array-like
        Observed labels array
    y_pred : array-like
        Estimated labels array

    Returns
    ______
    score : float
        Recall score
    """
    assert len(np.unique(y)) == 2, 'Recall is available for binary problem only'
    TP = np.sum(y[y == 1] == y_pred[y == 1])
    FN = np.sum(y[y == 1] != y_pred[y == 1])
    score = TP / (TP + FN)
    return score


def precision(y, y_pred):
    """
    Calculate precision for classification problem

    Params
    ______
    y : array-like
        Observed labels array
    y_pred : array-like
        Estimated labels array

    Returns
    ______
    score : float
        Precision score
    """
    assert len(np.unique(y)) == 2, 'Precision is available for binary problem only'
    TP = np.sum(y[y == 1] == y_pred[y == 1])
    FP = np.sum(y[y == 0] != y_pred[y == 0])
    score = TP / (TP + FP)
    return score


def f1_score(y, y_pred):
    """
    Calculate F1 for classification problem

    Params
    ______
    y : array-like
        Observed labels array
    y_pred : array-like
        Estimated labels array

    Returns
    ______
    score : float
        F1 score
    """
    assert len(np.unique(y)) == 2, 'F1 is available for binary problem only'
    TP = np.sum(y[y == 1] == y_pred[y == 1])
    FP = np.sum(y[y == 0] != y_pred[y == 0])
    FN = np.sum(y[y == 1] != y_pred[y == 1])
    recall_score = TP / (TP + FN)
    precision_score = TP / (TP + FP)
    score = (2 * recall_score * precision_score) / (recall_score + precision_score)
    return score
