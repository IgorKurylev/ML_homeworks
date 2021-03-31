import numpy as np
from math import inf


def get_confusion_matrix(y_true, y_predict):
    cm = {"TN": 0,
          "TP": 0,
          "FP": 0,
          "FN": 0}

    for y_tr, y_pr in zip(y_true, y_predict):
        if y_tr > y_pr:
            cm["FN"] += 1
        elif y_tr < y_pr:
            cm["FP"] += 1
        else:
            if y_tr:
                cm["TP"] += 1
            else:
                cm["TN"] += 1
    return cm


def eval_percent(y_true, y_predict, percent):
    res_pred = np.zeros(y_predict.shape)
    res_true = np.array(y_true.shape)
    if percent is None:
        for i, item in enumerate(y_predict):
            if item >= 0.5:
                res_pred[i] = 1
            else:
                res_pred[i] = 0
        res_true = y_true
    elif 0 < percent <= 100:
        raise NotImplementedError
    else:
        raise ValueError
    return res_pred, res_true


def accuracy_score(y_true, y_predict, percent=None):
    y_predict, y_true = eval_percent(y_true, y_predict, percent)
    res = get_confusion_matrix(y_true, y_predict)
    accuracy = (res["TP"] + res["TN"]) / (res["TN"] + res["FP"] + res["FN"] + res["TP"])
    return accuracy


def precision_score(y_true, y_predict, percent=None):
    y_predict, y_true = eval_percent(y_true, y_predict, percent)
    res = get_confusion_matrix(y_true, y_predict)
    precision = res["TP"] / (res["TP"] + res["FP"])
    return precision


def recall_score(y_true, y_predict, percent=None):
    y_predict, y_true = eval_percent(y_true, y_predict, percent)
    res = get_confusion_matrix(y_true, y_predict)
    recall = res["TP"] / (res["TP"] + res["FN"])
    return recall


def lift_score(y_true, y_predict, percent=None):
    y_predict, y_true = eval_percent(y_true, y_predict, percent)
    res = get_confusion_matrix(y_true, y_predict)
    lift = (res["TP"] / (res["TP"] + res["FN"])) / \
           ((res["TP"] + res["FP"]) / \
           (res["TP"] + res["FP"] + res["TN"] + res["FN"]))
    return lift


def f1_score(y_true, y_predict, percent=None):
    y_predict, y_true = eval_percent(y_true, y_predict, percent)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1_score = 2 * (precision * recall) / (precision + recall if precision + recall else inf)
    return f1_score


if __name__ == "__main__":
    file = np.loadtxt('HW2_labels.txt', delimiter=',')
    y_predict, y_true = file[:, :2], file[:, -1]
    print(y_predict[:, 0])
    print(accuracy_score(y_true, y_predict[:, 1]))
    print(precision_score(y_true, y_predict[:, 1]))
    print(recall_score(y_true, y_predict[:, 1]))
    print(lift_score(y_true, y_predict[:, 1]))
    print(f1_score(y_true, y_predict[:, 1]))


