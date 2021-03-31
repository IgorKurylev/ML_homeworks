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
    if percent is None:
        res_pred = np.zeros(y_predict.shape)
        for i, item in enumerate(y_predict):
            if item >= 0.5:
                res_pred[i] = 1
            else:
                res_pred[i] = 0

    elif 0 < percent <= 100:
        new_len = int(len(y_predict) * percent / 100)
        res_pred = np.zeros(y_predict.shape)
        indices = y_predict.argsort()[-new_len:][::-1]
        for i, index in enumerate(indices):
            res_pred[i] = 1
    else:
        raise ValueError
    return res_pred, y_true


def accuracy_score(y_true, y_predict, percent=None):
    y_predict, y_true = eval_percent(y_true, y_predict, percent)
    res = get_confusion_matrix(y_true, y_predict)
    accuracy = (res["TP"] + res["TN"]) / \
               (res["TN"] + res["FP"] + res["FN"] + res["TP"] if res["TN"] + res["FP"] + res["FN"] + res["TP"]  else inf)
    return accuracy


def precision_score(y_true, y_predict, percent=None):
    y_predict, y_true = eval_percent(y_true, y_predict, percent)
    res = get_confusion_matrix(y_true, y_predict)
    precision = res["TP"] / \
                (res["TP"] + res["FP"] if res["TP"] + res["FP"] else inf)
    return precision


def recall_score(y_true, y_predict, percent=None):
    y_predict, y_true = eval_percent(y_true, y_predict, percent)
    res = get_confusion_matrix(y_true, y_predict)
    recall = res["TP"] / \
             (res["TP"] + res["FN"] if res["TP"] + res["FN"] else inf)
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
    print(accuracy_score(y_true, y_predict[:, 1], 90))
    print(precision_score(y_true, y_predict[:, 1]))
    print(recall_score(y_true, y_predict[:, 1]))
    print(lift_score(y_true, y_predict[:, 1]))
    print(f1_score(y_true, y_predict[:, 1]))


