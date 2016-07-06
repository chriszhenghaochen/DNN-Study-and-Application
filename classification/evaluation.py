import numpy as np
import pandas as pd

def calculate_precision_one_class(voted, annotated, given_class):
    v = voted.loc[voted.iloc[:, -1] == given_class, :]
    a = annotated.loc[annotated.iloc[:, -1] == given_class, :]
    tp = len(np.intersect1d(v.iloc[:, 0], a.iloc[:, 0]))
    p = len(a)
    if p != 0:
        return tp / float(p)
    else:
        return np.nan

def calculate_precision_overall(voted, annotated):
    up = 0
    down = 0
    classes = np.unique(voted.iloc[:, -1]).tolist()
    for i in classes:
        p = calculate_precision_one_class(voted, annotated, i)
        if np.isnan(p):
            continue
        else:
            up += p * len(voted.loc[voted.iloc[:, -1] == i, :])
            down += len(voted.loc[voted.iloc[:, -1] == i, :])
    return up / float(down)
    
def calculate_recall_one_class(voted, annotated, given_class):
    v = voted.loc[voted.iloc[:, -1] == given_class, :]
    a = annotated.loc[annotated.iloc[:, -1] == given_class, :]
    tp = len(np.intersect1d(v.iloc[:, 0], a.iloc[:, 0]))
    n = len(v)
    if n != 0:
        return tp / float(n)
    else:
        return np.nan

def calculate_recall_overall(voted, annotated):
    up = 0
    down = 0
    classes = np.unique(voted.iloc[:, -1]).tolist()
    for i in classes:
        r = calculate_recall_one_class(voted, annotated, i)
        if np.isnan(r):
            continue
        else:
            up += r * len(voted.loc[voted.iloc[:, -1] == i, :])
            down += len(voted.loc[voted.iloc[:, -1] == i, :])
    return up / float(down)
    
def calculate_F_score_one_class(voted, annotated, given_class):
    p = calculate_precision_one_class(voted, annotated, given_class)
    r = calculate_recall_one_class(voted, annotated, given_class)
    if np.isnan(p) or np.isnan(r) or p + r == 0:
        return np.nan
    else:
        return 2 * p * r / float(p + r)

def calculate_F_score_overall(voted, annotated):
    up = 0
    down = 0
    classes = np.unique(voted.iloc[:, -1]).tolist()
    for i in classes:
        f = calculate_F_score_one_class(voted, annotated, i)
        if np.isnan(f):
            continue
        else:
            up += f * len(voted.loc[voted.iloc[:, -1] == i, :])
            down += len(voted.loc[voted.iloc[:, -1] == i, :])
    return up / float(down)