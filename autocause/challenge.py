import os
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from boomlet.storage import glob_one


def target_score(y_true, predictions):
    return (roc_auc_score(y_true == 1, predictions) +
            roc_auc_score(y_true == -1, -predictions)) / 2


def read_pairs(hint):
    folder = glob_one("data", hint)
    pairs = glob_one(folder, "pairs")
    m = pd.read_csv(pairs).as_matrix()
    return [(np.array(map(int, A.split())), np.array(map(int, B.split())))
            for _, A, B in m]


def read_publicinfo(hint):
    folder = glob_one("data", hint)
    publicinfo = glob_one(folder, "publicinfo")
    m = pd.read_csv(publicinfo).as_matrix()
    # return the first letter of the type e.g.
    # Numerical -> N
    # Categorical -> C
    # Binary -> B
    return [(A[0], B[0]) for _, A, B in m]


def read_target(hint):
    folder = glob_one("data", hint)
    target = glob_one(folder, "target")
    m = pd.read_csv(target).as_matrix()
    return [(int(target), int(details))
            for _, target, details in m]


def read_all(hint):
    pairs = read_pairs(hint)
    publicinfo = read_publicinfo(hint)
    target = read_target(hint)
    together = []
    for p, pi in zip(pairs, publicinfo):
        A, B = p
        A_type, B_type = pi
        together.append((A, A_type, B, B_type))
    return together, np.array(target)


# TODO
# train on sample data
# cross validate
# write output
