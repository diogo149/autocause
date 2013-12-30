from collections import defaultdict

import numpy as np
from scipy.special import psi
from scipy.stats import pearsonr, chisquare, f_oneway, kruskal
from scipy.spatial.distance import dice, sokalsneath, yule
from sklearn.decomposition import FastICA

from boomlet.metrics import categorical_gini_coefficient, gini_coefficient


def normalized_entropy(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)
    len_x = len(x)

    hx = 0.0
    for i in range(len_x - 1):
        delta = x[i + 1] - x[i]
        if delta != 0:
            hx += np.log(np.abs(delta))
    hx = hx / (len_x - 1) + psi(len_x) - psi(1)

    return hx


def independent_component(x):
    clf = FastICA(random_state=1)
    transformed = clf.fit(x.reshape(-1, 1))
    comp = clf.components_[0, 0]
    mm = clf.mixing_[0, 0]
    src_max = transformed.max()
    src_min = transformed.min()
    return [comp, mm, src_max, src_min]


def correlation_magnitude(x, y):
    return abs(pearsonr(x, y)[0])


def chi_square(x, y):
    return chisquare(x - min(x) + 1, y - min(y) + 1)


def categorical_categorical_homogeneity(x, y):
    grouped = defaultdict(list)
    [grouped[x_val].append(y_val) for x_val, y_val in zip(x, y)]
    homogeneity = [categorical_gini_coefficient(val) for val in grouped.values()]
    return (max(homogeneity), np.mean(homogeneity), min(homogeneity), np.std(homogeneity))


def categorical_numerical_homogeneity(x, y):
    grouped = defaultdict(list)
    [grouped[x_val].append(y_val) for x_val, y_val in zip(x, y)]
    homogeneity = [gini_coefficient(val) for val in grouped.values()]
    return (max(homogeneity), np.mean(homogeneity), min(homogeneity), np.std(homogeneity))


def anova(x, y):
    grouped = defaultdict(list)
    [grouped[x_val].append(y_val) for x_val, y_val in zip(x, y)]
    grouped_values = grouped.values()
    if len(grouped_values) < 2:
        return (0, 0, 0, 0)
    f_oneway_res = list(f_oneway(*grouped_values))
    try:
        kruskal_res = list(kruskal(*grouped_values))
    except ValueError:  # when all numbers are identical
        kruskal_res = [0, 0]
    return f_oneway_res + kruskal_res


def bucket_variance(x, y):
    grouped = defaultdict(list)
    [grouped[x_val].append(y_val) for x_val, y_val in zip(x, y)]
    grouped_values = grouped.values()
    weighted_avg_var = 0.0
    max_var = 0.0
    for bucket in grouped_values:
        var = np.std(bucket) ** 2
        max_var = max(var, max_var)
        weighted_avg_var += len(bucket) * var
    weighted_avg_var /= len(x)
    return (max_var, weighted_avg_var)


def dice_(x, y):
    try:
        return dice(x, y)
    except (ZeroDivisionError, TypeError):
        return 0


def sokalsneath_(x, y):
    try:
        return sokalsneath(x, y)
    except ValueError:
        return 0


def yule_(x, y):
    try:
        return yule(x, y)
    except ZeroDivisionError:
        return 0
