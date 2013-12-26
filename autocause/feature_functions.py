import numpy as np
from scipy.special import psi


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
