import copy
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from boomlet.utils.estimators import binarizer_from_classifier
from boomlet.transform.preprocessing import InfinityReplacer
from boomlet.parallel import pmap

from autocause_settings import *


def to_2d(m):
    if len(m.shape) == 2:
        return m
    elif len(m.shape) == 1:
        return m.reshape(-1, 1)
    else:
        raise Exception("Improper shape: {}".fotmat(m.shape))


def result_append(array, result):
    """
    appends a result or results to an array; this is necessary
    because some relevant functions return iterables, and it would
    be messy to wrap every function that returns a scalar
    """
    try:
        l = list(result)
    except TypeError:
        # assuming that result is a numeric type
        # if it is not an iterable
        assert isinstance(result, (int, float))
        array.append(result)
    else:
        array += l

    return array


def aggregate_apply(func, items, **kwargs):
    """
    performs a function with each item in a collection of items as a first argument,
    then aggregating the results into a 1-D list
    """
    preaggregate = []
    for item in items:
        preaggregate.append(func(item, **kwargs))
    features = []
    if len(preaggregate) > 0:
        preaggregate2d = to_2d(np.array(preaggregate))
        for feat in preaggregate2d.T:
            for aggregator in AGGREGATORS:
                result_append(features, aggregator(feat))
    return features


def aggregate_proxy(func, data, aggregate, **kwargs):
    """
    performs the logic on whether or not to aggregate based on a boolean flag (aggregate)
    """
    if aggregate:
        # have each element be a column
        return aggregate_apply(func, to_2d(data).T, **kwargs)
    else:
        assert len(data.shape) == 1
        return func(data, **kwargs)


def unary_features(X, X_type):
    assert len(X.shape) == 1
    if X_type == "N":
        funcs = UNARY_NUMERICAL_FEATURES
    elif X_type == "C":
        funcs = UNARY_CATEGORICAL_FEATURES
    else:
        raise Exception("improper type: {}".format(X_type))

    features = []
    for feat in funcs:
        result_append(features, feat(X))
    return features


def binary_features(A_feat, B_feat, current_type):
    assert len(A_feat.shape) == 1
    assert len(B_feat.shape) == 1

    if current_type == "NN":
        funcs = BINARY_NN_FEATURES
    elif current_type == "NC":
        funcs = BINARY_NC_FEATURES
    elif current_type == "CN":
        funcs = BINARY_CN_FEATURES
    elif current_type == "CC":
        funcs = BINARY_CC_FEATURES
    else:
        raise Exception("improper type: {}".format(current_type))

    features = []
    for feat in funcs:
        result_append(features, feat(A_feat, B_feat))
    return features


def regression_features(X, y):
    features = []
    for clf in REGRESSION_ESTIMATORS:
        clf = copy.deepcopy(clf)
        clf.fit(X, y)
        for metric in REGRESSION_MODEL_METRICS:
            result_append(features, metric(clf, X, y))
        # TODO predicting in-sample, might be good to change someday
        pred = clf.predict(X)
        for metric in REGRESSION_METRICS:
            result_append(features, metric(y, pred))
        residual = y - pred
        for metric in REGRESSION_RESIDUAL_METRICS:
            result_append(features, metric(residual))
    return features


def classification_features(X, y):
    # TODO check if this actually happens in practice
    # special casing when there is only one value of y
    # has_one_class = len(np.unique(y)) == 1

    def apply_col(idx, f, matrices):
        return f(*[m[:, idx] for m in matrices])

    features = []
    for clf in CLASSIFICATION_ESTIMATORS:
        clf = copy.deepcopy(clf)
        clf.fit(X, y)
        for metric in CLASSIFICATION_MODEL_METRICS:
            result_append(features, metric(clf, X, y))
        binarizer = binarizer_from_classifier(clf)
        y_bin = binarizer.transform(y).astype(np.float)
        cols = y_bin.shape[1]

        # TODO predicting in-sample, might be good to change someday

        probs = clf.predict_proba(X)
        # special case when there are only two classes:
        # predict_proba returns 2 columns when the LabelBinarizer
        # returns 1
        if probs.shape[1] == 2:
            probs = probs[:, 1:]
        for metric in BINARY_PROBABILITY_CLASSIFICATION_METRICS:
            assert probs.shape[1] == cols, (cols, probs.shape)
            features += aggregate_apply(apply_col,
                                        range(cols),
                                        f=metric,
                                        matrices=[y_bin, probs])

        residual = y_bin - probs
        for metric in RESIDUAL_PROBABILITY_CLASSIFICATION_METRICS:
            features += aggregate_apply(apply_col,
                                        range(cols),
                                        f=metric,
                                        matrices=[residual])

        pred = clf.predict(X)
        for metric in ND_CLASSIFICATION_METRICS:
            result_append(features, metric(y, pred))

        pred_bin = binarizer.transform(pred).astype(np.float)
        for metric in BINARY_CLASSIFICATION_METRICS:
            assert pred_bin.shape[1] == cols
            features += aggregate_apply(apply_col,
                                        range(cols),
                                        f=metric,
                                        matrices=[y_bin, pred_bin])
    return features


def estimator_features(A_feat, B_feat, current_type):
    assert current_type in ("NN", "NC", "CN", "CC")
    A_type, B_type = current_type
    y = B_feat

    if A_type == "N":
        # convert numerical to 2-D matrix
        X = to_2d(A_feat)
    elif A_type == "C":
        # convert categorical to binary matrix
        X = LabelBinarizer().fit_transform(A_feat)
    else:
        raise Exception("improper A_type: {}".format(A_type))

    if B_type == "N":
        return regression_features(X, y)
    elif B_type == "C":
        return classification_features(X, y)
    else:
        raise Exception("improper B_type: {}".format(B_type))


def convert(X_raw, X_current_type, Y_raw, Y_type):
    """
    converts X_raw to different types of the same data, whether or
    not to aggregate over that data (in order to be explicit), and
    the new type of that data
    """
    assert X_current_type in "NBC"
    # conversion to numerical
    if CONVERT_TO_NUMERICAL:
        converter = NUMERICAL_CONVERTERS[X_current_type]
        yield (converter(X_raw, X_current_type, Y_raw, Y_type).astype(np.float),
               NUMERICAL_CAN_BE_2D,
               "N")
    if CONVERT_TO_CATEGORICAL:
        converter = CATEGORICAL_CONVERTERS[X_current_type]
        yield (converter(X_raw, X_current_type, Y_raw, Y_type).astype(np.float),
               CATEGORICAL_CAN_BE_2D,
               "C")


def featurize_one_way(A, A_type, B, B_type):
    features = []
    for A_data, A_aggregate, A_desired in convert(A, A_type, B, B_type):
        # unary features
        features += aggregate_proxy(unary_features,
                                    A_data,
                                    A_aggregate,
                                    X_type=A_desired)

        def estimator_features_wrapper(B_data, current_type):
            assert len(B_data.shape) == 1
            return estimator_features(to_2d(A_data), B_data, current_type)

        def binary_features_wrapper(B_data, current_type):
            assert len(B_data.shape) == 1
            return aggregate_proxy(binary_features,
                                   A_data,
                                   A_aggregate,
                                   B_feat=B_data,
                                   current_type=current_type)

        for B_data, B_aggregate, B_desired in convert(B, B_type, A, A_type):
            current_type = A_desired + B_desired  # e.g. "NC"
            # aggregate over B_data for estimator features
            features += aggregate_proxy(estimator_features_wrapper,
                                        B_data,
                                        B_aggregate,
                                        current_type=current_type)
            # aggregate over both A and B for binary features
            features += aggregate_proxy(binary_features_wrapper,
                                        B_data,
                                        B_aggregate,
                                        current_type=current_type)

    return np.array(features)


def featurize_pair(pair):
    A, A_type, B, B_type = pair
    A_to_B = featurize_one_way(A, A_type, B, B_type)
    B_to_A = featurize_one_way(B, B_type, A, A_type)
    return A_to_B, B_to_A


def postprocess(array):
    return InfinityReplacer().transform(array)


def reflect_data(A_to_B, B_to_A):
    """
    returns features equivalent to if the input to the algorithm contained
    B, A for each A, B in the input
    """
    if not REFLECT_DATA:
        return A_to_B, B_to_A
    return np.vstack((A_to_B, B_to_A)), np.vstack((B_to_A, A_to_B))


def relative_features(A_to_B, B_to_A):
    return np.hstack([relative_feat(A_to_B, B_to_A)
                      for relative_feat in RELATIVE_FEATURES])


def featurize(pairs):
    """
    takes in input of the form (A, A_type, B, B_type) with A_type and
    B_type in {"N", "C", "B"} for numerical, cateogrical, binary respectively
    """
    featurized = pmap(featurize_pair, pairs)
    A_to_B = np.array([i[0] for i in featurized])
    B_to_A = np.array([i[1] for i in featurized])
    # import pdb
    # pdb.set_trace()
    # TODO add metafeatures
    # TODO optionally perform metafeature cross-product
    A_to_B, B_to_A = reflect_data(A_to_B, B_to_A)
    relative = relative_features(A_to_B, B_to_A)
    return postprocess(relative)
