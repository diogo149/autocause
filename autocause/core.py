import copy
from collections import defaultdict
from imp import load_source

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from boomlet.utils.array import to_2d, column_combinations
from boomlet.utils.estimators import binarizer_from_classifier
from boomlet.transform.preprocessing import InfinityReplacer, InvalidRemover
from boomlet.parallel import pmap

CONFIG = __import__("autocause.autocause_settings").autocause_settings


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
            for aggregator in CONFIG.AGGREGATORS:
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
        funcs = CONFIG.UNARY_NUMERICAL_FEATURES
    elif X_type == "C":
        funcs = CONFIG.UNARY_CATEGORICAL_FEATURES
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
        funcs = CONFIG.BINARY_NN_FEATURES
    elif current_type == "NC":
        funcs = CONFIG.BINARY_NC_FEATURES
    elif current_type == "CN":
        funcs = CONFIG.BINARY_CN_FEATURES
    elif current_type == "CC":
        funcs = CONFIG.BINARY_CC_FEATURES
    else:
        raise Exception("improper type: {}".format(current_type))

    features = []
    for feat in funcs:
        result_append(features, feat(A_feat, B_feat))
    return features


def regression_features(X, y):
    features = []
    for clf in CONFIG.REGRESSION_ESTIMATORS:
        clf = copy.deepcopy(clf)
        clf.fit(X, y)
        for metric in CONFIG.REGRESSION_MODEL_METRICS:
            result_append(features, metric(clf, X, y))
        # TODO predicting in-sample, might be good to change someday
        pred = clf.predict(X)
        for metric in CONFIG.REGRESSION_METRICS:
            result_append(features, metric(y, pred))
        residual = y - pred
        for metric in CONFIG.REGRESSION_RESIDUAL_METRICS:
            result_append(features, metric(residual))
    return features


def classification_features(X, y):
    # special casing when there is only one value of y
    # this is handled this way because we wouldn't be
    # able to provide stubbed values for a trained classifier
    # for the model metric features, and we cannot just
    # fill in 0's because we don't know a priori how many
    # arguments each of the feature functions will return.
    # it may be possible to instead look at the number returned
    # by previous invocations of the functions, but that would
    # a) require code restructuring, b) non-local information and
    # c) might not work all the time
    if len(np.unique(y)) == 1:
        assert len(y.shape) == 1
        # set a new value in y
        y[0] += 1

    def apply_col(idx, f, matrices):
        return f(*[m[:, idx] for m in matrices])

    features = []
    for clf in CONFIG.CLASSIFICATION_ESTIMATORS:
        clf = copy.deepcopy(clf)
        clf.fit(X, y)
        for metric in CONFIG.CLASSIFICATION_MODEL_METRICS:
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
        for metric in CONFIG.BINARY_PROBABILITY_CLASSIFICATION_METRICS:
            assert probs.shape[1] == cols, (cols, probs.shape)
            features += aggregate_apply(apply_col,
                                        range(cols),
                                        f=metric,
                                        matrices=[y_bin, probs])

        residual = y_bin - probs
        for metric in CONFIG.RESIDUAL_PROBABILITY_CLASSIFICATION_METRICS:
            features += aggregate_apply(apply_col,
                                        range(cols),
                                        f=metric,
                                        matrices=[residual])

        pred = clf.predict(X)
        for metric in CONFIG.ND_CLASSIFICATION_METRICS:
            result_append(features, metric(y, pred))

        pred_bin = binarizer.transform(pred).astype(np.float)
        for metric in CONFIG.BINARY_CLASSIFICATION_METRICS:
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
    if CONFIG.CONVERT_TO_NUMERICAL:
        converter = CONFIG.NUMERICAL_CONVERTERS[X_current_type]
        yield (converter(X_raw, X_current_type, Y_raw, Y_type).astype(np.float),
               CONFIG.NUMERICAL_CAN_BE_2D,
               "N")
    if CONFIG.CONVERT_TO_CATEGORICAL:
        converter = CONFIG.CATEGORICAL_CONVERTERS[X_current_type]
        yield (converter(X_raw, X_current_type, Y_raw, Y_type).astype(np.float),
               CONFIG.CATEGORICAL_CAN_BE_2D,
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
    no_inf = InfinityReplacer().transform(array)
    no_nan = InvalidRemover().transform(no_inf)
    return no_nan


def reflect_data(A_to_B, B_to_A):
    """
    returns features equivalent to if the input to the algorithm contained
    B, A for each A, B in the input
    """
    if not CONFIG.REFLECT_DATA:
        return A_to_B, B_to_A
    return np.vstack((A_to_B, B_to_A)), np.vstack((B_to_A, A_to_B))


def relative_features(A_to_B, B_to_A):
    return np.hstack([relative_feat(A_to_B, B_to_A)
                      for relative_feat in CONFIG.RELATIVE_FEATURES])


def metafeatures(pairs):
    """
    generates binary features for data on the types of the pair

    e.g. 1/0 feature on whether or not A is Numerical
    """
    types = [(A_type, B_type) for (_, A_type, _, B_type) in pairs]
    if CONFIG.REFLECT_DATA:
        types += map(lambda x: list(reversed(x)), types)

    features = []
    for A_type, B_type in types:
        # generate features for a single row / observation
        one_feature = []

        def a(elem):
            one_feature.append(elem)

        a(A_type == "N")
        a(A_type == "B")
        a(A_type == "C")
        a(B_type == "N")
        a(B_type == "B")
        a(B_type == "C")

        unordered_combination_type = A_type + B_type
        a(unordered_combination_type == "NN")
        a(unordered_combination_type == "NB")
        a(unordered_combination_type == "NC")
        a(unordered_combination_type == "BN")
        a(unordered_combination_type == "BB")
        a(unordered_combination_type == "BC")
        a(unordered_combination_type == "CN")
        a(unordered_combination_type == "CB")
        a(unordered_combination_type == "CC")

        ordered_combination_type = "".join(sorted(unordered_combination_type))
        a(ordered_combination_type == "NN")
        a(ordered_combination_type == "BN")
        a(ordered_combination_type == "BB")
        a(ordered_combination_type == "BC")
        a(ordered_combination_type == "CN")
        a(ordered_combination_type == "CC")

        features.append(one_feature)

    return np.array(features, dtype=np.float)


def add_metafeatures(pairs, data):
    if not CONFIG.ADD_METAFEATURES:
        return data

    mf = metafeatures(pairs)

    if CONFIG.COMPUTE_METAFEATURE_COMBINATIONS:
        return to_2d(data, mf, column_combinations(data, mf))
    else:
        return to_2d(data, mf)


def load_settings(filepath):
    """
    setting a global configuration object so that the settings don't have
    to be passed around all over the place
    """
    global CONFIG
    CONFIG = load_source("autocause_settings", filepath)


def featurize(pairs, config_path=None):
    """
    takes in input of the form (A, A_type, B, B_type) with A_type and
    B_type in {"N", "C", "B"} for numerical, cateogrical, binary respectively
    """
    if config_path is not None:
        load_settings(config_path)
    featurized = pmap(featurize_pair, pairs)
    A_to_B = np.array([i[0] for i in featurized])
    B_to_A = np.array([i[1] for i in featurized])
    del featurized
    A_to_B, B_to_A = reflect_data(A_to_B, B_to_A)
    relative = relative_features(A_to_B, B_to_A)
    del A_to_B, B_to_A
    with_mf = add_metafeatures(pairs, relative)
    del pairs, relative
    return postprocess(with_mf)


import os
from boomlet.storage import joblib_dump, joblib_load
from boomlet.experimental import folder_apply


PAIRS_PICKLE = "pairs.pkl"


def _featurize_many_helper(config_path):
    """
    Separate helper function so that this can be pickled.
    """
    pairs = joblib_load(PAIRS_PICKLE)
    return featurize(pairs,config_path)


def featurize_many(pairs, config_folder):
    assert not os.path.exists(PAIRS_PICKLE)
    joblib_dump(PAIRS_PICKLE, pairs, compress=9)
    try:
        folder_apply(_featurize_many_helper,
                     config_folder,
                     ext=".py",
                     parallel=False)
    finally:
        os.remove(PAIRS_PICKLE)
