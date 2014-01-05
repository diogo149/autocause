from sklearn.cluster import MiniBatchKMeans, Ward, MeanShift, SpectralClustering, AffinityPropagation, DBSCAN
from sklearn.decomposition import RandomizedPCA, FastICA

from boomlet.contextmanagers import seed_random
from boomlet.transform.type_conversion import Discretizer, BinaryTransformer, Clusterizer, DiscreteConstantPredictor, DiscreteOrdinalPredictor


def identity(X_raw, X_current_type, Y_raw, Y_type):
    return X_raw


def discretizer(levels=10):
    def converter(X_raw, X_current_type, Y_raw, Y_type):
        assert X_current_type == "N"
        return Discretizer(levels).fit_transform(X_raw)
    return converter


def clusterizer(clusterer, random_seed=0):
    def converter(X_raw, X_current_type, Y_raw, Y_type):
        assert X_current_type == "N"
        with seed_random(random_seed):
            return Clusterizer(clusterer).fit_transform(X_raw)
    return converter


def binary_transformer(transformer=None, random_seed=0):
    def converter(X_raw, X_current_type, Y_raw, Y_type):
        assert X_current_type == "C"
        with seed_random(random_seed):
            return BinaryTransformer(transformer).fit_transform(X_raw)
    return converter


def discrete_constant_predictor(aggregator="mean"):
    def converter(X_raw, X_current_type, Y_raw, Y_type):
        assert X_current_type == "C"
        return DiscreteConstantPredictor(aggregator).fit_transform(X_raw, Y_raw)
    return converter


def discrete_ordinal_predictor(aggregator="mean"):
    def converter(X_raw, X_current_type, Y_raw, Y_type):
        assert X_current_type == "C"
        return DiscreteOrdinalPredictor(aggregator).fit_transform(X_raw, Y_raw)
    return converter


# this somewhat repetitive route for keeping track of converters
# was chosen in order to make it easier to reproduce results
NUMERICAL_TO_NUMERICAL = dict(
    identity=identity,
    discretizer3=discretizer(3),
    discretizer10=discretizer(10),
)

NUMERICAL_TO_CATEGORICAL = dict(
    discretizer3=discretizer(3),
    discretizer10=discretizer(10),
    kmeans3=clusterizer(MiniBatchKMeans(3)),
    kmeans10=clusterizer(MiniBatchKMeans(10)),
    ward3=clusterizer(Ward(3)),
    ward10=clusterizer(Ward(10)),
    meanshift=clusterizer(MeanShift()),
    # spectral3=clusterizer(SpectralClustering(3)),  # FIXME
    # spectral10=clusterizer(SpectralClustering(10)), # FIXME
    affinity_prop=clusterizer(AffinityPropagation()),
    dbscan=clusterizer(DBSCAN()),
)

BINARY_TO_NUMERICAL = dict(
    identity=identity,
)

BINARY_TO_CATEGORICAL = dict(
    identity=identity,
)

CATEGORICAL_TO_NUMERICAL = dict(
    noop=identity,
    binarize=binary_transformer(),
    pca1=binary_transformer(RandomizedPCA(1)),
    # ica1=binary_transformer(FastICA(1)),  # FIXME
    median_ordinal_pred=discrete_ordinal_predictor("median"),
    mean_ordinal_pred=discrete_ordinal_predictor("mean"),
    max_ordinal_pred=discrete_ordinal_predictor("max"),
    min_ordinal_pred=discrete_ordinal_predictor("min"),
)

CATEGORICAL_TO_CATEGORICAL = dict(
    identity=identity,
)
