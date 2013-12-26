import numpy as np
from scipy.stats import skew, kurtosis, shapiro
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, average_precision_score, f1_score, hinge_loss, matthews_corrcoef, precision_score, recall_score, zero_one_loss

from boomlet.utils import aggregators
from boomlet.metrics import max_error, error_variance, relative_error_variance, gini_loss, categorical_gini_loss
from boomlet.transform.type_conversion import Discretizer

from feature_functions import *


"""
Functions used to combine a list of features into one coherent one.

Sample use:
1. to convert categorical to numerical, we perform a one hot encoding
2. treat each binary column as a separate numerical feature
3. compute numerical features as usual
4. use each of the following functions to create a new feature
   (with the input as the nth feature for each of the columns)

WARNING: these will be used in various locations throughout the code base
and will result in feature size growing at faster than a linear rate
"""
AGGREGATORS = aggregators.AGGREGATORS.values()

"""
Boolean flags specifying whether or not to perform conversions
"""
CONVERT_TO_NUMERICAL = True
CONVERT_TO_CATEGORICAL = True

"""
Functions that compute a metric on a single 1-D array
"""
UNARY_NUMERICAL_FEATURES = [
    normalized_entropy,
    skew,
    kurtosis,
    np.std,
    shapiro,
]
UNARY_CATEGORICAL_FEATURES = [
    lambda x: len(set(x)),  # number of unique
]

"""
Functions that compute a metric on two 1-D arrays
"""
BINARY_NN_FEATURES = [
]
BINARY_NC_FEATURES = [
]
BINARY_CN_FEATURES = [
]
BINARY_CC_FEATURES = [
]

"""
Dictionaries of input type (e.g. B corresponds to pairs where binary
data is the input) to pairs of converter functions and a boolean flag
of whether or not to aggregate over the output of the converter function

converter functions should have the type signature:
converter(X_raw, X_current_type, Y_raw, Y_type)
where X_raw is the data to convert
"""
NUMERICAL_CONVERTERS = dict(
    N=(lambda x, *args: x, False),  # identity function
    B=(lambda x, *args: x, False),  # identity function
    C=(lambda x, *args: LabelBinarizer().fit_transform(x), True),
)
CATEGORICAL_CONVERTERS = dict(
    N=(lambda x, *args: Discretizer().fit_transform(x), False),
    B=(lambda x, *args: x, False),  # identity function
    C=(lambda x, *args: x, False),  # identity function
)

"""
Estimators used to provide a fit for a variable
"""
REGRESSION_ESTIMATORS = [
    Ridge(),
    LinearRegression(),
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(random_state=0),
    GradientBoostingRegressor(subsample=0.5, n_estimators=10, random_state=0),
    KNeighborsRegressor(),
]
CLASSIFICATION_ESTIMATORS = [
    LogisticRegression(random_state=0),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    GradientBoostingClassifier(subsample=0.5, n_estimators=10, random_state=0),
    KNeighborsClassifier(),
    GaussianNB(),
]

"""
Functions to provide a value of how good a fit on a variable is
"""
REGRESSION_METRICS = [
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
    error_variance,
    relative_error_variance,
    gini_loss,
] + BINARY_NN_FEATURES
REGRESSION_RESIDUAL_METRICS = [
] + UNARY_NUMERICAL_FEATURES
BINARY_PROBABILITY_CLASSIFICATION_METRICS = [
    roc_auc_score,
    hinge_loss,
] + REGRESSION_METRICS
RESIDUAL_PROBABILITY_CLASSIFICATION_METRICS = [
] + REGRESSION_RESIDUAL_METRICS
BINARY_CLASSIFICATION_METRICS = [
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    zero_one_loss,
    categorical_gini_loss,
]
ND_CLASSIFICATION_METRICS = [ # metrics for N-dimensional classification
] + BINARY_CC_FEATURES

"""
Functions to assess the model (e.g. complexity) of the fit on a numerical variable

of type signature:
    metric(clf, X, y)
"""
REGRESSION_MODEL_METRICS = [
    # TODO model complexity metrics
]
# TODO idea:
# dim reduction down to 1 dimension on X
# for different values in the available range (many ways to do this)
# convert back to N-d space and calculate distance between consecutive
# points (twists: relative to y range, also try square / sqrt distance
# as well
CLASSIFICATION_MODEL_METRICS = [
    # TODO
]

"""
The operations to perform on the A->B features and B->A features.
"""
RELATIVE_FEATURES = [
    # Identity functions, comment out the next 2 lines for only relative features
    lambda x, y: x,
    lambda x, y: y,
    lambda x, y: x - y,
]

"""
Whether or not to treat each observation (A,B) as two observations: (A,B) and (B,A)
"""
REFLECT_DATA = True
