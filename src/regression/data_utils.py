#!usr/bin/python

###############################################################################
# utilities to manipulate data for testing various classifiers / regressors
#
# shane ryan
###############################################################################

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

## after the following functions were timed, pandas was deemed most efficient.
##TODO: move DataFrame slicing into this set of utils in order to return
# feature vectors and class vectors automatically... and scaling functions.

# import data of type int, using numpy
def import_int_data(file):
    data = np.genfromtxt(file, delimiter=',', dtype=int)
    return data

# import data of mixed type, using numpy
def import_mixed_data(file):
    data = np.genfromtxt(file, delimiter=',', dtype=None)
    return data

# import data of any type using pandas returning DataFrame
def import_pandas_data(file):
    df = pd.read_csv(file, sep=',', header=None)
    return df

# filter outliers out of data using isolated forest technique
def filter_outliers(x_input, y_input, contamination_val):
    rng = np.random.RandomState(42)
    outlier_filter = IsolationForest(contamination=contamination_val,
            random_state=rng, n_jobs=-1)
    outlier_filter.fit(x_input)

    outlier_prediction = outlier_filter.predict(x_input)
    outlier_map = list(map(lambda x: x == 1, outlier_prediction))
    outlier_count = 0
    for x in outlier_map:
        if not x:
            outlier_count += 1

    print("Found %d outliers" % outlier_count)

    return x_input[outlier_map, ], y_input[outlier_map, ]


