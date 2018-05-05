#!usr/bin/python

###############################################################################
# utilities to manipulate data for testing various classifiers / regressors
#
# shane ryan
###############################################################################

import numpy as np
import pandas as pd

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
