#!usr/bin/python

######################################################################
# characterize the performance of importing datasets into memory
# for numpy vs. pandas over various sizes of datasets
#
# shane ryan
######################################################################

#TODO: replicate each of these datasets by 100 and by 10000 for medium
# and large dataset import comparisons

import timeit

LOOP = 10
SMALL_INT_FILE = '../data/classification_train.data'
SMALL_MIXED_FILE = '../data/regression_train.data'
#MEDIUM_FILE
#LARGE_FILE

times = list()
names = ["Numpy int", "Numpy mixed",
        "Pandas int", "Pandas mixed"]

# characterize small datasets
times.append(timeit.timeit("import_int_data(\"" + SMALL_INT_FILE + "\")",
    number=LOOP, setup="from data_utils import import_int_data"))
times.append(timeit.timeit("import_mixed_data(\"" + SMALL_MIXED_FILE + "\")",
    number=LOOP, setup="from data_utils import import_mixed_data"))
times.append(timeit.timeit("import_pandas_data(\"" + SMALL_INT_FILE + "\")",
    number=LOOP, setup="from data_utils import import_pandas_data"))
times.append(timeit.timeit("import_pandas_data(\"" + SMALL_MIXED_FILE + "\")",
    number=LOOP, setup="from data_utils import import_pandas_data"))

print("Small Dataset Results:\n")

for n,time in enumerate(times):
    print("{}:\t\t{}".format(names[n], time))


