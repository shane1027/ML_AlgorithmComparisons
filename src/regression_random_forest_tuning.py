#!usr/bin/python

################################################################################
# implementation of a random forest decision tree regression algorithm built
# after finding decision trees to be most optimal for the given dataset using
# regression_validation.py and making predictions using regression.py
#
# this program is used to experimentally find the optimal tree depth
# and attempt to utilize only the significantly weighted features for regress
#
# shane ryan
################################################################################

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from data_utils import *

# location of training data
REG_TRAIN_DATA='../data/regression_train.data'
REG_TEST_DATA='../data/regression_test.test'

# location to save prediction output
PRED_OUT='../data/random_forest_prediction.txt'

# regression algorithm
regressor_name = 'Random Forest Regressor'

# build data frame of feature vals and target vals
model_data = import_pandas_data(REG_TRAIN_DATA)
validation_data = import_pandas_data(REG_TEST_DATA)

# split dataset into testing / training for k-folds validation
num_features = len(model_data.columns) - 1
feature_data = model_data.loc[:, 0:num_features-1].values
class_data = model_data.loc[:, num_features].values

validation_feature_data = validation_data.loc[:, 0:num_features-1].values
validation_class_data = validation_data.loc[:, num_features].values

# implement cross-validation and report R2 scores over a wide range of possible
# tree depths
cv_avgs = list()
optimal_depth = 0
optimal_r2 = 0
DEPTH = 150
for tree_depth in range(1, DEPTH+1):
    regressor = RandomForestRegressor(max_depth=tree_depth, n_jobs=-1)
    cv_score = cross_val_score(regressor, feature_data, class_data, cv=10)
    cv_avg = np.asarray(cv_score).mean()
    if cv_avg > optimal_r2:
        optimal_r2 = cv_avg
        optimal_depth = tree_depth
    # report avg R2 scores
    print("Mean at depth {}:\t{}".format(tree_depth, cv_avg))
    cv_avgs.append(cv_avg)

# find the avg R2 score without putting a cap on max depth
regressor = RandomForestRegressor()
cv_score = cross_val_score(regressor, feature_data, class_data, cv=10)
cv_avg = np.asarray(cv_score).mean()

# report findings
print("Optimal depth from 1 to %d:\t%d" % (DEPTH, optimal_depth))
print("Optimal R2 score at this depth:\t%f" % optimal_r2)
print("R2 Score for no limit on depth:\t%f" % cv_avg)

# plot results
plt.plot(range(len(cv_avgs)), cv_avgs,
        color='black')
plt.title("%s - R2 Scores vs. Max Tree Depth" %
        (regressor_name))
plt.xlabel('Max Depth\n(No Max Depth R2 Score: %f)' % cv_avg)
plt.ylabel('R2 Scores')

plt.show()





