#!usr/bin/python

################################################################################
# implementation of a random forest decision tree regression algorithm built
# after finding decision trees to be most optimal for the given dataset using
# regression_validation.py and making predictions using regression.py
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
regressor = RandomForestRegressor()
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

# implement cross-validation and report R2 scores
cv_score = cross_val_score(regressor, feature_data, class_data, cv=10)
cv_avg = np.asarray(cv_score).mean()

# report R2 scores
print("Cross-Validation R2 Scores:")
for score in cv_score:
    print(score)
print("Mean: {}".format(cv_avg))

# fit regressor and return a prediction set for valdation data
regressor.fit(feature_data, class_data)
validation_prediction = regressor.predict(validation_feature_data)

# display the feature contributions
print("Feature Weights:\n")
for n, weight in enumerate(regressor.feature_importances_, 1):
    print("Feature %d:\t%f" % (n,weight))

# save resulting class predictions
np.savetxt(PRED_OUT, validation_prediction)

# plot results
plt.scatter(range(len(validation_prediction)), validation_prediction,
        color='black')
# would include the following line if the validation class data was given
#plt.scatter(range(len(validation_prediction)), validation_class_data,
        #color='blue')
plt.title("%s - Prediction Results - R2 Score of %.03f" %
        (regressor_name, cv_avg))
plt.xlabel('Individual Feature Vectors')
plt.ylabel('Prediction Output')

plt.show()





