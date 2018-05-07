#!usr/bin/python

################################################################################
# use an isolation forest to visualize the anomaly scores in our given data
#
# ultimately, this visualiztion aims to aid in selecting a contamination
# threshold for filtering out outliers
#
# shane ryan
################################################################################

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
from data_utils import *

# location of training data
REG_TRAIN_DATA='../data/regression_train.data'

# build data frame of feature vals and target vals
model_data = import_pandas_data(REG_TRAIN_DATA)

# split dataset into testing / training for k-folds validation
num_features = len(model_data.columns) - 1
feature_data = model_data.loc[:, 0:num_features-1].values
class_data = model_data.loc[:, num_features].values

# outlier filter
rng = np.random.RandomState(42)
outlier_filter = IsolationForest(random_state=rng)
outlier_filter.fit(feature_data)

outlier_prediction = outlier_filter.predict(feature_data)
outlier_map = list(map(lambda x: x == 1, outlier_prediction))
outlier_count = 0
for x in outlier_map:
    if not x:
        outlier_count += 1

print("Found %d outliers" % outlier_count)

filtered_feature_data = feature_data[outlier_map, ]
filtered_class_data = class_data[outlier_map, ]

# plot results
plt.subplot(1,2,1)
plt.scatter(range(len(class_data)), class_data,
        color='black')
plt.xlabel('Individual Feature Vectors')
plt.ylabel('Prediction Output')
plt.subplot(1,2,2)
plt.scatter(range(len(filtered_class_data)), filtered_class_data,
        color='blue')
plt.suptitle("Outlier Detection - Before and After\nOutliers Found: %d"
        % outlier_count)
plt.xlabel('Individual Feature Vectors')
plt.ylabel('Prediction Output')

plt.show()





