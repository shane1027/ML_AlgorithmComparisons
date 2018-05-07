"""
========================
Plotting Learning Curves
========================

Largely taken from sklearn learning curve example
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import (
        AdaBoostRegressor,
        RandomForestRegressor,
        BaggingRegressor )
from data_utils import *


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# location of training data
REG_TRAIN_DATA='regression_train.data'

# build data frame of feature vals and target vals
model_data = import_pandas_data(REG_TRAIN_DATA)

# split dataset into testing / training for k-folds validation
num_features = len(model_data.columns) - 1
contam = 0.01
feature_data = model_data.loc[:, 0:num_features-1].values
class_data = model_data.loc[:, num_features].values

# apply outlier filtering to feature values
X,y = filter_outliers(feature_data, class_data, contam)

# scale feature data
scaler = StandardScaler().fit(feature_data)
X = scaler.transform(X)

title = "Learning Curves (Bagging Random Forest)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)

estimator = BaggingRegressor(RandomForestRegressor(n_jobs=-1))
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves (AdaBoost Random Forest)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
estimator = AdaBoostRegressor(RandomForestRegressor(n_jobs=-1))
plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
