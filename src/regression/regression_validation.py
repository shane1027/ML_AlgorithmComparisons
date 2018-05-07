#!usr/bin/python

################################################################################
# training and comparison of multiple regression techniques to select optimal
# regressor for the given dataset
#
# shane ryan
################################################################################

from sklearn.model_selection import KFold
from sklearn.linear_model import ( RidgeCV, LinearRegression, LassoCV,
        ElasticNetCV, LogisticRegression )
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt
from data_utils import *

# location of training data
REG_TRAIN_DATA='../../data/regression_train.data'
REG_TEST_DATA='../../data/regression_test.test'

# algorithms to be tested
regessors = [DecisionTreeRegressor(),
        LinearRegression(), RidgeCV(alphas=[0.5, 5.0, 10], fit_intercept=True),
        LassoCV(fit_intercept=True),
        ElasticNetCV(fit_intercept=True, precompute='auto'),
        LogisticRegression(),
        AdaBoostRegressor(RandomForestRegressor(n_jobs=-1))]
reg_name = ["Decision Tree Regressor",
        "Linear Regressor", "Ridge w/ cross-val", "Lasso w/ cross-val",
        "ElasticNet w/ cross-val", "Logistic Regression",
        "AdaBoost Random Forest"]

# build data frame of feature vals and target vals
model_data = import_pandas_data(REG_TRAIN_DATA)
validation_data = import_pandas_data(REG_TEST_DATA)

# note: to retrieve a row in the pandas dataframe: 'data.loc[row_index, :]'
# then append '.values' if a numpy array of values is the desired output rather
# than a newly formatted pandas dataframe... and len(data.columns) gives
# num_columns

# split dataset into testing / training for k-folds validation
num_features = len(model_data.columns) - 1
feature_data = model_data.loc[:, 0:num_features-1].values
class_data = model_data.loc[:, num_features].values

validation_feature_data = validation_data.loc[:, 0:num_features-1].values
validation_class_data = validation_data.loc[:, num_features].values

# apply outlier filtering to feature values
feature_data, class_data = filter_outliers(feature_data, class_data, 0.1)

# scale feature data
scaler = StandardScaler().fit(feature_data)
feature_data = scaler.transform(feature_data)
validation_feature_data = scaler.transform(validation_feature_data)

kf = KFold(n_splits=10)

# train each regressor and test performance
feature_train = list()
feature_test = list()
class_train = list()
class_test = list()

plots = list()
trial_results = list()

i = 1
for train_indices, test_indices in kf.split(feature_data, class_data):

    # clear the testing / training vectors
    feature_train.clear()
    feature_test.clear()
    class_train.clear()
    class_test.clear()
    trial_results.append(list())

    print("Fold {}:".format(i))

    # build new training / testing dataset based on this fold's indices
    for index in train_indices:
        feature_train.append(feature_data[index])
        class_train.append(class_data[index])
    for index in test_indices:
        feature_test.append(feature_data[index])
        class_test.append(class_data[index])

    # for each algorithm, test the dataset / fold
    for n, algo in enumerate(regessors):
        algo.fit(feature_train, class_train)
        predicted = algo.predict(feature_test)
        r2_val = r2_score(class_test, predicted)
        var_val = explained_variance_score(class_test, predicted)
        # for the logistic regression, score() results in the accuracy
        results = algo.score(feature_test, class_test)
        if results >= 0:
            trial_results[i-1].append(r2_val)
        else:
            trial_results[i-1].append(0)
        print("%s has an R2 score of %f and variance score of %f" %
                (reg_name[n], r2_val, var_val))

        # plot the results
        plt.subplot(2,4,n+1)
        plt.scatter(range(len(predicted)), predicted, color='black')
        plt.scatter(range(len(predicted)), class_test, color='blue')
        plt.title(reg_name[n])
        plt.xlabel("R2 Score: %1.03f\nVariance Score: %1.03f" %
                (r2_val, var_val))
    i += 1
    title = plt.suptitle('Algorithm Comparison', fontsize="x-large")
    title.set_y(0.95)
    plt.subplots_adjust(hspace=0.35)
    # uncomment the following line to see a scatter plot of each regression
    # plt.show()

bar_width = 1 / (2.5*(len(reg_name) + 1))
opacity = 0.8
index = np.arange(len(reg_name))
plt.clf()
plt.subplots()
plt.xlabel('Regression Algorithm')
plt.ylabel('R2 Score')
plt.title('R2 Scores Over All Folds', fontsize='x-large')
plt.xticks(index + bar_width, reg_name)

for n, vec in enumerate(trial_results):
    plt.bar(index + bar_width*n, vec, bar_width,
            alpha=opacity, label='Fold {}'.format(n+1))

plt.legend()
plt.show()


