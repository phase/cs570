#
# CS 570 Spring 2022
# Homework 4 - Jadon Fowler
#
import pandas
import pandas as pd
from numpy import NaN, ndarray, isnan, logical_not
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import plotnine as p9
import urllib.request
import os.path

# download the test data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

zip_file_name = "data/zip.test.gz"
if not os.path.exists(zip_file_name):
    url = "https://github.com/tdhock/cs570-spring-2022/raw/master/data/zip.test.gz"
    urllib.request.urlretrieve(url, zip_file_name)

# download the spam data
spam_file_name = "data/spam.data"
if not os.path.exists(spam_file_name):
    url = "https://github.com/tdhock/cs570-spring-2022/raw/master/data/spam.data"
    urllib.request.urlretrieve(url, spam_file_name)

zip_df = pd.read_csv(
    zip_file_name,
    header=None,
    sep=" "
)

spam_df = pd.read_csv(
    spam_file_name,
    header=None,
    sep=" "
)

zip_df = zip_df.loc[zip_df[0].isin([0, 1]), :]
spam_df = spam_df.loc[spam_df.iloc[:, -1].isin([0, 1]), :]

data_dict = {
    "zip": (zip_df.loc[:, 1:].to_numpy(), zip_df[0]),
    "spam": (spam_df.iloc[:, :-1].to_numpy(), spam_df.iloc[:, -1]),
}

print(data_dict)


class MyKNN:
    def __init__(self, n_neighbors=5):
        self.train_features = None
        self.train_labels = None
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.train_features = X
        self.train_labels = y

    def predict(self, X):
        n_test = len(X)
        test_predictions = np.empty(n_test)
        for test_i in range(n_test):
            test_i_features = X[test_i, :]
            diff_mat = self.train_features - test_i_features
            squared_diff_mat = diff_mat ** 2
            distance_vec = squared_diff_mat.sum(axis=1)  # sum over rows
            sorted_indices = distance_vec.argsort()
            nearest_indices = sorted_indices[:self.n_neighbors]
            nearest_labels = self.train_labels.iloc[nearest_indices]

            # predicted probability and class
            pred_class = pd.Series(nearest_labels).value_counts().idxmax()
            test_predictions[test_i] = pred_class
        return test_predictions

class MyLogReg:
    def __init__(self, max_iterations, step_size=0.01):
        self.intercept_ = None
        self.coef_ = None
        self.good_indices = None
        self.max_iterations = max_iterations
        self.step_size = step_size

    def fit(self, X: pandas.DataFrame, y: pandas.DataFrame):
        # print("MyLogReg.fit")
        # print(features.mean())
        # print(features.mean(axis=0))
        # print(features.var(axis=0))  # var = sd^2
        # print(np.sqrt(features.var(axis=0)))

        np.random.seed(1)
        original_mean: ndarray = X.mean(axis=0)
        original_sd: ndarray = np.sqrt(X.var(axis=0))

        # filter out 0 SD
        good_indices = np.where(original_sd > 0)[0]
        features = X.T[good_indices].T
        sd = original_sd.T[good_indices].T
        mean = original_mean.T[good_indices].T
        labels = y.replace(0, -1)

        # print("features", features.shape)
        scaled_features: ndarray = ((features - mean) / sd)
        # print("scaled_features", scaled_features.shape)

        nrow, ncol = scaled_features.shape
        weight_vec = np.zeros(ncol+1)
        learn_features = np.column_stack([
            np.repeat(1, nrow),
            np.where(isnan(scaled_features), 0, scaled_features)
        ])
        # print("learn_features", learn_features.shape)

        for iteration in range(self.max_iterations):
            pred_vec = np.matmul(learn_features, weight_vec)
            log_loss = np.log(1 + np.exp(-labels * pred_vec))
            # print("iteration=%d log_loss=%s" % (iteration, log_loss.mean()))
            grad_loss_pred = np.array(-labels / (1 + np.exp(labels * pred_vec)))
            grad_loss_weight_mat = grad_loss_pred * learn_features.T
            grad_vec = grad_loss_weight_mat.sum(axis=1)
            weight_vec -= self.step_size * grad_vec

        self.good_indices = good_indices
        self.coef_ = weight_vec[1:] / sd
        d = self.coef_ * mean
        self.intercept_ = weight_vec[0] - d[np.where(~np.isinf(d))].sum()

    def decision_function(self, X):
        return np.matmul(X, self.coef_) + self.intercept_

    def predict(self, X):
        X2 = X.T[self.good_indices].T
        return np.where(self.decision_function(X2) > 0, 1, 0)


# start the main part of HW 3
class MyCV:
    def __init__(self, estimator=MyKNN(), param_grid={'n_neighbors': [x for x in range(1, 10)]}, cv=5):
        self.fit_results = None
        self.best_params_ = None
        self.cv = cv
        self.estimator = estimator
        self.param_grid = param_grid

    def fit(self, X, y):
        # giving these more descriptive names
        train_features = X
        train_labels = y
        train_split = KFold(n_splits=self.cv, shuffle=True, random_state=1).split(train_features)
        fit_results = {}
        # split the train features into subtrain & validation sets
        for train_fold_id, (subtrain_indices, validation_indices) in enumerate(train_split):
            # go over the params we want to maximize
            for param_key, param_values in self.param_grid.items():
                for param_value in param_values:
                    setattr(self.estimator, param_key, param_value)
                    # print(param_key, getattr(self.estimator, param_key))

                    # split features & labels into subtrain & validation
                    subtrain_features = train_features[subtrain_indices]
                    subtrain_labels = train_labels.iloc[subtrain_indices]
                    validation_features = train_features[validation_indices]
                    validation_labels = train_labels.iloc[validation_indices]
                    # get the accuracy of the estimator by fitting it to the subtrain
                    # set and predicting the validation set labels
                    self.estimator.fit(subtrain_features, subtrain_labels)
                    estimated_validated_labels = self.estimator.predict(validation_features)
                    accuracy = (estimated_validated_labels == validation_labels).mean() * 100
                    fit_results[str(train_fold_id) + param_key + str(param_value)] = {
                        param_key: getattr(self.estimator, param_key),
                        "accuracy": accuracy
                    }
        # get the best results by max on the accuracy of the results
        self.fit_results = fit_results
        best_result = fit_results[max(fit_results, key=lambda k: fit_results.get(k)["accuracy"])]
        self.best_params_ = best_result

    def predict(self, X):
        for param, value in self.best_params_.items():
            setattr(self.estimator, param, value)
        return self.estimator.predict(X)


test_acc_df_list = []
for data_set, (input_data, output_labels) in data_dict.items():
    print(data_set, type(input_data), type(output_labels))
    # split the data set into K training sets
    k_fold_split = KFold(n_splits=3, shuffle=True, random_state=1).split(input_data)
    for fold_id, indices in enumerate(k_fold_split):
        mapped_data = {}
        for name, split_indices in zip(["train", "test"], indices):
            mapped_data[name] = {
                "X": input_data[split_indices],
                "y": output_labels.iloc[split_indices]
            }

        # HW 4
        my_logreg = MyCV(MyLogReg(max_iterations=10), param_grid={'max_iterations': [x for x in [1, 10, 100, 1000]]})
        my_logreg.fit(**mapped_data["train"])

        # setup grid search with training data
        nearest_neighbor = GridSearchCV(KNeighborsClassifier(), {
            'n_neighbors': [x for x in range(1, 21)]
        })
        nearest_neighbor.fit(**mapped_data["train"])
        # scaled version
        nearest_neighbor_scaled = make_pipeline(StandardScaler(), GridSearchCV(KNeighborsClassifier(), {
            'n_neighbors': [x for x in range(1, 21)]
        }))
        nearest_neighbor_scaled.fit(**mapped_data["train"])

        # print("best params for train: ", nearest_neighbor.best_params_)
        results = pd.DataFrame(nearest_neighbor.cv_results_)

        # setup logistic regression with training data
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100))
        pipe.fit(**mapped_data["train"])

        # create featureless vec
        # either all 0 or all 1 (whichever was more frequent in the train set labels)
        train_set_labels = pd.DataFrame(mapped_data["train"]["y"])
        # print("train_set_labels:", train_set_labels, "\ntrain_set_labels.mode: ", train_set_labels.mode())
        mode = train_set_labels.mode().iloc[:, 0].values[0]

        # HW 3 defined classes
        # my_knn = MyCV()
        # my_knn.fit(**mapped_data["train"])s

        # my_knn_scaled = make_pipeline(StandardScaler(), MyCV())
        # my_knn_scaled.fit(**mapped_data["train"])

        test_data = mapped_data["test"]["X"]
        test_labels = mapped_data["test"]["y"]
        pred_dict = {
            "nearest_neighbors": nearest_neighbor.predict(test_data),
            "nearest_neighbors_scaled": nearest_neighbor_scaled.predict(test_data),
            # "my_nearest_neighbors": my_knn.predict(test_data),
            # "my_nearest_neighbors_scaled": my_knn_scaled.predict(test_data),
            "my_logreg": my_logreg.predict(test_data),
            "logistic_regression": pipe.predict(test_data),
            "featureless": np.repeat(mode, test_data.shape[0])
        }
        for algorithm, prediction in pred_dict.items():
            test_acc_dict = {
                "test_accuracy_percent": (prediction == test_labels).mean() * 100,
                "data_set": data_set,
                "fold_id": fold_id,
                "algorithm": algorithm
            }
            test_acc_df_list.append(pd.DataFrame(test_acc_dict, index=[0]))

test_acc_df = pd.concat(test_acc_df_list)
print(test_acc_df)

# make a gg plot
gg = p9.ggplot() + \
     p9.geom_point(
         p9.aes(
             x="test_accuracy_percent",
             y="algorithm",
         ),
         data=test_acc_df) + \
     p9.facet_grid(". ~ data_set")
gg.save("homework4.png")
