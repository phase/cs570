#
# CS 570 Spring 2022
# Homework 3 - Jadon Fowler
#
import pandas as pd
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


# start the main part of HW 3
class MyCV:
    def __init__(self, estimator=MyKNN(), param_grid={'n_neighbors': [x for x in range(1, 10)]}, cv=5):
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
            for param_key, param_value in self.param_grid.items():
                if param_key == "n_neighbors":
                    for n_neighbors in param_value:
                        self.estimator.n_neighbors = n_neighbors

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
                        fit_results[str(train_fold_id) + param_key + str(n_neighbors)] = {
                            param_key: self.estimator.n_neighbors,
                            "accuracy": accuracy
                        }
        # get the best results by max on the accuracy of the results
        best_result = fit_results[max(fit_results, key=lambda k: fit_results.get(k)["accuracy"])]
        self.best_params_ = best_result["n_neighbors"]

    def predict(self, X):
        self.estimator.n_neighbors = self.best_params_
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

        # print results
        print("best params for train: ", nearest_neighbor.best_params_)
        results = pd.DataFrame(nearest_neighbor.cv_results_)

        # setup logistic regression with training data
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100))
        pipe.fit(**mapped_data["train"])

        # create featureless vec
        # either all 0 or all 1 (whichever was more frequent in the train set labels)
        train_set_labels = pd.DataFrame(mapped_data["train"]["y"])
        print("train_set_labels:", train_set_labels, "\ntrain_set_labels.mode: ", train_set_labels.mode())
        mode = train_set_labels.mode().iloc[:, 0].values[0]

        # HW 3 defined classes
        my_knn = MyCV()
        my_knn.fit(**mapped_data["train"])

        my_knn_scaled = make_pipeline(StandardScaler(), MyCV())
        my_knn_scaled.fit(**mapped_data["train"])

        test_data = mapped_data["test"]["X"]
        pred_dict = {
            "nearest_neighbors": nearest_neighbor.predict(test_data),
            "nearest_neighbors_scaled": nearest_neighbor_scaled.predict(test_data),
            "my_nearest_neighbors": my_knn.predict(test_data),
            "my_nearest_neighbors_scaled": my_knn_scaled.predict(test_data),
            "linear_model": pipe.predict(test_data),
            "featureless": np.repeat(mode, test_data.shape[0])
        }
        for algorithm, pred_vec in pred_dict.items():
            test_acc_dict = {
                "test_accuracy_percent": (pred_vec == mapped_data["test"]["y"]).mean() * 100,
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
gg.save("homework3.png")
