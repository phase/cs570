#
# CS 570 Spring 2022
# Homework 2 - Jadon Fowler
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
spam_df = spam_df.loc[spam_df[0].isin([0, 1]), :]

data_dict = {
    "zip": (zip_df.loc[:, 1:].to_numpy(), zip_df[0]),
    "spam": (spam_df.loc[:, 1:].to_numpy(), spam_df[0]),
}

print(data_dict)

test_acc_df_list = []
for data_set, (input_data, output_labels) in data_dict.items():
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
        nearest_neighbor_scaled = make_pipeline(StandardScaler(), nearest_neighbor)
        nearest_neighbor_scaled.fit(**mapped_data["train"])

        # print results
        print("best params for train: ", nearest_neighbor.best_params_)
        results = pd.DataFrame(nearest_neighbor.cv_results_)
        print("results:", results.loc[:, ["param_n_neighbors", "mean_test_score"]])

        # setup logistic regression with training data
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100))
        pipe.fit(**mapped_data["train"])

        # create featureless vec
        # either all 0 or all 1 (whichever was more frequent in the train set labels)
        train_set_labels = pd.DataFrame(mapped_data["train"]["y"])
        mode = train_set_labels.mode()[0].values[0]

        test_data = mapped_data["test"]["X"]
        pred_dict = {
            "nearest_neighbors": nearest_neighbor.predict(test_data),
            "nearest_neighbors_scaled": nearest_neighbor_scaled.predict(test_data),
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
gg.save("homework2.png")
