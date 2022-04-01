#
# CS 570 Spring 2022
# Homework week 10 - Jadon Fowler
#
import os.path
import random
import time
import urllib.request
from typing import List, Dict, Any

import numpy as np
import pandas
import pandas as pd
import plotnine as p9
import sklearn
import torch
from numpy import ndarray, isnan
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

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

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, units_per_layer):
        super(NeuralNetwork, self).__init__()
        layers = []
        for layer_i in range(1, len(units_per_layer)):
            layers.append(nn.Linear(units_per_layer[layer_i - 1], units_per_layer[layer_i]))
            layers.append(nn.ReLU())
        # remove last ReLU layer
        layers.remove(layers[-1])
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class RegularizedMLP:
    def __init__(self, max_epochs, units_per_layer, batch_size=20, step_size=0.01):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.hidden_layers = 1

    def __setup_nn(self):
        if self.hidden_layers >= 1 and len(self.units_per_layer) >= 3:
            # extend units per layer to match hidden layers
            new_units_per_layer = [
                self.units_per_layer[0],
                # repeat middle layer
                *([self.units_per_layer[1]] * self.hidden_layers),
                self.units_per_layer[-1]
            ]
            self.units_per_layer = new_units_per_layer

        self.nn = NeuralNetwork(self.units_per_layer)
        self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.step_size)
        self.loss_fun = nn.BCEWithLogitsLoss()

    def take_step(self, X, y):
        # Compute prediction error
        pred = self.nn(torch.Tensor(X))
        loss = self.loss_fun(pred, torch.Tensor(np.array(y)).reshape(len(y), 1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        correct = (pred.argmax(1) == y).sum().item()
        return loss.item(), correct

    def fit(self, X, y):
        self.__setup_nn()
        # randomize the indices for batches
        indx = list(range(len(X)))
        random.shuffle(indx)
        self.nn.train()
        # print("training MLP w/ ", self.max_epochs)
        ret = {}
        for epoch in range(self.max_epochs):
            total_loss = 0
            correct = 0
            for i in range(0, len(X), self.batch_size):
                # get the random batch
                batch_indices = indx[i:i + self.batch_size]
                # print("  batch_indices", batch_indices)
                batch_X = X[batch_indices]
                batch_y = y.iloc[batch_indices]
                (loss, c) = self.take_step(batch_X, batch_y)
                total_loss += loss
                correct += c
            total_loss /= (len(X) / self.batch_size)
            correct /= len(X)
            ret = {'loss': total_loss, 'accuracy': correct}
        # print(self.max_epochs, ret)
        return ret

    def decision_function(self, X):
        self.nn.eval()
        with torch.no_grad():
            return self.nn(torch.Tensor(X)).numpy().ravel()

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, 1, 0)


class MyCV:
    def __init__(self, estimator, param_grid: List[Dict[str, Any]], cv=4):
        self.loss_results = None
        self.fit_results = None
        self.best_params_ = None
        self.cv = cv
        self.estimator = estimator
        self.param_grid = param_grid

    def fit(self, X, y):
        # giving these more descriptive names
        train_features = X
        train_labels = y
        loss_results = []
        fit_results = {}

        # go over the param sets we want to maximize
        for hyperparams in self.param_grid:
            print("cross-validating with hyperparams", hyperparams)
            # set the params for the estimator
            for param_key, param_value in hyperparams.items():
                setattr(self.estimator, param_key, param_value)
            # split the train features into subtrain & validation sets
            train_split = KFold(n_splits=self.cv, shuffle=True, random_state=1).split(train_features)

            validation_losses = []
            subtrain_losses = []
            for train_fold_id, (subtrain_indices, validation_indices) in enumerate(train_split):
                # split features & labels into subtrain & validation
                subtrain_features = train_features[subtrain_indices]
                subtrain_labels = train_labels.iloc[subtrain_indices]
                validation_features = train_features[validation_indices]
                validation_labels = train_labels.iloc[validation_indices]
                # get the accuracy of the estimator by fitting it to the subtrain
                # set and predicting the validation set labels
                self.estimator.fit(subtrain_features, subtrain_labels)
                estimated_subtrain_labels = self.estimator.predict(subtrain_features)
                estimated_validated_labels = self.estimator.predict(validation_features)
                accuracy = (estimated_validated_labels == validation_labels).mean() * 100
                # log_loss = np.log(1 + np.exp(np.matmul(-estimated_validated_labels, validation_labels)))
                subtrain_loss = sklearn.metrics.log_loss(subtrain_labels, estimated_subtrain_labels)
                validation_loss = sklearn.metrics.log_loss(validation_labels, estimated_validated_labels)

                loss = {
                    "validation": validation_loss,
                    "subtrain": subtrain_loss
                }
                validation_losses.append(validation_loss)
                subtrain_losses.append(subtrain_loss)
                result = {
                    "train_fold": train_fold_id,
                    "accuracy": accuracy,
                    "loss": loss,
                }
                print(" train fold", train_fold_id, "loss =", loss)
                for param_key, param_value in hyperparams.items():
                    result[param_key] = getattr(self.estimator, param_key)
                fit_results[str(train_fold_id) + str(hyperparams)] = result
            # record results for these hyperparams
            validation_loss = np.min(validation_losses)
            loss_results.append({
                "loss": validation_loss,
                "train_set": "validation",
                **hyperparams,
            })
            subtrain_loss = np.min(subtrain_losses)
            loss_results.append({
                "loss": subtrain_loss,
                "train_set": "subtrain",
                **hyperparams,
            })

        # get the best results by max on the accuracy of the results
        self.fit_results = fit_results
        self.loss_results = loss_results
        best_result = fit_results[max(fit_results, key=lambda k: fit_results.get(k)["accuracy"])]
        self.best_params_ = best_result

        # actually fit
        for key, value in best_result.items():
            setattr(self.estimator, key, value)
        self.estimator.fit(X, y)

    def predict(self, X):
        for param, value in self.best_params_.items():
            setattr(self.estimator, param, value)
        return self.estimator.predict(X)


# Full Runs
def full_run(param_of_interest):
    loss_results = []
    for data_set, data in data_dict.items():
        # HW 6
        time_start = time.time()
        print("Running RegularizedMLP...")
        train_data = data[0]
        mlp_layers = [train_data.shape[1], 10, 1]
        step_size = 0.1
        batch_size = 5
        #hidden_layers = [0, 2, 3, 4, 6, 7, 8]
        if data_set == "spam":
            print(train_data.shape)
            mlp_layers = [train_data.shape[1], 5, 1]
            step_size = 0.005
            batch_size = 10
            # hidden_layers = [0, 1, 2, 3, 4, 5, 6]
        max_epochs = [50, 100, 250, 400, 500, 600, 750, 900, 1000, 1100]
        # max_epochs = [200] * len(hidden_layers)
        hidden_layers = [3] * len(max_epochs)
        # max_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160,
        #               180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
        #               420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
        #               660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880,
        #               900, 920, 940, 960, 980, 1000]

        # max_epochs = [200, 400, 600, 800, 1000, 2000, 3000]
        batch_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        step_sizes = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        hyperparams = [{'hidden_layers': h, 'max_epochs': m} for h, m in zip(hidden_layers, max_epochs)]
        mlp = RegularizedMLP(max_epochs=250, units_per_layer=mlp_layers, batch_size=batch_size, step_size=step_size)
        automlp_cv = MyCV(mlp, cv=2, param_grid=hyperparams)
        auto_mlp = make_pipeline(StandardScaler(), automlp_cv)
        auto_mlp.fit(data[0], data[1])

        # record loss
        print("loss results =", automlp_cv.loss_results)
        for result in automlp_cv.loss_results:
            loss_results.append(pd.DataFrame({
                **result,
                "data_set": data_set,
            }, index=[0]))

        print("best_params_ =", automlp_cv.best_params_)
        print("fit_results =", automlp_cv.fit_results)
        time_taken = time.time() - time_start
        print("Done running RegularizedMLP in", time_taken, "seconds.")

    loss_df = pd.concat(loss_results)
    gg = p9.ggplot() + \
         p9.geom_line(
             p9.aes(
                 x=param_of_interest,
                 y="loss",
                 group="train_set",
                 color="train_set",
             ),
             data=loss_df) + \
         p9.facet_grid(". ~ data_set")
    gg.save("homework10_" + param_of_interest + "_loss.png")
    return loss_df


# Experiments
def experiments():
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
            train_data = mapped_data["train"]["X"]
            # HW 10
            time_start = time.time()
            print("Running RegularizedMLP...")
            mlp_layers = [train_data.shape[1], 10, 1]
            step_size = 0.1
            batch_size = 5
            hidden_layers = [0, 2, 3, 4, 6, 7, 8]
            if data_set == "spam":
                print(train_data.shape)
                mlp_layers = [train_data.shape[1], 5, 1]
                step_size = 0.005
                batch_size = 10
                hidden_layers = [0, 1, 2, 3, 4, 5, 6]
            # max_epochs = [50, 100, 250, 400, 500, 600, 750, 900, 1000, 1100]
            max_epochs = [200] * len(hidden_layers)
            # max_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160,
            #               180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
            #               420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
            #               660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880,
            #               900, 920, 940, 960, 980, 1000]

            # max_epochs = [200, 400, 600, 800, 1000, 2000, 3000]
            batch_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            step_sizes = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
            hyperparams = [{'hidden_layers': h, 'max_epochs': m} for h, m in zip(hidden_layers, max_epochs)]

            torch_mlp = RegularizedMLP(max_epochs=250, units_per_layer=mlp_layers, batch_size=batch_size, step_size=step_size)
            torch_mlp_cv = MyCV(torch_mlp, cv=2, param_grid=hyperparams)
            torch_pipeline = make_pipeline(StandardScaler(), torch_mlp_cv)
            torch_pipeline.fit(**mapped_data["train"])
            print(torch_mlp_cv.best_params_)
            time_taken = time.time() - time_start
            print("Done running RegularizedMLP in", time_taken, "seconds.")

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
                # "nearest_neighbors": nearest_neighbor.predict(test_data),
                "nearest_neighbors_scaled": nearest_neighbor_scaled.predict(test_data),
                # "my_nearest_neighbors": my_knn.predict(test_data),
                # "my_nearest_neighbors_scaled": my_knn_scaled.predict(test_data),
                # "my_logreg": my_logreg.predict(test_data),
                "torch_mlp": torch_pipeline.predict(test_data),
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
                print(test_acc_dict)
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
    gg.save("homework10_comparison.png")


#full_run("hidden_layers")
full_run("max_epochs")
experiments()
