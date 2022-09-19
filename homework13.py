#
# CS 570 Spring 2022
# Homework week 13 - Jadon Fowler
#
import os.path
import random
import time
import urllib.request
from typing import List, Dict, Any
import itertools
from collections import ChainMap

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
from torch import nn, optim
import torchtext
import torch.nn.functional as F

# returns tuple of feature list and label list
def parse_features(dataset):
    items_list = list(dataset)
    items_list = random.sample(items_list, 50)

    word_dict = {}
    split_list = []
    for label, text in items_list:
        # TODO maybe remove HTML tags + punctuation from text before
        # splitting into words, to reduce noise.
        word_list = text.split()
        for word in word_list:
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
        item = label, word_list
        split_list.append(item)

    # TODO for increased prediction accuracy use a larger dictionary (but
    # that is slower for the demo in class).
    frequent_word_dict = {
        word: count
        for word, count in word_dict.items()
        if count > 50
    }
    print(frequent_word_dict)
    word_to_ix = {
        word: ix
        for ix, word in enumerate(frequent_word_dict)
    }

    feature_list = []
    label_list = []
    word_lists = []
    for label, word_list in split_list:
        word_vec = np.zeros(len(word_to_ix))
        for word in word_list:
            if word in word_to_ix:
                ix = word_to_ix[word]
                word_vec[ix] += 1
        feature_list.append(word_vec)
        label_list.append(1 if label is "pos" else 0)
        word_lists.append(np.array([word_to_ix.get(w, len(word_to_ix)) for w in word_list]))
    return np.array(feature_list), np.array(label_list), frequent_word_dict, word_to_ix, np.array(word_lists)


data_dict = {
    "IMDB": parse_features(torchtext.datasets.IMDB(split="train")),
    # getting "RuntimeError: Internal error: headers don't contain content-disposition."
    # when trying to use these datasets
#    "Yelp": parse_features(torchtext.datasets.YelpReviewPolarity(split="train")),
#    "Amazon": parse_features(torchtext.datasets.AmazonReviewPolarity(split="train")),
}


def prepare_sequence(seq, to_ix, default):
    idxs = [to_ix.get(w, default) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, (hidden, cell) = self.lstm(embeds.view(len(sentence), 1, -1))
        pred_score = self.hidden2tag(hidden.reshape(1, self.hidden_dim))
        pred_score = torch.squeeze(pred_score, 1)
        return torch.sigmoid(pred_score)


def prepare_sequence(seq, to_ix):
    print(seq), print(to_ix)
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class MyLSTM:
    def __init__(self, word_to_ix, tag_to_ix, max_epochs, embed_dim=4, hidden_dim=3, batch_size=20, step_size=0.01):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.model = LSTMTagger(embed_dim, hidden_dim, len(word_to_ix)+1, 1)
        self.loss_fun = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=step_size)
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def take_step(self, X, y):
        # Compute prediction error
        losses = []
        correct = 0
        for sentence, label in zip(X, y):
            self.model.zero_grad()
            pred = self.model(torch.tensor(sentence, dtype=torch.long))
            label_tensor = torch.tensor([float(label)])
            pred_label = np.where(pred > 0, 1, 0)[0]
            loss = self.loss_fun(pred, label_tensor)

            loss.backward()
            self.optimizer.step()
            correct += 1 if (pred_label == label) else 0
            losses.append(loss.item())
            #print(loss.item(), pred, pred_label, label, correct)
        print("Loss:", np.mean(losses), "Correct", correct, "/", len(X))
        return np.array(losses).mean(), correct

    def fit(self, X, y):
        # randomize the indices for batches
        indx = list(range(len(X)))
        random.shuffle(indx)
        self.model.train()
        # print("training MLP w/ ", self.max_epochs)
        ret = {}
        for epoch in range(self.max_epochs):
            total_loss = 0
            correct = 0
            for i in range(0, len(X), self.batch_size):
                # print("epoch: ", epoch, " i: ", i)
                # get the random batch
                batch_indices = indx[i:i + self.batch_size]
                # print("  batch_indices", batch_indices)
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                (loss, c) = self.take_step(batch_X, batch_y)
                total_loss += loss
                correct += c
            total_loss /= (len(X) / self.batch_size)
            correct /= len(X)
            ret = {'loss': total_loss, 'accuracy': correct}
        print(self.max_epochs, ret)
        return ret

    def predict(self, X):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for sentence in X:
                pred = self.model(torch.tensor(sentence, dtype=torch.long))
                pred_label = np.where(pred > 0, 1, 0)[0]
                preds.append(pred_label)
        return np.array(preds)


class MyCV:
    def __init__(self, estimator, param_grid: List[Dict[str, Any]], cv=3):
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
        print("cross validating with", len(self.param_grid), "hyperparam sets")
        completed = 0
        for hyperparams in self.param_grid:
            print("cross-validating with hyperparams", hyperparams)
            time_start = time.time()
            # set the params for the estimator
            for param_key, param_value in hyperparams.items():
                setattr(self.estimator, param_key, param_value)
            # split the train features into subtrain & validation sets
            train_split = KFold(n_splits=self.cv, shuffle=True, random_state=1).split(train_features)

            validation_losses = []
            subtrain_losses = []
            for train_fold_id, (subtrain_indices, validation_indices) in enumerate(train_split):
                print("  fold", train_fold_id)
                # split features & labels into subtrain & validation
                subtrain_features = train_features[subtrain_indices]
                subtrain_labels = train_labels[subtrain_indices]
                validation_features = train_features[validation_indices]
                validation_labels = train_labels[validation_indices]
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

            time_taken = time.time() - time_start
            completed += 1
            print(" done cross validating in", time_taken, "seconds. ", completed, "/", len(self.param_grid))

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


# create a list of all the hyperparameter combinations
def combine_hyperparams(names, values):
    hyperparams = []
    for item in itertools.product(*values):
        hyperparams.append(dict(ChainMap(*[{name: value} for name, value in list(zip(names, item))])))
    return hyperparams


# Full Runs
def full_run(params_of_interest):
    loss_results = []
    for data_set, data in data_dict.items():
        # HW 6
        time_start = time.time()
        print("Running MyLSTM...")
        features = data[0]
        labels = data[1]
        frequent_word_dict, word_to_ix, word_lists = data[2], data[3], data[4]
        tag_to_ix = {"pos": 1, "neg": 0}
        step_size = 0.1
        batch_size = 30
        max_epochs = [10, 30, 50]#[50, 100, 400, 600, 900, 1100]
        # max_epochs = [200] * len(hidden_layers)
        # hidden_layers = [3] * len(max_epochs)
        # max_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160,
        #               180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
        #               420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
        #               660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880,
        #               900, 920, 940, 960, 980, 1000]

        # max_epochs = [200, 400, 600, 800, 1000, 2000, 3000]
        batch_sizes = [5, 10, 100]
        step_sizes = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]

        hyperparams = combine_hyperparams(
            ["max_epochs"],
            [max_epochs]
        )

        mlp = MyLSTM(word_to_ix=word_to_ix, tag_to_ix=tag_to_ix, max_epochs=250, batch_size=batch_size, step_size=step_size)
        automlp_cv = MyCV(mlp, cv=3, param_grid=hyperparams)
        #auto_mlp = make_pipeline(StandardScaler(), automlp_cv)
        automlp_cv.fit(word_lists, labels)

        # record loss
        print("loss results =", automlp_cv.loss_results)
        for result in automlp_cv.loss_results:
            loss_results.append(pd.DataFrame({
                **result,
                "data_set": data_set,
             #   "optname": result["opt"].__str__(),
            }, index=[0]))

        print("best_params_ =", automlp_cv.best_params_)
        print("fit_results =", automlp_cv.fit_results)
        time_taken = time.time() - time_start
        print("Done running MyLSTM in", time_taken, "seconds.")

    loss_df = pd.concat(loss_results)
    for param_of_interest in params_of_interest:
        # get a view of the dataframe with only the param_of_interest
        param_view = loss_df.groupby(["train_set", "data_set", param_of_interest])["loss"].mean().reset_index()
        gg = p9.ggplot() + \
             p9.geom_line(
                 p9.aes(
                     x=param_of_interest,
                     y="loss",
                     group="train_set",
                     color="train_set",
                 ),
                 data=param_view) + \
             p9.facet_grid("data_set ~ .") + \
             p9.theme(figure_size=(24, 16))
        gg.save("homework13_" + param_of_interest + "_loss.png")
    pass


# Experiments
def experiments():
    test_acc_df_list = []
    for data_set, (input_data, output_labels, frequent_word_dict, word_to_ix, word_lists) in data_dict.items():
        print(data_set, type(input_data), type(output_labels))
        # split the data set into K training sets
        k_fold_split = KFold(n_splits=3, shuffle=True, random_state=1).split(input_data)
        for fold_id, indices in enumerate(k_fold_split):
            mapped_data = {}
            for name, split_indices in zip(["train", "test"], indices):
                mapped_data[name] = {
                    "X": input_data[split_indices],
                    "y": output_labels[split_indices],
                    "word_lists": word_lists[split_indices],
                }
            train_data = mapped_data["train"]["X"]
            train_labels = mapped_data["train"]["y"]
            train_word_lists = mapped_data["train"]["word_lists"]

            # HW 10
            time_start = time.time()
            print("Running MyLSTM...")
            tag_to_ix = {"pos": 1, "neg": 0}
            step_size = 0.1
            batch_size = 30
            max_epochs = [10, 30, 50]  # [50, 100, 400, 600, 900, 1100]
            # max_epochs = [200] * len(hidden_layers)
            # hidden_layers = [3] * len(max_epochs)
            # max_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160,
            #               180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
            #               420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
            #               660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880,
            #               900, 920, 940, 960, 980, 1000]

            # max_epochs = [200, 400, 600, 800, 1000, 2000, 3000]
            batch_sizes = [5, 10, 100]
            step_sizes = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]

            hyperparams = combine_hyperparams(
                ["max_epochs"],
                [max_epochs]
            )

            mlp = MyLSTM(word_to_ix=word_to_ix, tag_to_ix=tag_to_ix, max_epochs=250, batch_size=batch_size,
                         step_size=step_size)
            cv = MyCV(mlp, cv=3, param_grid=hyperparams)
            # auto_mlp = make_pipeline(StandardScaler(), automlp_cv)
            cv.fit(train_word_lists, train_labels)
            print(cv.best_params_)
            time_taken = time.time() - time_start
            print("Done running MyLSTM in", time_taken, "seconds.")

            # setup grid search with training data
            nearest_neighbor = GridSearchCV(KNeighborsClassifier(), {
                'n_neighbors': [x for x in range(1, 21)]
            })
            nearest_neighbor.fit(train_data, train_labels)
            # scaled version
            nearest_neighbor_scaled = make_pipeline(StandardScaler(), GridSearchCV(KNeighborsClassifier(), {
                'n_neighbors': [x for x in range(1, 21)]
            }))
            nearest_neighbor_scaled.fit(train_data, train_labels)

            # print("best params for train: ", nearest_neighbor.best_params_)
            results = pd.DataFrame(nearest_neighbor.cv_results_)

            # setup logistic regression with training data
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100))
            pipe.fit(train_data, train_labels)

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
            test_word_lists = mapped_data["test"]["word_lists"]
            pred_dict = {
                # "nearest_neighbors": nearest_neighbor.predict(test_data),
                "nearest_neighbors_scaled": nearest_neighbor_scaled.predict(test_data),
                # "my_nearest_neighbors": my_knn.predict(test_data),
                # "my_nearest_neighbors_scaled": my_knn_scaled.predict(test_data),
                # "my_logreg": my_logreg.predict(test_data),
                "lstm": cv.predict(test_word_lists),
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
    gg.save("homework13_comparison.png")


#full_run(["max_epochs"])
experiments()
