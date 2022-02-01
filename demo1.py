import pandas as pd
import numpy as np
import os
import urllib.request

zip_file_name = "data/zip.test.gz"
if not os.path.exists(zip_file_name):
    url = "https://github.com/tdhock/cs570-spring-2022/raw/master/data/zip.test.gz"
    urllib.request.urlretrieve(url, zip_file_name)


zip_df = pd.read_csv(
    zip_file_name,
    header=None,
    sep=" "
)

zip_features = zip_df.loc[:, 1:].to_numpy()
zip_labels = zip_df[0].to_numpy()
n_folds = 5
# np.random.seed(1)
fold_vec = np.random.randint(low=1, high=n_folds+1, size=zip_labels.size)
test_fold = 0
is_set_dict = {
    "test": fold_vec == test_fold,
    "train": fold_vec != test_fold
}

set_features = {
    set_name: zip_features[is_set, :] for set_name, is_set in is_set_dict.items()
}
shapes = {
    set_name: array.shape for set_name, array in set_features.items()
}
print(shapes)
set_labels = {
    set_name: zip_labels[is_set] for set_name, is_set in is_set_dict.items()
}
print(set_labels)

test_i = 0
test_i_features = set_features["test"][test_i, :]
# matrix of differences, element-wise
diff_mat = set_features["train"] - test_i_features
squared_diff_mat = diff_mat ** 2
# squared_diff_mat.sum(axis=0) # sum of columns
distance_vec = squared_diff_mat.sum(axis=1)  # sum over rows

n_neighbors = 10
sorted_indices = distance_vec.argsort()
nearest_indices = sorted_indices[:n_neighbors]
set_labels["train"][nearest_indices]
