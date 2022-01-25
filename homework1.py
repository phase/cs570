#
# CS 570 Spring 2022
# Homework 1 - Jadon Fowler
#
import urllib.request
import os.path

import numpy as np
import pandas as pd
import plotnine as p9
import matplotlib

matplotlib.use('cairo')

# download the test data
data_file_name = "data/zip.test.gz"
if not os.path.exists(data_file_name):
    url = "https://github.com/tdhock/cs570-spring-2022/raw/master/data/zip.test.gz"
    urllib.request.urlretrieve(url, data_file_name)

# create a pandas dataframe from the data
zip_df = pd.read_csv(
    data_file_name,
    header=None,
    sep=" ")

# print(zip_df)
# 1. how many rows/columns are there?
print("# Part 1.1")
print("Rows:", zip_df.shape[0])
print("Columns:", zip_df.shape[1])
print()

# remove the first column and convert the dataframe to a numpy array
zip_mat = zip_df.iloc[:, 1:].to_numpy()

# 2. What is the number of rows/observations/example digits?
# What is the number of columns/features/pixels per example?
print("# Part 1.2")
print("Rows / observations / example digits:", zip_mat.shape[0])
print("Columns / features / pixels per example:", zip_mat.shape[1])
print()

# get one number out of the dataset that's n_pixels by n_pixels
n_pixels = 16
image_index = 0
index_vec = np.arange(n_pixels)
# one number image as a dataframe
one_image_df = pd.DataFrame({
    "col": np.repeat(index_vec, n_pixels),
    "intensity": zip_mat[image_index, :]
})
# print(one_image_df)

# I couldn't get plotnine installed on my macbook :/
# gg = p9.ggplot() + \
#      p9.geom_raster(
#          p9.aes(
#              x="col",
#              y="col",
#              fill="intensity",
#          ),
#          data=one_image_df)
# gg.save("homework1.png")
