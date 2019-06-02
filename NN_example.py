import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from utils.dnn_app_utils import load_data, print_mislabeled_images
from NN_model import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Example of a picture
# for index in range(len(train_x_orig)):
#     plt.imshow(train_x_orig[index])
#     plt.show()
#     print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# print("Number of training examples: " + str(m_train))
# print("Number of testing examples: " + str(m_test))
# print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print("train_x_orig shape: " + str(train_x_orig.shape))
# print("train_y shape: " + str(train_y.shape))
# print("test_x_orig shape: " + str(test_x_orig.shape))
# print("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                       -1).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

dtype = "float16"
# precision = 33  # decimal float128
# precision = 70 # decimal float256
precision = 24  # float32
# precision = 11 # float16


if dtype == "decimal":
    set_decimal_precision(precision)
    train_x, test_x, train_y, test_y = to_decimal(train_x), to_decimal(test_x), to_decimal(train_y), to_decimal(test_y)
elif dtype == "mpmath":
    set_mp_precision(precision)
    train_x, test_x, train_y, test_y = to_mp(train_x), to_mp(test_x), to_mp(train_y), to_mp(test_y)
elif dtype == "float32" or dtype == "float16":
    train_x, test_x, train_y, test_y = train_x.astype(dtype), test_x.astype(dtype), train_y.astype(
        dtype), test_y.astype(dtype)

### FOUR LAYER MODEL ###
layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model
activation_list = ["relu", "relu", "relu", "sigmoid"]
print_ctrl = {"print_cost": True, "print_parameter": True, "iterations": 20}
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2400, print_ctrl=print_ctrl,
                           activation_list=activation_list, dtype=dtype, precision=precision)
pred_train = predict(train_x, train_y, parameters, activation_list=activation_list)
pred_test = predict(test_x, test_y, parameters, activation_list=activation_list)
