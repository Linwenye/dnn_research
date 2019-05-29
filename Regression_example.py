from NN_model import *
import numpy as np
from numpy_decimal import *

size_total = 500
size_train = size_total // 5 * 4
size_test = size_total - size_train

X = np.linspace(-1, 3, size_total)
np.random.shuffle(X)  # randomize the data

Y = 2 * X + 1.5 + np.random.rand(size_total) * 0.6

# set type and precision

dtype = "mpmath"
precision = 53

if dtype == "decimal":
    set_decimal_precision(precision)
    X = to_decimal(X)
    Y = to_decimal(Y)
elif dtype == "mpmath":
    set_mp_precision(precision)
    X = to_mp(X)
    Y = to_mp(Y)

# train test split
X_train, Y_train = X[:size_train], Y[:size_train]
X_test, Y_test = X[size_train:], Y[size_train:]
X_train = X_train.reshape(1, size_train)
Y_train = Y_train.reshape(1, size_train)

X_test = X_test.reshape(1, size_test)
Y_test = Y_test.reshape(1, size_test)

layers_dims = [1, 3, 3, 1]  # 3-layer model
activation_list = ["relu", "relu", "linear"]
parameters = L_layer_model(X_train, Y_train, layers_dims, activation_list=activation_list, num_iterations=2500,
                           print_cost=True,
                           cost_type="MSE", dtype=dtype, precision=precision)
# pred_train = predict(X_train, Y_train, parameters, activation_list=activation_list)
pred_test = predict(X_test, Y_test, parameters, activation_list=activation_list, to_plot=True)
