import numpy as np
import h5py
import matplotlib.pyplot as plt
from testcase.testCases_v3 import *
from utils.activation_function import *
import decimal
from numpy_decimal import *

# pre setting
plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# parameters = initialize_parameters(3, 2, 1)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def initialize_parameters_deep(layer_dims, dtype):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    if dtype == "decimal":
        for l in range(1, L):
            parameters['W' + str(l)] = to_decimal(np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
                layer_dims[l - 1]))  # *0.01

            parameters['b' + str(l)] = to_decimal(np.zeros((layer_dims[l], 1)))
            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    elif dtype == "mpmath":
        for l in range(1, L):
            parameters['W' + str(l)] = to_mp(np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
                layer_dims[l - 1]))  # *0.01

            parameters['b' + str(l)] = to_mp(np.zeros((layer_dims[l], 1)))
            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    else:
        for l in range(1, L):
            parameters['W' + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
                layer_dims[l - 1])).astype(dtype)  # *0.01

            parameters['b' + str(l)] = (np.zeros((layer_dims[l], 1))).astype(dtype)
            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


# parameters = initialize_parameters_deep([5, 4, 3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


# A, W, b = linear_forward_test_case()
#
# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    else:  # no activation
        A, linear_cache = linear_forward(A_prev, W, b)
        activation_cache = A

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# A_prev, W, b = linear_activation_forward_test_case()
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
# print("With sigmoid: A = " + str(A))
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
# print("With ReLU: A = " + str(A))


def L_model_forward(X, parameters, activation_list):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    activation_list -- layer activation
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation_list[l - 1])
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation_list[L - 1])
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


# X, parameters = L_model_forward_test_case_2hidden()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))


def compute_cost(AL, Y, dtype, cost_type="cross-entropy"):
    """
    Implement the cost function defined by equation).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    # log use e as base
    if (cost_type == "cross-entropy"):
        if dtype == "decimal":
            cost = np.sum(np.multiply(Y, decimal_log(AL)) + np.multiply(1 - Y, decimal_log(1 - AL)))
            cost = -cost / m
        else:
            cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    elif (cost_type == "MAE"):
        cost = np.sum(np.abs(AL - Y)) / m
    elif (cost_type == "MSE"):
        cost = np.sum(np.square(AL - Y)) / (2 * m)
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


# Y, AL = compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


# Set up some test inputs
# dZ, linear_cache = linear_backward_test_case()
#
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print("dA_prev = " + str(dA_prev))
# print("dW = " + str(dW))
# print("db = " + str(db))


def linear_activation_backward(dA, cache, activation, dtype):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, cache[1], dtype)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache[1])
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
    else:  # linear
        dZ = np.array(dA, copy=True)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


# AL, linear_activation_cache = linear_activation_backward_test_case()
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")
# print("sigmoid:")
# print("dA_prev = " + str(dA_prev))
# print("dW = " + str(dW))
# print("db = " + str(db) + "\n")
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="relu")
# print("relu:")
# print("dA_prev = " + str(dA_prev))
# print("dW = " + str(dW))
# print("db = " + str(db))


def L_model_backward(AL, Y, caches, activation_list, cost_type="cross-entropy", dtype="float"):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    activation_list -- layer activation function "sigmoid" "tanh" "linear"
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    if cost_type == "cross-entropy":
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    elif cost_type == "MSE":
        dAL = AL - Y
    current_cache = caches[- 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation_list[
                                                                                                          L - 1], dtype)

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation_list[l], dtype)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


#
# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print_grads(grads)


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def predict(X, y, parameters, activation_list, not_convert=False):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters, activation_list)

    if not not_convert:
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        print("Accuracy: " + str(np.sum((p == y) / m)))
        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        return p

    else:
        return probas
