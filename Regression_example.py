from NN_model import *
import numpy as np

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  # randomize the data

Y = 2 * X + 2

# plot data
plt.scatter(X, Y)
plt.show()

# train test split
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]
X_train = X_train.reshape(1, 160)
Y_train = Y_train.reshape(1, 160)

X_test = X_test.reshape(1, 40)
Y_test = Y_test.reshape(1, 40)

print(str(X_train.shape))
layers_dims = [1, 1, 1]  # 2-layer model

parameters = L_layer_model(X_train, Y_train, layers_dims, activation_list=["relu","linear"],num_iterations=2500, print_cost=True,
                           cost_type="MSE")
pred_train = predict(X_train, Y_train, parameters)
pred_test = predict(X_test, Y_test, parameters)
