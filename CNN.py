import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)
# 3filters
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, padding="same", activation="relu", use_bias=True,
                           input_shape=(1, 1)),
    tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, padding="same", activation="linear", use_bias=True),
    tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding="same", use_bias=True),
])

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae', 'mse'])

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  # randomize the data

Y = 2 * X + 2

# plot data
plt.scatter(X, Y)
plt.show()

# train test split
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]
X_train = X_train.reshape(160, 1, 1)
Y_train = Y_train.reshape(160, 1, 1)

X_test = X_test.reshape(40, 1, 1)
Y_test = Y_test.reshape(40, 1, 1)

print('Training -----------')
for step in range(10001):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)

Y_pred = model.predict(X_test)
print(model.summary())

X_test = X_test.flatten()
Y_test = Y_test.flatten()
Y_pred = Y_pred.flatten()

plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
