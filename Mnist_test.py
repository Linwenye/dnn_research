import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

np.random.seed(1)
tf.set_random_seed(1)

mnist = tf.keras.datasets.mnist
MY_DTYPE = 'float64'

K.set_floatx(MY_DTYPE)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)
print(type(y_train[0]))

x_train = x_train.reshape(x_train.shape[0], -1).astype(MY_DTYPE)
x_test = x_test.reshape(x_test.shape[0], -1).astype(MY_DTYPE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation=tf.nn.relu, dtype=MY_DTYPE),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax, dtype=MY_DTYPE)
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=1000)
print(model.evaluate(x_test, y_test))
