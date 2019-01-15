from readCsvFiles import load
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot

TRAINING_EPOCHS = 700

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X_train, y_train = load()
print("X_train.shape == {}; X_train.min == {:.3f}; X_train.max == {:.3f}".format(
    X_train.shape, X_train.min(), X_train.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

x1 = tf.placeholder(tf.float32, [None, 9216])
W1 = tf.Variable(tf.zeros([9216, 100]))
W2 = tf.Variable(tf.zeros([100, 30]))
b1 = tf.Variable(tf.zeros([100]))
b2 = tf.Variable(tf.zeros([30]))
y = tf.nn.relu(tf.matmul(x1, W1) + b1)  # the equation
y1 = tf.nn.tanh(tf.matmul(y, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 30])
mse = tf.reduce_mean(tf.square(tf.subtract(y_, y1)))
train_step = tf.train.AdamOptimizer(0.0001).minimize(mse)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
loss = []
avg_loss = []
for j in range(TRAINING_EPOCHS):
    for i in range(20):
        nb = np.array([random.randint(0, len(X_train) - 1) for _ in range(100)])
        batch_xs = X_train[nb]
        batch_ys = y_train[nb]
        _, c = sess.run([train_step, mse], feed_dict={x1: batch_xs, y_: batch_ys})
    loss.append(c)
    avg_loss.append(np.mean(loss))
    print("Epoch " + str(j) + ": loss=", str(np.mean(loss)))
print("done")
pyplot.plot(avg_loss)
pyplot.show()
X_test, y = load(True)

classification = sess.run(y1, feed_dict={x1: X_test})
fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X_test[i], classification[i], ax)
pyplot.show()
