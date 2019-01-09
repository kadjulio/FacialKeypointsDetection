from readCsvFiles import load
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot

TRAINING_EPOCHS = 100

# # def init_neural_network():
# x1 = tf.placeholder(tf.float32, [None, 9216])
# W1 = tf.Variable(tf.zeros([9216, 100]))
# W2 = tf.Variable(tf.zeros([100, 30]))
# b1 = tf.Variable(tf.zeros([100]))
# b2 = tf.Variable(tf.zeros([30]))
# y = tf.nn.softmax(tf.matmul(x1, W1) + b1)  # the equation
# y1 = tf.nn.softmax(tf.matmul(y, W2) + b2)
# print(y)
# print(y1)
# y_ = tf.placeholder(tf.float32, [None, 30])
# cross_entropy = -tf.reduce_sum(y_*tf.log(y1))
# train_step = tf.train.GradientDescentOptimizer(
#     0.0001).minimize(cross_entropy)
# print(train_step)
# return train_step


X_train, y_train = load()
print("X_train.shape == {}; X_train.min == {:.3f}; X_train.max == {:.3f}".format(
    X_train.shape, X_train.min(), X_train.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))
# train_step = init_neural_network()


x1 = tf.placeholder(tf.float32, [None, 9216])
W1 = tf.Variable(tf.zeros([9216, 100]))
W2 = tf.Variable(tf.zeros([100, 30]))
b1 = tf.Variable(tf.zeros([100]))
b2 = tf.Variable(tf.zeros([30]))
y = tf.nn.relu(tf.matmul(x1, W1) + b1)  # the equation
y1 = tf.nn.tanh(tf.matmul(y, W2) + b2)
print(y)
print(y1)
y_ = tf.placeholder(tf.float32, [None, 30])
mse = tf.reduce_mean(tf.square(tf.subtract(y_, y1)))
train_step = tf.train.AdamOptimizer(0.0001).minimize(mse)
print(train_step)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
avg_cost = []

for j in range(TRAINING_EPOCHS):
    # batch_xs, batch_ys = X, y
    total_batch = int(len(X_train) / 100)
    for i in range(20):
        nb = np.array([random.randint(0, len(X_train) - 1) for _ in range(100)])
        # print(nb.shape, len(X_train))
        batch_xs = X_train[nb]
        batch_ys = y_train[nb]
    # batch_x, batch_y = batch_xs[i], batch_ys[i]
    # print(type(train_step), type(batch_x), type(batch_y))
    # print(batch_y.shape)
        _, c = sess.run([train_step, mse], feed_dict={x1: batch_xs, y_: batch_ys})
    avg_cost.append(c)
    # print(avg_cost)
    print("Epoch " + str(j) + ":", str(np.mean(avg_cost)))
print("done")
pyplot.plot(avg_cost)
pyplot.show()
# # evaluate the accuracy of the model
# X_test, Y_test = load(True)
# correct_prediction = tf.equal(tf.argmax(y1,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(sess.run(accuracy, feed_dict={x1: X_test, y_: Y_test}))
