from readCsvFiles import load
import matplotlib.pyplot as plt
import tensorflow as tf

def init_neural_network():
    x = tf.placeholder(tf.float32, [None, 9216])

    W = tf.Variable(tf.zeros([9216, 30]))
    b = tf.Variable(tf.zeros([30]))

    y = tf.nn.softmax(tf.matmul(x, W) + b) # the equation
    print(y)

X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))
init_neural_network()