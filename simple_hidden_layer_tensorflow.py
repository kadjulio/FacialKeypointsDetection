from readCsvFiles import load
import matplotlib.pyplot as plt
import tensorflow as tf

def init_neural_network():
    x1 = tf.placeholder(tf.float32, [None, 9216])
    W1 = tf.Variable(tf.zeros([9216, 100]))
    W2 = tf.Variable(tf.zeros([100, 30]))
    b1 = tf.Variable(tf.zeros([100]))
    b2 = tf.Variable(tf.zeros([30]))
    y = tf.nn.softmax(tf.matmul(x1, W1) + b1) # the equation
    y1 = tf.nn.softmax(tf.matmul(y, W2) + b2)
    print(y)
    print(y1)
    y_ = tf.placeholder(tf.float32, [None, 30])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y1))
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
    print(train_step)
    return train_step

X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))
train_step = init_neural_network()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# for i in range(1000):
#     batch_xs, batch_ys = X, y
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
# print("done")

# # evaluate the accuracy of the model
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))