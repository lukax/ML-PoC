# Using one-dimensional logistic regression for classification

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


x1 = np.random.normal(-4, 2, 10)
x2 = np.random.normal(4, 2, 10)
xs = np.append(x1, x2)
ys = np.asarray([0.] * len(x1) + [1.] * len(x2))

X = tf.placeholder(tf.float32, shape=(None,), name="x")
Y = tf.placeholder(tf.float32, shape=(None,), name="y")
w = tf.Variable([0., 0.], name="parameters", trainable=True)

y_model = tf.sigmoid(w[1] * X + w[0])
cost = tf.reduce_mean(-Y * tf.log(y_model) - (1 - Y) * tf.log(1 - y_model))

learning_rate = 0.01
training_epochs = 1000
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_err = 0

    for epoch in range(training_epochs):
        current_cost, _ = sess.run([cost, train_op], feed_dict={X: xs, Y: ys})
        if epoch % 100 == 0:
            print(epoch, current_cost)
        if abs(prev_err - current_cost) < 0.0001:
            break
        prev_err = current_cost
    w_val = sess.run(w, {X: xs, Y: ys})

    print('learned parameters', w_val)


# Plotting
all_xs = np.linspace(-10, 10, 100)
plt.scatter(xs, ys)
plt.plot(all_xs, sigmoid((all_xs * w_val[1] + w_val[0])))
plt.show()

