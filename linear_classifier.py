# Using linear regression for classification

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.normal(5, 1, 10)
x2 = np.random.normal(2, 1, 10)
xs = np.append(x1, x2)
ys = np.asarray([0.] * len(x1) + [1.] * len(x2))

X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable([0., 0.], name="parameters")

y_model = tf.add(tf.multiply(w[1], tf.pow(X, 1)), tf.multiply(w[0], tf.pow(X, 0)))
cost = tf.reduce_sum(tf.square(Y - y_model))

learning_rate = 0.001
training_epochs = 1000
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        current_cost, _ = sess.run([cost, train_op], feed_dict={X: xs, Y: ys})
        if epoch % 100 == 0:
            print(epoch, current_cost)

    w_val = sess.run(w)
    print('learned parameters', w_val)

    # Measuring accuracy
    correct_prediction = tf.equal(Y, tf.to_float(tf.greater(y_model, 0.5)))
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    print('accuracy', sess.run(accuracy, feed_dict={X: xs, Y: ys}))


# Plotting
all_xs = np.linspace(0, 10, 100)
plt.scatter(xs, ys)
plt.plot(all_xs, all_xs*w_val[1] + w_val[0])
plt.show()