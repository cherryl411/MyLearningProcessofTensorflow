# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: NeuralNetwork.py
@date: 2019-06-15

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 1.load the iris's data
iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])

sess = tf.Session()
# 2.set the seed for result recurrence
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# 3.set train data and test data and normalize those
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
train_x = x_vals[train_indices]
test_x = x_vals[test_indices]
train_y = y_vals[train_indices]
test_y = y_vals[test_indices]

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

train_x = np.nan_to_num(normalize_cols(train_x))
test_x = np.nan_to_num(normalize_cols(test_x))

# 4.declare the variables and placeholder
barch_size = 50
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_label = tf.placeholder(shape=[None, 1], dtype=tf.float32)

hidden_layer_nodes = 10
w1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))

# 5.calculate the outputs of hidden_layer and output_layer
h1_out = tf.nn.relu(tf.add(tf.matmul(x_data, w1), b1))
y_out = tf.nn.relu(tf.add(tf.matmul(h1_out, w2), b2))

# 6.declare the loss function and optional algorithm
loss = tf.reduce_mean(tf.square(y_label-y_out))
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# 7.initial
init = tf.global_variables_initializer()
sess.run(init)

# 8.iteration
loss_vec = []
test_loss_vec = []

for i in range(500):
    rand_index = np.random.choice(len(train_x), size=barch_size)
    rand_x = train_x[rand_index]
    rand_y = np.transpose([train_y[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_label: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_label: rand_y})
    loss_vec.append(np.sqrt(temp_loss))
    test_loss = sess.run(loss, feed_dict={x_data: test_x, y_label: np.transpose([test_y])})
    test_loss_vec.append(np.sqrt(test_loss))
    if (i+1)%50 == 0:
        print('Generation' + str(i+1) + '.Loss = ' + str(temp_loss))

# 9.plot
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss_vec, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
