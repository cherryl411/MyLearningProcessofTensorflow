# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: BodyWeightPredict.py
@date: 2019-07-03

"""

import tensorflow as tf
import requests
import numpy as np
import os
from matplotlib import pyplot as plt

sess = tf.Session()

# data download(if file not exist)
birth_weight_file = 'birth_weight.csv'
birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master' \
                '/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'

if not os._exists(birth_weight_file):
    file = requests.get(birthdata_url)
    data = file.text.split('\r\n')
    header = data[0].split('\t')
    data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in data[1:] if len(y) >= 1]
    # print(x[8] for x in data)
    y_vals = np.array([x[8] for x in data])
    cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
    x_vals = np.array([[x[ix] for ix, feature in enumerate(header) if feature in cols_of_interest] for x in data])

# set seed and batch_size
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)
batch_size = 100

# set train data and test data to 8 : 2
train_index = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_index = np.array(list(set(range(len(x_vals))) - set(train_index)))
x_train = x_vals[train_index]
x_test = x_vals[test_index]
y_train = y_vals[train_index]
y_test = y_vals[test_index]


# normalized
def my_normal_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)


x_train = np.nan_to_num(my_normal_cols(x_train))
x_test = np.nan_to_num(my_normal_cols(x_test))


# declare initial function for w and b
def my_init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))

    return weight


# initial the placeholder
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# declare fully connected layer
def fully_connected(input_layer, weights, bias):
    layer = tf.add(tf.matmul(input_layer, weights), bias)

    return layer


# create first layer(25 hidden nodes)
weight_1 = my_init_weight(shape=[7, 25], st_dev=10)
bias_1 = my_init_weight(shape=[25], st_dev=10)
layer_1 = fully_connected(x_data, weight_1, bias_1)

# create second layer(10 hidden nodes)
weight_2 = my_init_weight(shape=[25, 10], st_dev=10)
bias_2 = my_init_weight(shape=[10], st_dev=10)
layer_2 = fully_connected(layer_1, weight_2, bias_2)

# create third layer(3 hidden nodes)
weight_3 = my_init_weight(shape=[10, 3], st_dev=10)
bias_3 = my_init_weight(shape=[3], st_dev=10)
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# create output layer(1 value)
weight_4 = my_init_weight(shape=[3, 1], st_dev=10)
bias_4 = my_init_weight(shape=[1], st_dev=10)
output = fully_connected(layer_3, weight_4, bias_4)

# declare the loss function
loss = tf.reduce_mean(tf.abs(y_data - output))
my_opt = tf.train.AdamOptimizer(0.05)
train_step = my_opt.minimize(loss)

# declare the initial variables
init = tf.global_variables_initializer()
sess.run(init)

# initialize the loss vector
train_loss = []
test_loss = []
for i in range(200):
    # choose random indices for batch selection
    rand_index = np.random.choice(len(x_train), size=batch_size, replace=False)
    rand_x = x_train[rand_index]
    # print(y_train[rand_index])
    rand_y = np.transpose([y_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y})
    # get and store the train loss
    train_temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_data: rand_y})
    train_loss.append(train_temp_loss)
    # get and store the test loss
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_test, y_data: np.transpose([y_test])})
    test_loss.append(test_temp_loss)
    if (i+1) % 25 == 0:
        print('Generation: ' + str(i+1) + '.Loss = ' + str(train_temp_loss))

plt.plot(train_loss, 'k-', label='Train Loss')
plt.plot(test_loss, 'r.', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# calculate the accuracy of train data and test data
actuals = np.array([x[0] for x in data])
test_actuals = actuals[test_index]
train_actuals = actuals[train_index]
test_predicts = [x[0] for x in sess.run(output, feed_dict={x_data: x_test})]
trian_predicts = [x[0] for x in sess.run(output, feed_dict={x_data: x_train})]
test_pre = np.array([1.0 if x < 2500 else 0.0 for x in test_predicts])
train_pre = np.array([1.0 if x < 2500 else 0.0 for x in trian_predicts])
test_acc = np.mean([x == y for x, y in zip(test_pre, test_actuals)])
train_acc = np.mean([x == y for x, y in zip(train_pre, train_actuals)])
print('On predicting the category of low birthweight from regression output (<2500g):')
print('Test Accuracy: {}'.format(test_acc))
print('Train Accuracy: {}'.format(train_acc))