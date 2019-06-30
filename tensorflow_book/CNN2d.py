# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: CNN2d.py
@date: 2019-06-27

"""

import tensorflow as tf
import numpy as np

sess = tf.Session()

# Initialization
data_size = [10, 10]
data = np.random.normal(size=data_size)
x_input = tf.placeholder(dtype=tf.float32, shape=data_size)

# Declare the convolution function
def conv_func(input, filter):
    input_3d = tf.expand_dims(input, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    # Perform the convolution
    conv_output = tf.nn.conv2d(input_4d, filter=filter, strides=[1, 2, 2, 1], padding='VALID')

    return conv_output
my_filter = tf.Variable(tf.random_normal(shape=[2, 2, 1, 1]))
my_conv = conv_func(x_input, my_filter)

# Activate
def act_func(input):
    act_output = tf.nn.relu(input)

    return act_output
my_act = act_func(my_conv)

# Do pool
def pool_func(input, height, width):
    # Perform the pooling
    pool_output = tf.nn.max_pool(input, ksize=[1, height, width, 1], strides=[1, 1, 1, 1], padding='VALID')

    # Drop the extra dimensions
    pool_out_2d = tf.squeeze(pool_output)

    return pool_out_2d

my_pool = pool_func(my_act, 2, 2)

# Do fully connection
def fully_con(input, out_number):
    # flat the input into 1d
    flat_input = tf.reshape(input, [-1])
    # create weights
    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [out_number]]))
    weigth = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[out_number])
    # change into 2d
    input_2d = tf.expand_dims(flat_input, 0)
    # perform the fully connection
    fully_output = tf.add(tf.matmul(input_2d, weigth), bias)
    # drop the extra dimensions
    fully_con_out = tf.squeeze(fully_output)

    return fully_con_out
my_fully = fully_con(my_pool, 5)

init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input:data}

# print every layer's information
# 1. convolution layer
print('Input [10, 10] array')
print('[1, 2, 2, 1] convolution, stride size = [1, 2, 2, 1], result in this layer is [1, 5, 5, 1] array:')
print(sess.run(my_conv, feed_dict=feed_dict))
# 2. activation layer's information
print('Input [1, 5, 5, 1] array')
print('Result in this layer is [1, 5, 5, 1] array:')
print(sess.run(my_act, feed_dict=feed_dict))
# 3. max_pool layer's information
print('Input [1, 5, 5, 1] array')
print('[1, 2, 2, 1] pooling, stride size is [1, 1, 1, 1], Result in this layer is [4, 4] array:')
print(sess.run(my_pool, feed_dict=feed_dict))
# 4. fully_connection layer's information
print('Input [4, 4] array')
print('Fully connected layer on all four rows with five outputs:')
print(sess.run(my_fully, feed_dict=feed_dict))
