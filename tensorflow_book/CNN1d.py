# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: CNN1d.py
@date: 2019-06-26

"""

import tensorflow as tf
import numpy as np
sess = tf.Session()

# data was initialized
data_size = 25
data_1d = np.random.normal(size=data_size)
print(data_1d)
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])

# declarte the convolution function
def conv_layer_1d(input_1d, my_filter):
    # make 1d input to 4d
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    # print('input_4d: ')
    # print(input_4d)

    # perform the convolution
    conv_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,1,1,1], padding='VALID')
    # print('conv_output')
    # print(conv_output)

    # # drop the extra dimensions
    # conv_output_1d = tf.squeeze(conv_output)
    return (conv_output)

my_filter = tf.Variable(tf.random.normal(shape=[1,5,1,1]))
my_conv_output = conv_layer_1d(x_input_1d, my_filter)

# activation function
def activation_(input_1d):
    return (tf.nn.relu(input_1d))

my_act_func = activation_(my_conv_output)

# declarate the pool function
def max_pool(input_1d, width):
    # # make the 1d input into 4d
    # input_2d = tf.expand_dims(input_1d, 0)
    # input_3d = tf.expand_dims(input_2d, 0)
    # input_4d = tf.expand_dims(input_3d, 3)

    # perform the pooling
    pool_output = tf.nn.max_pool(input_1d, ksize=[1, 1, width, 1], strides=[1, 1, 1, 1], padding='VALID')

    # drop the extra dimensions
    pool_output_1d = tf.squeeze(pool_output)
    return (pool_output_1d)

my_pool_output = max_pool(my_act_func, 5)

# fully connected layer
def fully_connect(input_layer, num_output):
    # create weights, the dimension of weights will be (length of input) by (num_outputs)
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_output]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_output])

    # make input to 2d
    input_layer_2d = tf.expand_dims(input_layer, 0)

    # perform the fully connect
    fully_output = tf.add(tf.matmul(input_layer_2d, weight), bias)

    # drop the extra dimensions
    fully_output_1d = tf.squeeze(fully_output)
    return (fully_output_1d)

my_fully_output = fully_connect(my_pool_output, 5)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_1d: data_1d}

print('>>>> 1D Data <<<<')

# Convolution Output
print('Input = array of length {}'.format(x_input_1d.shape.as_list()[0]))
print('Convolution w/ filter, length = {}, stride size = {}, results in an array of length {}:'.format(5, 1, my_conv_output.shape.as_list()[0]))
print(sess.run(my_conv_output, feed_dict=feed_dict))

# Activation Output
print('\nInput = above array of length {}'.format(my_conv_output.shape.as_list()[0]))
print('ReLU element wise returns an array of length {}:'.format(my_act_func.shape.as_list()[0]))
print(sess.run(my_act_func, feed_dict=feed_dict))

# Max Pool Output
print('\nInput = above array of length {}'.format(my_act_func.shape.as_list()[0]))
print('MaxPool, window length = {}, stride size = {}, results in the array of length {}'.format(5, 1, my_pool_output.shape.as_list()[0]))
print(sess.run(my_pool_output, feed_dict=feed_dict))

# Fully Connected Output
print('\nInput = above array of length {}'.format(my_pool_output.shape.as_list()[0]))
print('Fully connected layer on all 4 rows with {} outputs:'.format(my_fully_output.shape.as_list()[0]))
print(sess.run(my_fully_output, feed_dict=feed_dict))
