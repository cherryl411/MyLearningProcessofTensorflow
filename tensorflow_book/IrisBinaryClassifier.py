# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: IrisBinaryClassifier.py
@data: 2018-11-30

"""

import tensorflow as tf
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# 1. 导入数据，将label置为0和1
iris_data = datasets.load_iris()
iris_labels = np.array([1. if x == 0 else 0. for x in iris_data.target]).reshape(150, 1)
iris_features = iris_data.data
# print(iris_features)
# print(iris_labels.shape)

# 2. 参数声明
batch_size = 20
feature = tf.placeholder(shape=[None, 4], dtype=tf.float32)
# feature_2 = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# feature_3 = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# feature_4 = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_pre = tf.placeholder(shape=[None, 1], dtype=tf.float32)
weight = tf.Variable(tf.random_normal(shape=[4, 1]), dtype=tf.float32)
bias = tf.Variable(tf.random_normal(shape=[1, 1]), dtype=tf.float32)

# 3. 损失函数、参数优化
y_res = tf.add(tf.matmul(feature, weight), bias)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_res, logits=y_pre)
op = tf.train.GradientDescentOptimizer(0.05)
train = op.minimize(loss)

# 4. 初始化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 5. 模型训练
iter_num = 1000
for i in range(iter_num):
    rand_index = np.random.choice(len(iris_features), size=batch_size)
    feature_data = iris_features[rand_index]
    # print(feature_data)
    y_data = iris_labels[rand_index]
    sess.run(train, feed_dict={y_pre:y_data, feature:feature_data})
    if i % 100 == 0:
        print('w:' + str(sess.run(tf.transpose(weight))) + '\t' + 'b:' + str(sess.run(bias)))