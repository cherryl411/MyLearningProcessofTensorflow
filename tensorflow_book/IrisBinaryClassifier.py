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
iris_features = np.array([[x[0], x[3]] for x in iris_data.data])
# iris_features = iris_data.data
print(iris_features.shape)
# print(iris_labels.shape)

# 2. 参数声明
batch_size = 150
feature = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_real = tf.placeholder(shape=[None, 1], dtype=tf.float32)
weight = tf.Variable(tf.random_normal(shape=[2, 1]), dtype=tf.float32)
bias = tf.Variable(tf.random_normal(shape=[1, 1]), dtype=tf.float32)

# 3. 损失函数、参数优化
y_pre = tf.add(tf.matmul(feature, weight), bias)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_real, logits=y_pre))
op = tf.train.GradientDescentOptimizer(0.3)
train = op.minimize(loss)

# 4. 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 5. 模型训练
iter_num = 100
loss_ = []
for i in range(iter_num):
    rand_index = np.random.choice(len(iris_features), size=batch_size)
    feature_data = iris_features[rand_index]
    # print(feature_data)
    y_data = iris_labels[rand_index]
    sess.run(train, feed_dict={y_real:y_data, feature:feature_data})
    w = sess.run(tf.transpose(weight))
    b = sess.run(bias)
    loss_.append(sess.run(loss, feed_dict={y_real:y_data, feature:feature_data}))
    y_prediction = tf.round(tf.nn.sigmoid(w * feature + b))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_prediction, y_real), tf.float32))
    if i % 10 == 0:
        print('w:' + str(w) + '\t' + 'b:' + str(b))
        print('Accuracy = ' + str(sess.run(accuracy, feed_dict={y_real:y_data, feature:feature_data})))

# 6. 绘图
plt.plot(loss_, 'g-')
plt.title('Loss')
plt.show()