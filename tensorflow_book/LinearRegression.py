# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: LinearRegression.py
@date: 2018-11-30

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#  1. 构造y = 3x + 1 + e 函数的随机数据集, e为随机扰动
x_data1 = np.random.rand(1000)
# x_data2 = np.random.rand(1000)
# print(x_data.shape)
w_true = [3., 5.]
b_true = 1.
y_data = w_true[0] * x_data1 + b_true
y_true = y_data + np.random.normal([1])
# plt.scatter(x_data1, y_true)
# plt.show()

# 2. 随机初始化
w_pre = tf.Variable(tf.random.uniform(shape=[1]))
b_pre = tf.Variable(tf.zeros([1]))

# 3. 预测y
y_pre = w_pre * x_data1 + b_pre

# 4.损失函数
loss = tf.reduce_mean(tf.square(y_pre - y_true))

# 5.寻优
optimizer = tf.train.GradientDescentOptimizer(0.6)
train = optimizer.minimize(loss)

# 6.初始化
init = tf.global_variables_initializer()

# 7.会话
sess = tf.Session()
sess.run(init)

# 8.运行+绘图
iter_num = 200
loss_ = []
for step in range(iter_num):
    sess.run(train)
    w = sess.run(w_pre)
    b = sess.run(b_pre)

    temp_loss = sess.run(loss)
    loss_.append(temp_loss)
    if step % 20 == 0:
        print(step, w, b)
        print()
fits = []
for i in x_data1:
    fits.append(w * i + b)

plt.plot(x_data1, y_true, 'o', label = 'Origin')
plt.plot(x_data1, fits, 'r-', label = 'Fit')
plt.title('Curve-Fitting')
plt.show()

plt.plot(loss_, 'g-')
plt.title('Loss Function')
plt.show()

