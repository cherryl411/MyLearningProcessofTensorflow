# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: BagofWords.py
@date: 2018-12-03

"""

from zhon.hanzi import punctuation
import string
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 1. 读取数据
data_path = '/Users/liunianci/Downloads/smsspamcollection/SMSSpamCollection'
with open(data_path, 'r') as f:
    data_content = []
    for line in f.readlines():
        # print(line.split('\t'))
        data_content.append(line.split('\t'))
print(len(data_content))

# 2. 规则化处理
texts = [x[1] for x in data_content]
label = [x[0] for x in data_content]
texts = [x.lower() for x in texts]
texts = [
    ''.join(
        c for c in x if c not in string.punctuation and c not in punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
texts = [' '.join(x.split()) for x in texts]
label = [1. if x == 'spam' else 0. for x in label]

# texts_size = [len(x.split()) for x in texts]
# texts_size = [x for x in texts_size if x < 50]
# plt.hist(texts_size, bins=25)
# plt.show()
sentence_size = 25
min_fre = 3

# 3. 创建词汇表、声明One-Hot
vocabulary_pro = tf.contrib.learn.preprocessing.VocabularyProcessor(
    sentence_size, min_fre)
vocabulary_pro.fit_transform(texts)
embedding_size = len(vocabulary_pro.vocabulary_)
# print(embedding_size)
embedding_mat = tf.diag(tf.ones(shape=embedding_size))

# 4. 分割训练集和测试集
train_indcts = np.random.choice(
    len(texts), round(
        len(texts) * 0.8), replace=False)
test_indics = np.array(list(set(range(len(texts))) - set(train_indcts)))
train_data = [x for xi, x in enumerate(texts) if xi in train_indcts]
test_data = [x for xi, x in enumerate(texts) if xi in test_indics]
train_label = [x for xi, x in enumerate(label) if xi in train_indcts]
test_label = [x for xi, x in enumerate(label) if xi in test_indics]

# 5. 声明变量、占位符
W = tf.Variable(np.random.normal(size=(1, embedding_size)), dtype=tf.float32)
b = tf.Variable(np.random.normal(size=(1, 1)), dtype=tf.float32)
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_data = tf.placeholder(shape=[1, 1], dtype=tf.float32)

# 6. 建模
embedding_ = tf.nn.embedding_lookup(embedding_mat, x_data)
x_real = tf.reduce_sum(embedding_, 0)
x_real_2 = tf.expand_dims(x_real, 1)

y_pre = tf.add(tf.matmul(W, x_real_2), b)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_data, logits=y_pre)
predict_data = tf.sigmoid(y_pre)
op = tf.train.GradientDescentOptimizer(0.001)
train_step = op.minimize(loss)

# 7. 初始化
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 8. 迭代训练
train_acc_all = []
train_acc_avg = []
for xi, x in enumerate(vocabulary_pro.fit_transform(train_data)):
    y_label = [[train_label[xi]]]
    # print(x.shape)
    # print(x)
    # print(y_label)
    sess.run(train_step, feed_dict={x_data: x, y_data: y_label})
    loss_temp = sess.run(loss, feed_dict={x_data: x, y_data: y_label})
    y_pre = np.round(sess.run(predict_data, feed_dict={x_data: x}))
    acc_temp = train_label[xi] == y_pre
    train_acc_all.append(acc_temp)
    if len(train_acc_all) > 100:
        train_acc_avg.append(np.mean(train_acc_all[-100:]))
        # print(train_acc_avg)
    if (xi + 1) % 100 == 0:
        print('Training Num #' + str(xi + 1) + ': Loss = ' + str(loss_temp))
print('Train Accuracy is : {}'.format(np.mean(train_acc_all)))

# 9. 测试
test_acc_all = []
test_acc_avg = []
for xi, x in enumerate(vocabulary_pro.fit_transform(test_data)):
    y_label = [[test_label[xi]]]
    y_pre = np.round(sess.run(predict_data, feed_dict={x_data: x}))
    acc_temp = test_label[xi] == y_pre
    test_acc_all.append(acc_temp)
    if len(test_acc_all) > 100:
        test_acc_avg.append(np.mean(test_acc_all[-100:]))
        # print(train_acc_avg)
print('Test Accuracy is : {}'.format(np.mean(test_acc_all)))
