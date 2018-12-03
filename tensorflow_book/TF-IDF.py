# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: TF-IDF.py
@data: 2018-12-03

"""

from zhon.hanzi import punctuation
import string
import tensorflow as tf
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 读取数据
data_path = '/Users/liunianci/Downloads/smsspamcollection/SMSSpamCollection'
with open(data_path, 'r') as f:
    data_content = []
    for line in f.readlines():
        data_content.append(line.split('\t'))
# print(len(data_content))

# 2. 规则化处理
texts = [x[1] for x in data_content]
label = [x[0] for x in data_content]
texts = [x.lower() for x in texts]
texts = [''.join(c for c in x if c not in string.punctuation and c not in punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
texts = [' '.join(x.split()) for x in texts]
label = [1. if x == 'spam' else 0. for x in label]

# 3. 分词、创建tf-idf矩阵
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=1000)
sparse_mat = tfidf.fit_transform(texts)

# 4. 分割训练集和测试集
train_indcts = np.random.choice(sparse_mat.shape[0], round(sparse_mat.shape[0] * 0.8), replace=False)
test_indics = np.array(list(set(range(sparse_mat.shape[0])) - set(train_indcts)))
train_data = sparse_mat[train_indcts]
test_data = sparse_mat[test_indics]
train_label = np.array([x for xi, x in enumerate(label) if xi in train_indcts])
test_label = np.array([x for xi, x in enumerate(label) if xi in test_indics])

# 5. 声明变量、占位符
max_size = 1000
W = tf.Variable(np.random.normal(size=(max_size, 1)), dtype=tf.float32)
b = tf.Variable(np.random.normal(size=(1, 1)), dtype=tf.float32)
x_data = tf.placeholder(shape=[None, max_size], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 6. 建模
y_pre = tf.add(tf.matmul(x_data, W), b)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_data, logits=y_pre)
prediction = tf.round(tf.nn.sigmoid(y_pre))
pre_correct = tf.cast(tf.equal(prediction, y_data), tf.float32)
acc = tf.reduce_mean(pre_correct)
op = tf.train.GradientDescentOptimizer(0.025)
train_step = op.minimize(loss)

# 7. 初始化
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 8. 迭代训练
train_loss = []
train_acc = []
iter_num = 1000
batch_size = 200
for i in range(iter_num):
    rand_index = np.random.choice(train_data.shape[0], size=batch_size)
    rand_x = train_data[rand_index].todense()
    rand_y = np.transpose([train_label[rand_index]])
    sess.run(train_step, feed_dict={y_data:rand_y, x_data:rand_x})
    if (i+1)%100 == 0:
        temp_loss = sess.run(loss, feed_dict={y_data:rand_y, x_data:rand_x})
        # print(temp_loss)
        train_loss.append(temp_loss)
        temp_acc = sess.run(acc, feed_dict={y_data:rand_y, x_data:rand_x})
        print(temp_acc)
        train_acc.append(temp_acc)
