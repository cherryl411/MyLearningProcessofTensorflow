# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: FastTextTest.py
@date: 2019-01-16

"""
import fasttext

train_path = '/Users/liunianci/zhubajie/Data/Finance/ft_train.txt'
valid_path = '/Users/liunianci/zhubajie/Data/Finance/ft_valid.txt'
model_path = '/Users/liunianci/zhubajie/Data/Finance/ft_model.txt'
# train
classifier = fasttext.supervised(train_path, model_path, lr=0.5, word_ngrams=2, bucket=200000)


# valid
result = classifier.test(valid_path)
print('Precision:{}'.format(result.precision))
print('Recall:{}'.format(result.recall))