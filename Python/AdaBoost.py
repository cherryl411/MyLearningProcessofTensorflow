# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: AdaBoost.py
@date: 2019-01-31

"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
iris_labels = [float(x) for x in data.target]  # Two_Classifier
iris_features = np.array(data.data)

train_indcts = np.random.choice(len(iris_labels), round(len(iris_labels) * 0.8), replace=False)
test_indics = np.array(list(set(range(len(iris_labels))) - set(train_indcts)))
train_data = [x for xi, x in enumerate(iris_features) if xi in train_indcts]
test_data = [x for xi, x in enumerate(iris_features) if xi in test_indics]
train_label = [x for xi, x in enumerate(iris_labels) if xi in train_indcts]
test_label = [x for xi, x in enumerate(iris_labels) if xi in test_indics]

bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1, algorithm='SAMME.R')
bdt_real.fit(train_data, train_label)
bdt_real_t = bdt_real.predict(test_data)
precision_real = accuracy_score(bdt_real_t, test_label)

print('The precision of SAMME.R is: ', precision_real)

bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600,learning_rate=1.5, algorithm='SAMME')
bdt_discrete.fit(train_data, train_label)
bdt_discrete_t = bdt_discrete.predict(test_data)
precision_dis = accuracy_score(bdt_discrete_t, test_label)

print('The precision of SAMME is: ', precision_dis)