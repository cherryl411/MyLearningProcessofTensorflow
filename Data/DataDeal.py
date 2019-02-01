# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: DataDeal.py
@date: 2019-01-11

"""

import os
from urllib.request import urlopen
import io
from zhon.hanzi import punctuation
import string

# Judge if the data files exist, if exist return two classifer's data, else return Null
def judge_data(floderName, saveName):
    pos_path = floderName + saveName + '/rt-polarity.pos'
    print(pos_path)
    neg_path = floderName + saveName + '/rt-polarity.neg'
    pos_data = []
    neg_data = []
    # Check if the files are already exist
    if os.path.exists(pos_path) and os.path.exists(neg_path):

        with open(pos_path, 'r') as d:
            # every row is a case
            for row in d:
                pos_data.append(row)
        d.close()

        with open(neg_path, 'r') as d:
            # every line is a case
            for row in d:
                neg_data.append(row)
        d.close()

    return pos_data, neg_data

# Data Normalized
def text_normalized(texts, stopwords):
    # Lower case
    texts = [x.lower() for x in texts]

    # Remove all punctuation
    texts = [''.join(c for c in x if c not in string.punctuation and c not in punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in [0,1,2,3,4,5,6,7,8,9]) for x in texts]

    # Remove stopwords
    texts = [' '.join([c for c in x.split() if c not in stopwords]) for x in texts]

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]

    # Make sure the content is effective
    texts = [x for x in texts if len(x) > 3]


    return texts