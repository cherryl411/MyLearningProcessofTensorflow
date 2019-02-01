# -*- coding: utf-8 -*-

"""

@project: python and ML
@author: Cherry_L411@163.com
@file: TextRank.py
@date: 2019-01-08
@Pay attention：It's an implemention of get key words and abstract by textRank. And one of the get key words function'code was copied from Jieba.

"""

from collections import defaultdict
import jieba.posseg
import sys
import numpy as np
import networkx as nx
import math


def pairfilter(wp, pos_filt, stop_words):

    # Filter the words those flag not in allowPOS list.
    return (wp.flag in pos_filt and len(wp.word.strip()) >= 2 and wp.word.lower() not in stop_words)

def getKeyWords(sentence, topK=3, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False, span=5, STOP_WORDS = ("的")):
    """
    Extract keywords from sentence using TextRank algorithm.
    Parameter:
        - topK: return how many top keywords. `None` for all possible words.
        - withWeight: if True, return a list of (word, weight);
                      if False, return a list of words.
        - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v'].
                    if the POS of w is not in this list, it will be filtered.
        - withFlag: if True, return a list of pair(word, weight) like posseg.cut;
                    if False, return a list of words.
        - span: the size of windows.
        - STOP_WORDS: stop words set
    """
    pos_filt = frozenset(allowPOS)
    g = UndirectWeightedGraph()
    cm = defaultdict(int)
    words = tuple(jieba.posseg.dt.cut(sentence))
    for i, wp in enumerate(words):
        if pairfilter(wp, pos_filt, STOP_WORDS):
            for j in range(i + 1, i + span):
                if j >= len(words):
                    break
                if not pairfilter(words[j], pos_filt, STOP_WORDS):
                    continue
                if allowPOS and withFlag:
                    cm[(wp, words[j])] += 1
                else:
                    cm[(wp.word, words[j].word)] += 1

    for terms, w in cm.items():
        g.addEdge(terms[0], terms[1], w)
    nodes_rank = g.rank()

    if withWeight:
        tags = []
        tag = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)
        for i in range(len(tag)):
            if tag[i] in nodes_rank.keys():
                tup1 = (tag[i], nodes_rank[tag[i]])
                tags.append(tup1)

    else:
        tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

    if topK:
        return tags[:topK]
    else:
        return tags

def getMyGraph(text, windows, pos=('ns', 'n', 'v', 'vn'), STOP_WORDS = ("的")):
    word_list = []
    words = tuple(jieba.posseg.dt.cut(text))
    pos_filt = frozenset(pos)
    cm = defaultdict(int)

    for i, w in enumerate(words):
        if pairfilter(w, pos_filt, STOP_WORDS):
            for j in range(i + 1, i + windows):
                if j >= len(words):
                    break
                if not pairfilter(words[j], pos_filt, STOP_WORDS):
                    continue
                cm[(w.word, words[j].word)] += 1
    num = 0
    map_m = {}
    for terms, w in cm.items():
        if terms[0] not in map_m.keys() and terms[1] not in map_m.keys():
            map_m[terms[0]] = num
            num += 1
            map_m[terms[1]] = num
            num += 1
        elif terms[0] not in map_m.keys():
            map_m[terms[0]] = num
            num += 1
        elif terms[1] not in map_m.keys():
            map_m[terms[1]] = num
            num += 1
        else:
            continue
    for key in map_m.keys():
        word_list.append(key)
    mat = np.zeros((num, num))
    for terms, w in cm.items():
        if terms[0] in map_m.keys() and terms[1] in map_m.keys():
            mat[map_m[terms[0]]][map_m[terms[1]]] = w
        else:
            raise MyError('Something Wrong!!!')
    my_graph = nx.from_numpy_matrix(mat)
    return my_graph, word_list

def getKeyWords_nx(text, windows, pos=('ns', 'n', 'v', 'vn'), STOP_WORDS = ("的"), topK=5, alpha=0.85):
    """
    My textRank's implemention. The networkx was used to build the graph.
    :param text: the original text.
    :param windows: the size of windows.
    :param pos: the allowed POS list, eg. ['ns', 'n', 'vn', 'v'].
                if the POS of w is not in this list, it will be filtered.
    :param STOP_WORDS: the stop words set.
                       if the word is in this set, it will be filtered.
    :param topK: the size of key words which will be returned.
                 if topK = 'None', it'll return all possible words.
    :param alpha: the damping factor.
    :return: a list that contain topK key words.
    """
    my_graph, my_words = getMyGraph(text, windows, pos, STOP_WORDS)

    scores = nx.pagerank(my_graph, alpha=alpha)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result_list = []
    for i in range(len(sorted_scores)):
        if sorted_scores[i][0] < len(my_words):
            result = (my_words[sorted_scores[i][0]], sorted_scores[i][1])
            result_list.append(result)

    if topK:
        return result_list[:topK]
    else:
        return result_list

def calculate_sim_sens(sentence1, sentence2):
    all_words = list(set(sentence1 + sentence2))
    vec1 = [float(sentence1.count(word)) for word in all_words]
    vec2 = [float(sentence2.count(word)) for word in all_words]
    vec3 = [vec1[i]*vec2[i] for i in range(len(all_words))]
    vec4 = [1 if num > 0.0 else 0 for num in vec3]

    vec_sum = sum(vec4)

    all_vec = math.log(len(sentence1)) + math.log(len(sentence2))

    if vec_sum < 1 or all_vec < 1e-10:
        return 0
    else:
        return vec_sum / all_vec


def getMyAbstractGraph(text, windows, punc):
    sentence_list = []
    punc_set = frozenset(punc)
    cm = defaultdict(int)
    word_list = []

    j = 0
    for i in range(len(text)):
        if text[i] not in punc_set:
            continue
        else:
            if i > j:
                sentence_list.append(text[j:i+1])
            j = i+1

    for i in range(len(sentence_list)):
        for j in range(i + 1, i + windows):
            if j >= len(sentence_list):
                break
            score_i_j = calculate_sim_sens(sentence_list[i], sentence_list[j])
            cm[(sentence_list[i], sentence_list[j])] += score_i_j

    num = 0
    map_m = {}
    for terms, w in cm.items():
        if terms[0] not in map_m.keys() and terms[1] not in map_m.keys():
            map_m[terms[0]] = num
            num += 1
            map_m[terms[1]] = num
            num += 1
        elif terms[0] not in map_m.keys():
            map_m[terms[0]] = num
            num += 1
        elif terms[1] not in map_m.keys():
            map_m[terms[1]] = num
            num += 1
        else:
            continue
    for key in map_m.keys():
        word_list.append(key)
    mat = np.zeros((num, num))
    for terms, w in cm.items():
        if terms[0] in map_m.keys() and terms[1] in map_m.keys():
            mat[map_m[terms[0]]][map_m[terms[1]]] = w
        else:
            raise MyError('Something Wrong!!!')
    my_graph = nx.from_numpy_matrix(mat)
    return my_graph, word_list, sentence_list

def getAbstract_nx(text, windows, punc=('。', '！', '？', '......', '\n'), alpha=0.85, topK=3):
    """
    Get the abstract.
    :param text: the original text.
    :param windows: the size of windows.
    :param punc: the punctuation list, eg. ['。', '！', '？', '......', '\n'].
    :param alpha: the damping factor.
    :param topK: the size of sentences which will be returned.
    :return: a list that contain topK key words.
    """
    my_graph, my_words, my_sentences = getMyAbstractGraph(text, windows, punc)

    scores = nx.pagerank(my_graph, alpha=alpha)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result_dict = {}
    result_list = []
    for i in range(len(sorted_scores)):
        if topK and len(result_dict) == topK:
            break
        if sorted_scores[i][0] < len(my_words):
            # result = (my_words[sorted_scores[i][0]], sorted_scores[i][1])
            result_dict[my_words[sorted_scores[i][0]]] = sorted_scores[i][1]
            # result_list.append(result)

    # filter_list = []
    for i in range(len(my_sentences)):
        if my_sentences[i] in result_dict.keys() and my_sentences[i] not in result_list:
            # filter_list.append(my_sentences[i])
            # result = (my_sentences[i], result_dict[my_sentences[i]])
            senl = my_sentences[i].lstrip()
            sen = senl.rstrip()
            result_list.append(sen)

    return result_list


class MyError:
    pass


class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)
        outSum = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)

        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())
        for x in range(10):  # 10 iters
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s

        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])

        itervalues = lambda d: iter(d.values())
        for w in itervalues(ws):
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws


