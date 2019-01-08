# -*- coding: utf-8 -*-

"""

@project: python and ML
@author: Cherry_L411@163.com
@file: TextRank.py
@data: 2019-01-08
@Pay attention：Most of this code was copied from Jieba.

"""

from collections import defaultdict
import jieba.posseg
import sys

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

def pairfilter(wp, pos_filt, stop_words):
    return (wp.flag in pos_filt and len(wp.word.strip()) >= 2 and wp.word.lower() not in stop_words)

def getKeyWords(sentence, topK=3, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False, span=5, STOP_WORDS = ("的")):
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