# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: stanfordnlp.py
@date: 2019-04-03

"""

from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree

# nlp = StanfordCoreNLP(r'/Users/liunianci/Downloads/stanford-corenlp-full-2016-10-31/', lang='zh')    # lang='zh'英文使用 lang='en'
nlp = StanfordCoreNLP('/Users/liunianci/Downloads/stanford-corenlp-full-2018-10-05/',
                      lang='zh')  # lang='zh'英文使用 lang='en'

sentence = "帮我打开客厅的吸顶灯。"
# sentence = "The book is very interesting."
result = nlp.parse(sentence)
my_tree = Tree.fromstring(result)
# print(nlp.word_tokenize(sentence))
tree_dict = {}

# tree_root是树的根节点
# tree_root = my_tree[0]
def tree_to_dict(tree_root, tree_dict):
    # 如果节点没有子节点，递归结束
    #     print(len(tree_root))
    if len(tree_root) < 1:
        return
    for i in range(len(tree_root)):
        child = tree_root[i]

        #         print(type(tree_root))
        if type(tree_root) == str:
            child_data = tree_root
        #             continue
        else:
            child_data = tree_root.label()
        print(child_data)
        #         print(child)
        print(tree_dict)
        if not tree_dict.get(child_data):
            tree_dict[child_data] = {}
            if type(tree_root) == str:
                break
            tree_to_dict(child, tree_dict[child_data])

        else:
            # 如果tree_dict有对应的节点地址键，直接继续递归
            tree_to_dict(child, tree_dict[child_data])
tree_to_dict(my_tree, tree_dict)
print(tree_dict)

