#!/usr/bin/env python 3.6
# -*- coding: utf-8 -*-
"""
# @Company ：华中科技大学机械学院数控中心
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2019/11/27 16:25
# @File    : decision_tree_classification_visualization.py
# @Software: PyCharm
"""

import pydotplus
# from StringIO import StringIO

from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()

print(iris.data.shape)

print(iris.data[0])
print(iris.target[0])
print(len(iris.data[1]))
print(iris.feature_names)
print(iris.target_names)

test_index = [0, 50, 100]
print(test_index)

# traing data
train_target = np.delete(iris.target, test_index)
# print(train_target)
train_data = np.delete(iris.data, test_index, axis=0)
# print(train_data)

# testing data
test_target = iris.target[test_index]
test_data = iris.data[test_index]


clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(train_data, train_target)

print(test_target)

print(clf.predict(test_data))

# viz code


dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("iris_1.png")  # 当前文件夹生成out.png