#!/usr/bin/env python 3.6
# -*- coding: utf-8 -*-
"""
# @Company ：华中科技大学机械学院数控中心
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2019/11/27 16:27
# @File    : decision_tree_regression_visualization.py
# @Software: PyCharm
"""

from sklearn.datasets.california_housing import fetch_california_housing
import pydotplus

housing = fetch_california_housing()  ###调用sklearn自带的数集
# print(housing.DESCR)
print(housing.data.shape)
print(housing.data[1])
print(len(housing.data[1]))
print(housing.feature_names)
print(housing.target)

#####取要使用的特征做决策树
from sklearn import tree

dtr = tree.DecisionTreeRegressor(max_depth=4)
dtr.fit(housing.data[:, [3, 4, 5, 6, 7]], housing.target)  ###取房子所在的经度和纬度
###输出构造决策树模型的一些基本参数，有些事默认的
print(dtr)

# 要可视化显示 首先需要安装 graphviz   http://www.graphviz.org/Download..php
dot_data = tree.export_graphviz(
    dtr,
    out_file=None,
    feature_names=housing.feature_names[3:8],
    filled=True,
    impurity=False,
    rounded=True
)

# pip install pydotplus


import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
graph.write_png("out_1.png")

'''自动选择最合适的特征参数'''
####用切分的数据训练来进行特征参数的选择
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = \
    train_test_split(housing.data, housing.target, test_size=0.1,
                     random_state=42)  ##，取其中10%做测试集，random_state指定每次随机结果都是一致的
dtr = tree.DecisionTreeRegressor(random_state=42)  ##构造树模型
dtr.fit(data_train, target_train)



print("==============================")
print("测试分类的准确度:", dtr.score(data_test, target_test))  ##测试检验分类的准确度


'''随机森林'''
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=42)
rfr.fit(data_train, target_train)
print(rfr.score(data_test, target_test))



