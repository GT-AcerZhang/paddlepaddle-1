#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : read.py
# @Author: WangYe
# @Date  : 2019/11/25
# @Software: PyCharm
import csv
import numpy as np
path = 'train.csv'
def readCSV(path):  # 读取答案
    csvfile = open(path, encoding='UTF-8')  # 打开一个文件
    reader = csv.DictReader(csvfile)  # 返回的可迭代类型
    column = [row['image_path'] for row in reader]
    print(column)
    return column
def readCSV1(path):  # 读取问题
    csvfile = open(path, encoding='UTF-8')  # 打开一个文件
    reader = csv.DictReader(csvfile)  # 返回的可迭代类型
    column1 = [row['gt_path'] for row in reader]
    print(len(column1))
    return column1
a = readCSV(path)
b = readCSV1(path)
with open('train.list' ,'w') as f:
    for i in range(len(a)):
        w = 'image/'+a[i] + ' ' +'lable/' +b[i] +'\r'
        f.write(w)
    f.close()