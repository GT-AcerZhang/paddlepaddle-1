#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : read_num.py
# @Author: WangYe
# @Date  : 2020/1/13
# @Software: PyCharm
import os
import cv2
a ={}
path = 'out'

for i in os.listdir(path):

    new_path = path + '/' + i
    label = cv2.imread(new_path)
    # print(type(label))
    for i in label:
        for j in i:
            for k in j:
                if k not in a:
                    a[k] = 1
                else:
                    a[k] = a[k] +1
print(a)