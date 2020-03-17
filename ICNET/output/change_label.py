#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : change_label.py
# @Author: WangYe
# @Date  : 2020/1/3
# @Software: PyCharm
import cv2
import os
from PIL import Image
for i in range(6):
    path = 'label/label_' + str(i)
    path1 = 'label1/label_'+ str(i)
    for _,_,files in os.walk(path):
        if len(files) != 0:
            for file in files:
                new_path = path + '/' + file
                label = cv2.imread(new_path)
                d = label*5
                cv2.imwrite(path1 + '/' + file, d)