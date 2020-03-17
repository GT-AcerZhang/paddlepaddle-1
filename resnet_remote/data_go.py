#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : data_go.py
# @Author: WangYe
# @Date  : 2019/12/8
# @Software: PyCharm
import importlib

from .cv2 import *

# wildcard import above does not import "private" variables like __version__
# this makes them available
globals().update(importlib.import_module('cv2.cv2').__dict__)
import numpy as np
from PIL import Image
import paddle.fluid as fluid
path = 'label1.png'
path2 = 'label2.png'
q = cv2.imread(path)

matrix = np.asarray(q)
# matrix = matrix[:,:,:-1]
print(matrix)
print(matrix.shape)
print(np.max(matrix))
# a = fluid.global_scope().find_var('var_a')
# b = fluid.global_scope().find_var('var_a')
# a.set(matrix)
# b.set(matrix1)
# num_classes = 6
# def mean_iou(pred, label):
#     # label = fluid.layers.elementwise_min(
#     #     label, fluid.layers.assign(np.array(
#     #         [num_classes], dtype=np.int32)))
#     # label_ignore = (label == num_classes).astype('int32')
#     # label_nignore = (label != num_classes).astype('int32')
#     #
#     # pred = pred * label_nignore + label_ignore * num_classes
#
#     miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes + 1)
#     return miou, wrong, correct
# a,b,c = mean_iou(a,b)
# mp = (b + c) != 0
# miou2 = np.mean((b / (mp)))