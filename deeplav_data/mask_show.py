#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : mask_show.py
# @Author: WangYe
# @Date  : 2019/12/18
# @Software: PyCharm
import numpy as np
import cv2
path = 'image_20_12000_4800.png'
label = cv2.imread(path)
label = label[:, :, 0]
cmap = np.array([[0, 0, 0],
                 [128, 0, 0],
                 [128, 128, 0],
                 [0, 128, 0],
                 [0, 0, 128]]
                )
y = label
r = y.copy()
g = y.copy()
b = y.copy()
# print('r=',r)
for l in range(0, len(cmap)):
    r[y == l] = cmap[l, 0]
    g[y == l] = cmap[l, 1]
    b[y == l] = cmap[l, 2]
label = np.concatenate((np.expand_dims(b, axis=-1), np.expand_dims(g, axis=-1),
                        np.expand_dims(r, axis=-1)), axis=-1)
cv2.imwrite(path.split('/')[-1], label)