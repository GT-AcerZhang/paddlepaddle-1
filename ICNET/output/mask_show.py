#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : mask_show.py
# @Author: WangYe
# @Date  : 2019/12/18
# @Software: PyCharm
import numpy as np
import paddle.dataset as dataset
import cv2
from PIL import Image

path = '4.png'
label = cv2.imread(path)
#[4, 3, 12, 24, 22, 28, 18, 20, 14, 5, 21, 27, 13, 9, 8, 23, 10, 2]
# label1 = dataset.image.load_image(path, is_color=False).astype("float32") +1
print(label.shape)
# a ={}
# for i in label1:
#
#     for k in i:
#         if k not in a:
#             a[k] = 1
#         else:
#             a[k] = a[k] +1
# print(a)
# IMG_MEAN = np.array((103, 116, 123))
# label =label -IMG_MEAN
# cv2.imwrite("out.png", label)
# print(label[:,:,2])
# label = label[:,:,2]
d = label[:,:,2]
#{255: 178305, 2: 725171, 8: 100671, 10: 1121, 5: 34055, 7: 40722, 11: 136356, 12: 28209, 4: 2750, 1: 377604, 18: 7314, 0: 464874}
# #[3, 21, 11, 23, 4, 17, 26, 20, 8, 7, 33, 22, 1]
#{2: 483927, 1: 347441, 3: 608632}
a ={}
for i in d:

    for k in i:
        if k not in a:
            a[k] = 1
        else:
            a[k] = a[k] +1
print(a)

# for i in d:
#     a = []
#     for k in i:
#         for j in k:
#             if j not in a:
#                 a.append(j)
    # print(a)
# Image.fromarray(a).show()
# # a.show()
# print(label.shape)
label = label[:, :, 1]
cmap = np.array([
    [128, 64, 128],
    [244, 35, 231],
    [69, 69, 69]
    # 0 = road, 1 = sidewalk, 2 = building
    ,
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153]
    # 3 = wall, 4 = fence, 5 = pole
    ,
    [250, 170, 29],
    [219, 219, 0],
    [106, 142, 35]
    # 6 = traffic light, 7 = traffic sign, 8 = vegetation
    ,
    [152, 250, 152],
    [69, 129, 180],
    [219, 19, 60]
    # 9 = terrain, 10 = sky, 11 = person
    ,
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 69]
    # 12 = rider, 13 = car, 14 = truck
    ,
    [0, 60, 100],
    [0, 79, 100],
    [0, 0, 230]
    # 15 = bus, 16 = train, 17 = motocycle
    ,
    [119, 10, 32]
]
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
cv2.imwrite('out.png', label)