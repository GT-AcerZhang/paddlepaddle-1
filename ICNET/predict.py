#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : predict.py
# @Author: WangYe
# @Date  : 2019/12/31
# @Software: PyCharm
import cv2
import os
import numpy as np
import paddle.fluid as fluid
import paddle
# def color(input):
#     """
#     Convert infered result to color image.
#     """
#     result = []
#     for i in input.flatten():
#         result.append(
#             [label_colours[i][2], label_colours[i][1], label_colours[i][0]])
#     result = np.array(result).reshape([input.shape[0], input.shape[1], 3])
#     return result
model_path = "9000"
# IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
import sys
from icnet import icnet
label_colours = [
    [128, 64, 128],
    [244, 35, 231],
    [69, 69, 69]
    # 0 = road, 1 = sidewalk, 2 = building
    ,
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153]
    # 3 = wall, 4 = fence, 5 = pole
]

def color(input):
    """
    Convert infered result to color image.
    """
    result = []
    for i in input.flatten():
        result.append(
            [label_colours[i][2], label_colours[i][1], label_colours[i][0]])
    result = np.array(input).reshape([input.shape[0], input.shape[1], 3])
    return result
from PIL import Image
import paddle.dataset as dataset
data_shape = [3, 1200, 1200]
DATA_PATH = '2.jpg'
image = dataset.image.load_image(DATA_PATH, is_color=True).astype("float32")
# print(image)
images = fluid.layers.data(name='image', shape=data_shape, dtype='float32')
qq, pp, sub124_out = icnet(images, 19,np.array(data_shape[1:]).astype("float32"))
predict = fluid.layers.resize_bilinear(
        sub124_out, out_shape=data_shape[1:3])
# print(predict)
predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
predict = fluid.layers.reshape(predict, shape=[-1, 19])
_, predict = fluid.layers.topk(predict, k=1)
predict1 = fluid.layers.reshape(
    predict,
    shape=[data_shape[1], data_shape[2], -1])  # batch_size should be 1
inference_program = fluid.default_main_program().clone(for_test=True)
place = fluid.CPUPlace()
# if args.use_gpu:
#     place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
assert os.path.exists(model_path)
fluid.io.load_params(exe, model_path)
print("loaded model from: %s" % model_path)
# sys.stdout.flush()
# image -= IMG_MEAN
img = paddle.dataset.image.to_chw(image)[np.newaxis, :]
# print(image.shape())
image_t = fluid.LoDTensor()
image_t.set(img, place)
result = exe.run(inference_program,
                         feed={"image": image_t},
                         fetch_list=[predict1])
cv2.imwrite("output" + "/" +"_result.png",result[0])
# image = Image.fromarray(result)  #将之前的矩阵转换为图片
# image.show()            #调用本地软件显示图片，win10是叫照片的工具
q = result[0]
# print(type(q))
print(sum(q))
# image = Image.fromarray(q)  #将之前的矩阵转换为图片
# image.show()            #调用本地软件显示图片，win10是叫照片的工具