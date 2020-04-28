#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : face_paddle.py
# @Author: WangYe
# @Date  : 2020/4/15
# @Software: PyCharm
import paddlehub as hub
import cv2
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
module = hub.Module(name="pyramidbox_lite_server_mask")
test_img_path = args.path

image = cv2.imread(test_img_path)
# # set input dict
input_dict = {"data": [cv2.imread(test_img_path)]}
results = module.face_detection(data=input_dict)
print(results)
font = cv2.FONT_HERSHEY_SIMPLEX
for i in results:
    tar = i["data"].values()
    temp_list = []
    for j in tar:
        temp_list.append(j)
    # print(temp_list)
    out = cv2.rectangle(image, (int(temp_list[1]),
                                int(temp_list[3])),
                        (int(temp_list[2]),
                         int(temp_list[4])), (0,0,255), 1)

    text = temp_list[0]
    out = cv2.putText(out, text, (int(temp_list[1]), int(temp_list[3])), font, 1, (0, 255, 255), 2)
    # out = cv2.rectangle(image, (23, 32), (11,11), (0,0,255), 2)
    cv2.imwrite(test_img_path + '_out.jpg',out)
# print(results)
