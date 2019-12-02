#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : image_detect.py
# @Author: WangYe
# @Date  : 2019/11/16
# @Software: PyCharm
import paddle
import paddle.fluid as fluid
import os
from PIL import Image
import numpy as np
'''
read image
'''
image_path = 'D:/code/data/carnumber/carnumber/tf_car_license_dataset/train_images/training-set/'
def read_image(inputPath):
    train_list = []
    label_list = []
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        #print(eachDir)           #文件名存储为标签
        eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        for root, dirs, files in os.walk(eachPath):
            #files为每个类文件列表
            for file in files:
                image_file = os.path.join(root,file)  # 拼接目录
                x = Image.open(image_file)  # 打开图片
                matrix = np.asarray(x)  #图片矩阵
                train_list.append(matrix)
                label_list.append([int(eachDir)])
    def reader():         #生成器生成训练集
        for i in range(len(label_list)):
            yield train_list[i],label_list[i]
    return reader
batch_size = 64
data = read_image(inputPath=image_path)#数据集4285*[1*40*32,1]
train_reader = paddle.batch(paddle.reader.shuffle(data, buf_size=500),batch_size=batch_size)
'''
引入paddle神经网络
'''
#张量定义：
image = fluid.layers.data(name = 'image',shape = [1,40,32],dtype = 'float32')
label = fluid.layers.data(name = 'label',shape = [1],dtype = 'int64')

def softmax_regression():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    predict = fluid.layers.fc(
        input=img, size=10, act='softmax')
    return predict

def multilayer_perceptron():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 第一个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=img, size=200, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction

def convolutional_neural_network():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 第一个卷积-池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # 第二个卷积-池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction
#定义损失，反向传播
model = multilayer_perceptron(image)
cost = fluid.layers.cross_entropy(input = model,label = label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input = model,label = label)

optimizer = fluid.optimizer.AdadeltaOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

#定义网络训练
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(place = place , feed_list= [image,label])

NUM = 50000
for pass_id in range(NUM):
    for batch_id,data in enumerate(train_reader()):
        train_cost,train_acc = exe.run(program = fluid.default_main_program(),
                                       feed=feeder.feed(data),
                                       fetch_list= [avg_cost,acc])

        if batch_id %100 == 0:
            print(pass_id,batch_id,train_cost[0],train_acc[0])