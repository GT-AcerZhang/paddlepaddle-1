#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : lr_test.py
# @Author: WangYe
# @Date  : 2020/1/7
# @Software: PyCharm
from utils import add_arguments, print_arguments, get_feeder_data, check_gpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
import paddle.fluid as fluid
global_step = 60000
TOTAL_STEP = 10000
LEARNING_RATE = 0.003
POWER = 0.9
def poly_decay():
    decayed_lr = LEARNING_RATE * (pow(
        (1 - global_step / TOTAL_STEP), POWER))
    print(decayed_lr)
    return decayed_lr
poly_decay()