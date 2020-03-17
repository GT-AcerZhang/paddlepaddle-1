#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import paddle
import paddle.fluid as fluid
import numpy as np
import argparse
from reader import CityscapeDataset
import reader
import models
import sys
import utility
import paddle.dataset as dataset
data_shape = [3, 1200, 1200]
DATA_PATH = '2.jpg'
image = dataset.image.load_image(DATA_PATH, is_color=True).astype("float32")
images = fluid.layers.data(name='image', shape=data_shape, dtype='float32')


models.is_train = False
models.default_norm_type = args.norm_type
deeplabv3p = models.deeplabv3p

image_shape = [1025, 2049]
eval_shape = [1024, 2048]

sp = fluid.Program()
tp = fluid.Program()
batch_size = 1
reader.default_config['crop_size'] = -1
reader.default_config['shuffle'] = False
num_classes = 5

with fluid.program_guard(tp, sp):
    if args.use_py_reader:
        py_reader = fluid.layers.py_reader(capacity=64,
                                        shapes=[[1, 3, 0, 0], [1] + eval_shape],
                                        dtypes=['float32', 'int32'])
        img, label = fluid.layers.read_file(py_reader)
    else:
        img = fluid.layers.data(name='img', shape=[3, 0, 0], dtype='float32')
        label = fluid.layers.data(name='label', shape=eval_shape, dtype='int32')

    img = fluid.layers.resize_bilinear(img, image_shape)
    logit = deeplabv3p(img)
    logit = fluid.layers.resize_bilinear(logit, eval_shape)
    pred = fluid.layers.argmax(logit, axis=1).astype('int32')
    miou, out_wrong, out_correct = mean_iou(pred, label)

tp = tp.clone(True)

place = fluid.CPUPlace()
if args.use_gpu:
    place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(sp)

if args.init_weights_path:
    print("load from:", args.init_weights_path)
    load_model()

dataset = CityscapeDataset(args.dataset_path, 'val')
if args.total_step == -1:
    total_step = len(dataset.label_files)
else:
    total_step = args.total_step

batches = dataset.get_batch_generator(batch_size, total_step, use_multiprocessing=args.use_multiprocessing)
if args.use_py_reader:
    py_reader.decorate_tensor_provider(lambda :[ (yield b[0],b[1]) for b in batches])
    py_reader.start()

sum_iou = 0
all_correct = np.array([0], dtype=np.int64)
all_wrong = np.array([0], dtype=np.int64)

for i in range(total_step):
    if not args.use_py_reader:
        _, imgs, labels, names = next(batches)
        result = exe.run(tp,
                         feed={'img': imgs,
                               'label': labels},
                         fetch_list=[pred, miou, out_wrong, out_correct])
    else:
        result = exe.run(tp,
                         fetch_list=[pred, miou, out_wrong, out_correct])

    wrong = result[2][:-1] + all_wrong
    right = result[3][:-1] + all_correct
    all_wrong = wrong.copy()
    all_correct = right.copy()
    mp = (wrong + right) != 0
    miou2 = np.mean((right[mp] * 1.0 / (right[mp] + wrong[mp])))
    print('step: %s, mIoU: %s' % (i + 1, miou2))

print('eval done!')
