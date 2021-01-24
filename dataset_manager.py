# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:16:52 2021

@author: Johanan
"""

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="CPU"):


    data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)


    # define map operations
    trans = []
    if do_train:
        trans += [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        C.Resize((48,48)),
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.shuffle(buffer_size=10)
    data_set = data_set.batch(batch_size, drop_remainder=False)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set