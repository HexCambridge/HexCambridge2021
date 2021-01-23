# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:16:52 2021

@author: Johanan
"""

import os
import sys
import requests
from urllib.parse import urlparse
import gzip
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size

def unzipfile(gzip_path):
    """unzip dataset file
    Args:
        gzip_path: dataset file path
    """
    open_file = open(gzip_path.replace('.gz', ''), 'wb')
    gz_file = gzip.GzipFile(gzip_path)
    open_file.write(gz_file.read())
    gz_file.close()




def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="CPU", distribute=False):
    """
    create a train or evaluate cifar10 dataset for resnet50
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset
    """

    if distribute:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
    else:
        device_num = 1
    if device_num == 1:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            C.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        C.Resize((32, 32)),
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set