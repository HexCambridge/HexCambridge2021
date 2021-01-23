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
import multiprocessing


def unzipfile(gzip_path):
    """unzip dataset file
    Args:
        gzip_path: dataset file path
    """
    open_file = open(gzip_path.replace('.gz', ''), 'wb')
    gz_file = gzip.GzipFile(gzip_path)
    open_file.write(gz_file.read())
    gz_file.close()




def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="CPU"):


    data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=multiprocessing.cpu_count(), shuffle=True)


    # define map operations
    trans = []
    if do_train:
        trans += [
            C.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=multiprocessing.cpu_count())
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=multiprocessing.cpu_count())

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=False)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set