# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:23:12 2021

@author: Johanan
"""
import os

import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common.initializer import Normal
from mindspore.train.callback import LossMonitor
import dataset_manager as dm

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files if os.path.splitext(basename)[1]==".ckpt"]
    return max(paths)


class myNN(nn.Cell):
    # define the operator required
    def __init__(self, num_class=10, num_channel=3,count=0):
        super(myNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 16, 5,pad_mode='valid')
        self.conv2 = nn.Conv2d(16, 32, 5)
        
        self.fc1 = nn.Dense(3872, 128, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(128, 64, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(64, num_class, weight_init=Normal(0.02))
        
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(keep_prob=0.9)

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_net(network_model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """Define the training method."""
    print("============== Starting Training ==============")
    # load training dataset
    ds_train = dm.create_dataset(os.path.join(data_path, "./MindSpore_train_images_dataset/train"), do_train=True, repeat_num=1)
    network_model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=sink_mode)


def test_net(network, network_model, data_path):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    # load the saved model for evaluation
    latest_file = newest(data_path)
    print(latest_file)
    param_dict = load_checkpoint(latest_file)
    # load parameter to the network
    load_param_into_net(network, param_dict)
    # load testing dataset
    ds_eval = dm.create_dataset(os.path.join(data_path, "./MindSpore_train_images_dataset/test"), do_train=False)
    acc = network_model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))

    

