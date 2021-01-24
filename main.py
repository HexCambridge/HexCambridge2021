# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 14:53:42 2021

@author: Johanan
"""

import os
import neuralnet_test
from mindspore.nn.metrics import Accuracy
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import  Model
import mindspore.nn as nn

if __name__ == "__main__":
    
    data_path = os.getcwd()
    # learning rate setting
    lr = 0.01
    momentum = 0.9
    dataset_size = 1
    #net = resnet.resnet50(class_num=10)
    net = neuralnet_test.myNN()
    
    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_epoch = 1
    # create the network

    # define the optimizer
    net_opt = nn.Momentum(net.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=5,keep_checkpoint_max=2)
    # save the network model and parameters for subsequence fine-tuning
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck,directory=data_path)
    
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    
   # neuralnet_test.train_net(model, train_epoch, data_path, dataset_size, ckpoint, sink_mode=True)
    neuralnet_test.test_net(net, model, data_path)