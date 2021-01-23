# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:59:35 2021

@author: Johanan
"""

import os
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model
from mindspore import load_checkpoint, load_param_into_net
import argparse
from mindspore import context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

class LeNet5(nn.Cell):
    """
    Lenet network structure
    """
    #define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    #use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_net(args, model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """define the training method"""
    print("============== Starting Training ==============")
    #load training dataset
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=sink_mode) # train
    
def test_net(network,model,data_path):
    """define the evaluation method"""
    print("============== Starting Testing ==============")
    #load the saved model for evaluation
    param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
    #load parameter to the network
    load_param_into_net(network, param_dict)
    #load testing dataset
    ds_eval = create_dataset(os.path.join(data_path, "test")) # test
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    dataset_sink_mode = not args.device_target == "CPU"
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    #learning rate setting
    lr = 0.01
    momentum = 0.9
    #create the network
    net = LeNet5()
    #define the optimizer
    net_opt = nn.Momentum(net.trainable_params(), lr, momentum)
       # set parameters of check point
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # apply parameters of check point
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    train_epoch = 1
    dataset_path = "./MNIST_Data"
    dataset_size = 1
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net(args, model, train_epoch, dataset_path, dataset_size, ckpoint, dataset_sink_mode)
    test_net(net, model, dataset_path)
    