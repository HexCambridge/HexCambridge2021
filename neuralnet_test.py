# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:23:12 2021

@author: Johanan
"""
import os
import argparse
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context, Model, load_checkpoint, load_param_into_net
from mindspore.common.initializer import Normal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.nn.metrics import Accuracy
from mindspore import dtype as mstype
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
import dataset_manager as dm



class LeNet5(nn.Cell):
    """Lenet network structure."""
    # define the operator required
    def __init__(self, num_class=10, num_channel=3):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 1, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(1, 16, 5, pad_mode='valid')
        self.conv3 = nn.Conv2d(16, 6, 5, pad_mode='valid')
        self.conv4 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(44944, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_net(network_model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """Define the training method."""
    print("============== Starting Training ==============")
    # load training dataset
    ds_train = dm.create_dataset(data_path, 32, repeat_size)
    network_model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=sink_mode)


def test_net(network, network_model, data_path):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    # load the saved model for evaluation
    param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
    # load parameter to the network
    load_param_into_net(network, param_dict)
    # load testing dataset
    ds_eval = dm.create_dataset(data_path, False)
    acc = network_model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    dataset_sink_mode = not args.device_target == "CPU"
    
    data_path = "C:/Users/Johanan/OneDrive/Documents/HexCambridge/Project/MindSpore_train_images_dataset"
    # learning rate setting
    lr = 0.01
    momentum = 0.9
    dataset_size = 1

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_epoch = 1
    # create the network
    net = LeNet5()
    # define the optimizer
    net_opt = nn.Momentum(net.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # save the network model and parameters for subsequence fine-tuning
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    # group layers into an object with training and evaluation features
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    #train_net(model, train_epoch, data_path, dataset_size, ckpoint, dataset_sink_mode)
    test_net(net, model, data_path)