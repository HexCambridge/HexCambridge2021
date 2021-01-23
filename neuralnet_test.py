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


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files if os.path.splitext(basename)[1]==".ckpt"]
    return max(paths)

class myNN(nn.Cell):
    """Lenet network structure."""
    # define the operator required
    def __init__(self, num_class=10, num_channel=3):
        super(myNN, self).__init__()
        self.conv11 = nn.Conv2d(num_channel, 16, 3,pad_mode='valid')
        self.conv12 = nn.Conv2d(16, 16, 3)
        
     #   self.conv21 = nn.Conv2d(16, 8, 3,pad_mode='valid')
      #  self.conv22 = nn.Conv2d(8, 8, 3)
        
        self.fc1 = nn.Dense(14400, 256, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(256, 64, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(64, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        
        s = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_net(network_model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """Define the training method."""
    print("============== Starting Training ==============")
    # load training dataset
    ds_train = dm.create_dataset(os.path.join(data_path, "./MindSpore_train_images_dataset/train"), do_train=True, repeat_num=6)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Team GJPT')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    dataset_sink_mode = not args.device_target == "CPU"
    
    data_path = os.getcwd()
    # learning rate setting
    lr = 0.01
    momentum = 0.9
    dataset_size = 1

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_epoch = 1
    # create the network
    net = myNN()
    # define the optimizer
    net_opt = nn.Momentum(net.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=5,keep_checkpoint_max=2)
    # save the network model and parameters for subsequence fine-tuning
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck,directory=data_path)
    # group layers into an object with training and evaluation features
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    train_net(model, train_epoch, data_path, dataset_size, ckpoint, dataset_sink_mode)
    test_net(net, model, data_path)