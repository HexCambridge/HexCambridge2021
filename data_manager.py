# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:52:12 2021

@author: Johanan
"""

my_dict = "C:/Users/Johanan/OneDrive/Documents/HexCambridge/MindSpore/MindSpore_train_images_dataset/data_batch_1.bin"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict