'''
Created on 2020/04/14

@author: sou
'''
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import kaggleDataset

import config

def load_data_train(rpath = "../data"):
#torchvision.datasetsからMNISTという手書き文字認識のデータセットを利用
# MNIST Dataset 
    train_dataset = datasets.MNIST(root=rpath, 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=False)
 
    return train_dataset

def get_minibatch(train_dataset):
#minibatch学習の準備
# Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=config.batch_size, 
                                           shuffle=True)
 
    return train_loader

def load_data_test(rpath = "../data"):
#torchvision.datasetsからMNISTという手書き文字認識のデータセットを利用
# MNIST Dataset 
    #test_dataset = datasets.MNIST(root=rpath, 
    #                          train=False, 
    #                          transform=transforms.ToTensor())
    test_dataset = kaggleDataset.KaggleDataset("../data/digit-recognizer/test.csv", False)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=config.batch_size, 
                                          shuffle=False)
    
    return test_loader, test_dataset