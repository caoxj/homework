'''
Created on 2020/04/15

@author: sou
'''

import torch.utils.data
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import numpy as np

class KaggleDataset(torch.utils.data.Dataset):
    '''
    classdocs
    '''
    def __init__(self, path, train=True):
        '''
        Constructor
        '''
        # csvデータの読み出し
        df = pd.read_csv(path)
        
        """
        dataframe = []
        with open(path, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                dataframe.append(row)
        """
        self.dataframe = df.values
        self.train = train
        
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
        
    # データとラベルの取得
    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        t = transforms.ToTensor()
        # labelはint型に変換
        label = int(self.dataframe[idx][0])
        # dataはtorch.Tensorに変換しておく必要あり
        # ※画像の場合などは、transformにtransforms.ToTensorを指定して変換
        if self.train :
            data = torch.Tensor([self.dataframe[idx][1:]])
        else:
            data = torch.Tensor([self.dataframe[idx][:]])
        # data, labelの順でリターン
        return data, label
    
