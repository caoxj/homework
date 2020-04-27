'''
Created on 2020/04/14

@author: sou
'''

import torch
import matplotlib.pyplot as plt
import pandas as pd

import model
import config as CONFIG
import datasets
from torch.autograd import Variable
from torch import max
import numpy as np

#モジュール読み込み
net = model.Net(CONFIG.input_size, CONFIG.hidden_size, CONFIG.num_classes)
net.load_state_dict(torch.load(CONFIG.NET_NAME))

#テストデータの読み込み
#test_loader, test_dataset = datasets.load_data_test()

#kaggleのcsvファイルから取得
test_loader, test_dataset = datasets.load_data_test("../data/digit-recognizer/test.csv")

#評価
#モデルを評価してみます。
#訓練に使ったデータとは別のデータ（テストデータ）を使って、未知のデータをどのくらい正確にクラス分類できるのか確かめます。
def evaluate(net, test_loader=test_loader):
    w = np.empty((0,2), int)
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        outputs = net(images)
        _, predicted = max(outputs.data, 1)
        
        #ImageId列
        ids = np.arange(total+1, total+1+labels.size(0))
        ids = ids.reshape([labels.size(0), 1])
        #Label列
        lbls = np.array(predicted.reshape(labels.size(0), 1))
        #ImageId列前にLabel列挿入
        tmp = np.insert(lbls, [0], ids, 1)
        w = np.append(w, tmp, axis=0)
        
        total += labels.size(0)
        #correct += (predicted.cpu() == labels).sum()
 
    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
    df = pd.DataFrame(w,columns=["ImageId","Label"])
    print(w)
    df.to_csv("output.csv", header=True, index = False)
 
evaluate(net)

"""
test_iter = iter(test_loader)
inputs, labels = test_iter.next()
outputs = net(Variable(inputs.view(-1, 28*28)))
_, predicted = max(outputs.data, 1)

plt.imshow(inputs[7].numpy().reshape(28, 28), cmap='gray')
print('Label:', predicted[7])
"""
