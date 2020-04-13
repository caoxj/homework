'''
訓練済のネットワークを利用して推測を行う
Created on 2020/04/08

@author: sou
'''

import chainer
import datasets
import numpy as np
import pandas as pd

import chainer.links as L
import chainer.functions as F
from chainer import Sequential

import const as CONST


############--訓練済みネットワークを用いた推論--################
#保存したネットワークを読み込みます
loaded_net = Sequential(
    L.Linear(CONST.n_input, CONST.n_hidden), F.relu,
    L.Linear(CONST.n_hidden, CONST.n_hidden), F.relu,
    L.Linear(CONST.n_hidden, CONST.n_output)
)
#パラメータを読み込ませ
chainer.serializers.load_npz(CONST.NET_NAME, loaded_net)
    
############--訓練済みネットワークでtest.csvを推論--################
x_v,x_t,x_id = datasets.get_test()
#推論を行い
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_t = loaded_net(x_v)
#テストデータの 0 番目のサンプルの予測結果を確認
print("================================================-")
w = np.empty((0,2), int)

istrue = 0
for i in range(len(x_v)):
    y = np.argmax(y_t[i,:].array)
    if y==x_t[i]: 
        istrue = istrue+1
    print(y==x_t[i],"   ",y,"/",x_t[i],":",x_v[i])
    w = np.append(w, np.array([[x_id[i],y]]), axis=0)
    
print(istrue/len(x_v))
df = pd.DataFrame(w,columns=["PassengerId","Survived"])
df.to_csv("output.csv", header=True, index = False)