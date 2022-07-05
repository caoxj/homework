'''
Created on 2020/04/24

@author: sou
'''
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim

import pytorchSLTM.config as CONFIG
import pytorchSLTM.model as model
import pytorchSLTM.datasets
from pytorchSLTM.datasets import sentence2index


#================================================================================
#  ■　二、予測精度確認
#
#================================================================================
categories, data = pytorchSLTM.datasets.load_dotcom()

traindata, testdata = train_test_split(data, train_size=0.7)
word2index = pytorchSLTM.datasets.word2index(data["title"])
category2idx = pytorchSLTM.datasets.category2index(categories)

# データ全体の単語数
VOCAB_SIZE = len(word2index)
# 分類先のカテゴリの数
TAG_SIZE = len(categories)
#モジュール読み込み
model = model.LSTMClassifier(CONFIG.EMBEDDING_DIM, CONFIG.HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
model.load_state_dict(torch.load(CONFIG.NET_NAME))


# テストデータの母数計算
test_num = len(testdata)
# 正解の件数
a = 0
# 勾配自動計算OFF
with torch.no_grad():
    for title, category in zip(testdata["title"], testdata["category"]):
        # print(title,"/",category)
        # テストデータの予測
        inputs = sentence2index(title,word2index)
        out = model(inputs)

        # outの一番大きい要素を予測結果をする
        _, predict = torch.max(out, 1)  # @UndefinedVariable
        # print(categories[predict])

        answer = pytorchSLTM.datasets.category2tensor(category, category2idx)
        # print(categories[answer])
        if predict == answer:
            a += 1
print("predict : ", a / test_num)
