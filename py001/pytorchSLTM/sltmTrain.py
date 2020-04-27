'''
Created on 2020/04/23

@author: sou
'''

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim

import config as CONFIG
import model
import datasets
from pytorchSLTM.datasets import sentence2index

#================================================================================
#  ■　正解ラベルの変換
#
#================================================================================
categories, data = datasets.load()
word2index = datasets.word2index(data["title"])
category2idx = datasets.category2index(categories)

#================================================================================
#  ■　一、学習
#
#================================================================================
# 元データを7:3に分ける（7->学習、3->テスト）
traindata, testdata = train_test_split(data, train_size=0.7)

# データ全体の単語数
VOCAB_SIZE = len(word2index)
# 分類先のカテゴリの数
TAG_SIZE = len(categories)
# モデル宣言
model = model.LSTMClassifier(CONFIG.EMBEDDING_DIM, CONFIG.HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
# 損失関数はNLLLoss()を使う。LogSoftmaxを使う時はこれを使うらしい。
loss_function = nn.NLLLoss()
# 最適化の手法はSGDで。lossの減りに時間かかるけど、一旦はこれを使う。
optimizer = optim.SGD(model.parameters(), lr=CONFIG.learning_rate)

# 各エポックの合計loss値を格納する
losses = []
# 100ループ回してみる。（バッチ化とかGPU使ってないので結構時間かかる...）
for epoch in range(CONFIG.num_epochs):
    all_loss = 0
    for title, cat in zip(traindata["title"], traindata["category"]):
        # モデルが持ってる勾配の情報をリセット
        model.zero_grad()
        # 文章を単語IDの系列に変換（modelに食わせられる形に変換）
        inputs = sentence2index(title, word2index)
        # 順伝播の結果を受け取る
        out = model(inputs)
        # 正解カテゴリをテンソル化
        answer = datasets.category2tensor(cat, category2idx)
        # 正解とのlossを計算
        loss = loss_function(out, answer)
        # 勾配をセット
        loss.backward()
        # 逆伝播でパラメータ更新
        optimizer.step()
        # lossを集計
        all_loss += loss.item()
    losses.append(all_loss)
    print("epoch", epoch, "\t" , "loss", all_loss)
print("done.")

#Pytorchモデルの保存・読み込み
torch.save(model.state_dict(), CONFIG.NET_NAME)
#================================================================================
#  ■　二、予測精度確認
#
#================================================================================

# テストデータの母数計算
test_num = len(testdata)
# 正解の件数
a = 0
# 勾配自動計算OFF
with torch.no_grad():
    for title, category in zip(testdata["title"], testdata["category"]):
        # テストデータの予測
        inputs = sentence2index(title,word2index)
        out = model(inputs)

        # outの一番大きい要素を予測結果をする
        _, predict = torch.max(out, 1)  # @UndefinedVariable

        answer = datasets.category2tensor(category, category2idx)
        if predict == answer:
            a += 1
print("predict : ", a / test_num)
