'''
ネットワークを訓練する
Created on 2020/04/08

@author: sou
'''

import chainer
import datasets
import numpy as np
import onnx_chainer

import chainer.links as L
import chainer.functions as F
from chainer import Sequential
from chainer import optimizers
import matplotlib.pyplot as plt
from chainer.optimizer_hooks import WeightDecay

import const as CONST

#Step 1 : データセットの準備
x_train, x_val, t_train, t_val, x_test, t_test = datasets.load_split_data()

#Step 2 : ネットワークを決める
# net としてインスタンス化
#ネットワークの定義
net = Sequential(
    L.Linear(CONST.n_input, CONST.n_hidden), F.relu,
    L.Linear(CONST.n_hidden, CONST.n_hidden), F.relu,
    L.Linear(CONST.n_hidden, CONST.n_output)
)

#Step 3 : 目的関数を決める
#交差エントロピー
#Step 4 : 最適化手法を選択する
#SGD:確率的勾配降下法 (SGD)
optimizer = chainer.optimizers.SGD(lr=0.01)
#MomentumSGD は SGD の改良版で、パラメータ更新の際に前回の勾配を使って更新方向がスムーズになる
#optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
optimizer.setup(net)

for param in net.params():
    if param.name != 'b':  # バイアス以外だったら
        param.update_rule.add_hook(WeightDecay(0.0001))  # 重み減衰を適用

#Step 5 : ネットワークを訓練する
iteration = 0

# ログの保存用
results_train = {
    'loss': [],
    'accuracy': []
}
results_valid = {
    'loss': [],
    'accuracy': []
}


for epoch in range(CONST.n_epoch):

    # データセット並べ替えた順番を取得
    order = np.random.permutation(range(len(x_train)))

    # 各バッチ毎の目的関数の出力と分類精度の保存用
    loss_list = []
    accuracy_list = []

    for i in range(0, len(order), CONST.n_batchsize):
        # バッチを準備
        index = order[i:i+CONST.n_batchsize]
        x_train_batch = x_train[index,:]
        t_train_batch = t_train[index]

        # 予測値を出力
        y_train_batch = net(x_train_batch)

        # 目的関数を適用し、分類精度を計算
        loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
        accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

        loss_list.append(loss_train_batch.array)
        accuracy_list.append(accuracy_train_batch.array)

        # 勾配のリセットと勾配の計算
        net.cleargrads()
        loss_train_batch.backward()

        # パラメータの更新
        optimizer.update()

        # カウントアップ
        iteration += 1

    # 訓練データに対する目的関数の出力と分類精度を集計
    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(accuracy_list)

    # 1エポック終えたら、検証データで評価
    # 検証データで予測値を出力
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y_val = net(x_val)

    # 目的関数を適用し、分類精度を計算
    loss_val = F.softmax_cross_entropy(y_val, t_val)
    accuracy_val = F.accuracy(y_val, t_val)

    # 結果の表示
    print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
        epoch, iteration, loss_train, loss_val.array))

    # ログを保存
    results_train['loss'] .append(loss_train)
    results_train['accuracy'] .append(accuracy_train)
    results_valid['loss'].append(loss_val.array)
    results_valid['accuracy'].append(accuracy_val.array)

##########################################################################
# 目的関数の出力 (loss)
plt.plot(results_train['loss'], label='loss train')  # label で凡例の設定
plt.plot(results_valid['loss'], label='loss valid')  # label で凡例の設定
plt.legend()  # 凡例の表示

# 分類精度 (accuracy)
plt.plot(results_train['accuracy'], label='accuracy train')  # label で凡例の設定
plt.plot(results_valid['accuracy'], label='accuracy valid')  # label で凡例の設定
plt.legend()  # 凡例の表示
###########################################################################

# テストデータで予測値を計算
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = net(x_test)
accuracy_test = F.accuracy(y_test, t_test)
print(accuracy_test.array)

#ネットワークの保存
chainer.serializers.save_npz(CONST.NET_NAME, net)

# ONNX形式に学習モデルを出力
onnx_model = onnx_chainer.export(net, x_test, filename=CONST.NET_NAME_ONNX)

############--訓練済みネットワークを用いた推論--################
#保存したネットワークを読み込みます
loaded_net = Sequential(
    L.Linear(CONST.n_input, CONST.n_hidden), F.relu,
    L.Linear(CONST.n_hidden, CONST.n_hidden), F.relu,
    L.Linear(CONST.n_hidden, CONST.n_output)
)
#パラメータを読み込ませ
chainer.serializers.load_npz(CONST.NET_NAME, loaded_net)
#推論を行い
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = loaded_net(x_test)
#テストデータの 0 番目のサンプルの予測結果を確認
for i in range(len(x_test)):
    y = np.argmax(y_test[i,:].array)
    print(y==t_test[i],"   ",y,"/",t_test[i],":",x_test[i])
    
