'''
Created on 2020/04/14

@author: sou
'''
import torch.nn as nn
import torch.optim
import torch.onnx
from torch.autograd import Variable
 
import model as MODEL
import config as CONFIG
import datasets
import kaggleDataset

#モデルインスタンスの作成
net = MODEL.Net(CONFIG.input_size, CONFIG.hidden_size, CONFIG.num_classes)
#GPUで実行
#net.cuda()

#損失関数とオプティマイザの設定
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG.learning_rate) 

#pytorchのMNISTから取得
#train_dataset = datasets.load_data_train()
#train_loader = datasets.get_minibatch(train_dataset)

#kaggleのcsvファイルから取得
kdata = kaggleDataset.KaggleDataset("../data/digit-recognizer/train.csv")
train_loader = datasets.get_minibatch(kdata)

#plt.imshow(train_dataset[0][0][0].numpy(), cmap='gray')

#ネットワークの訓練を行う関数を定義
def fit(net, train_loader, num_epochs=CONFIG.num_epochs, t= 200):
    # モデルのトレーニング
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # pytorchのtensorをVariableに変換
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
 
            # フォワードプロパゲーション/ バックプロパゲーション/ 最適化
            optimizer.zero_grad()  # 勾配を0初期化
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # t回毎にログを表示
            if (i+1) % t == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                        %(epoch+1, num_epochs, i+1, len(train_loader)//CONFIG.batch_size, loss.data.item()))
    return net


#トレーニング 
net = fit(net, train_loader)

#Pytorchモデルの保存・読み込み
torch.save(net.state_dict(), CONFIG.NET_NAME)

# ONNX形式に学習モデルを出力
#net.train(False)
#dummy_param = Variable(torch.randn(1, 3, 224, 224), requires_grad=True)
#torch_out = torch.onnx._export(net,dummy_param,CONFIG.NET_NAME_ONNX,export_params=True)

