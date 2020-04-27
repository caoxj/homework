# Hyper Parameters 
# gray image 28*28
"""
input_size = 28*28
hidden_size = 500
# output(0~9) 
num_classes = 10
num_epochs = 8
batch_size = 200
learning_rate = 0.001
NET_NAME = "pytorch_mnist.net"
NET_NAME_ONNX = "pytorch_mnist.onnx"
"""

# 単語のベクトル次元数
EMBEDDING_DIM = 10
# 隠れ層の次元数
HIDDEN_DIM = 128

num_epochs = 100
learning_rate = 0.01

NET_NAME = "pytorch_sltm.net"
NET_NAME_ONNX = "pytorch_sltm.onnx"