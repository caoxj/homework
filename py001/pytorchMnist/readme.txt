参考：https://www.sejuku.net/blog/64175
kaggle:https://www.kaggle.com/c/digit-recognizer

pytorchサンプル
MNIST手書き文字認識


MNISTデータについて
- root_dir
  - processed
    - test.pt                  # テスト用データがPyTorchデータ向けに生成されている。
    - training.pt              # 画像用データがPyTorchデータ向けに生成されている。
  - raw
    - t10k-images-idx3-ubyte   # テスト用画像
    - t10k-labels-idx1-ubyte   # テスト用ラベル
    - train-images-idx3-ubyte  # 学習用データ画像
    - train-labels-idx1-ubyte  # 学習用データラベル
    
kaggleデータ
- digit-recognizer
	- sample_submission.csv
	- test.csv
	- train.csv