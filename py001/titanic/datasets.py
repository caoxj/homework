'''
データの読み込み、補完など
Created on 2020/04/08

@author: sou
'''

import pandas as pd

from os.path import join
from sklearn.model_selection import train_test_split

#trainデータを分割
def load_split_data():
    #訓練データ読み込み
    x, t = get_train()
    print('x:',x.shape)
    print('t:',t.shape)
    
    #サンプルデータ＝「訓練データセット＋検証データセット」＋「テストデータセット」
    x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.3, random_state=0)
    print('x_train_val:',x_train_val.shape)
    print('x_test:',x_test.shape)
    
    #「訓練データセット」＋「検証データセット」
    x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0)
    print('x_train:',x_train.shape)
    print('x_val:',x_val.shape)
    
    #NumPy配列ndarray1に変換して出力
    return x_train.values, x_val.values, t_train.values, t_val.values, x_test.values, t_test.values

#csvファイルからデータ読み込み
def load_data(module_path, data_file_name):
    """Loads data from module_path/data/data_file_name.

    Parameters
    ----------
    module_path : string
        The module path.

    data_file_name : string
        Name of csv file to be loaded from
        module_path/data/data_file_name. For example 'wine_data.csv'.

    Returns
    -------
    data : Numpy array
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    """
    data = pd.read_csv(join(module_path, 'data', data_file_name))

    return data

#訓練検証用データ（結果あり）を読み込み
def get_train():
    train = load_data("..","titanic/train.csv")
    #(1) 欠損データを代理データに入れ替える
    train = fillna(train)
    
    #(2) 文字列カテゴリカルデータを数字へ変換  
    train = trans(train)
    
    #(3)家族の人数」用の変数を追加,不要な変数を削除
    train = improve(train)
    test = train["Survived"]
    train = train.drop("Survived", axis=1) 
    
    # それぞれデータ型を変換
    train = train.astype('float32')
    test = test.astype('int32')
    
    return train, test

#予測用データ（結果なし）読み込み
def get_test():
    test = load_data("..","titanic/test.csv")
    t_test = load_data("..","titanic/gender_submission.csv")
    t_test = t_test["Survived"]
    t_id = test["PassengerId"]
    
    #(1) 欠損データを代理データに入れ替える
    test = fillna(test)
    
    #(2) 文字列カテゴリカルデータを数字へ変換  
    test = trans(test)  
    
    #(3)家族の人数」用の変数を追加,不要な変数を削除
    test = improve(test) 
    
    # それぞれデータ型を変換
    test = test.astype('float32')
    t_test = t_test.astype('int32')
    
    return test.values, t_test.values, t_id.values


#欠損データを代理データに入れ替える
def fillna(pddata):
    pddata["Age"] = pddata["Age"].fillna(pddata["Age"].median())
    pddata["Embarked"] = pddata["Embarked"].fillna("S")
    return pddata
    
#文字列カテゴリカルデータを数字へ変換
def trans(pddata):
    pddata["Sex"][pddata["Sex"] == "male"] = 0
    pddata["Sex"][pddata["Sex"] == "female"] = 1
    
    pddata["Embarked"][pddata["Embarked"] == "S" ] = 0
    pddata["Embarked"][pddata["Embarked"] == "C" ] = 1
    pddata["Embarked"][pddata["Embarked"] == "Q"] = 2
    return pddata

#家族の人数」用の変数を追加します。そして不要な変数を削除します。
def improve(pddata):
    #家族の人数」用の変数を追加します
    pddata["FamilySize"] = pddata["SibSp"] + pddata["Parch"] + 1
    #不要な変数を削除します。
    df2 = pddata.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

    return df2
    