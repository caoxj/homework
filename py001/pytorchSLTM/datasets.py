'''
Created on 2020/04/14

@author: sou
'''

import re
import torch
import MeCab

import os
import glob
import pandas as pd
import linecache
from sklearn.utils import shuffle

import pytorchSLTM.config as CONFIG

def load(rpath = "../data/sltm_livedoor/text/"):
    # カテゴリを配列で取得
    categories = [name for name in os.listdir(rpath) if os.path.isdir(rpath + name)]
    print(categories)
    # ['movie-enter', 'it-life-hack', 'kaden-channel', 'topic-news', 'livedoor-homme', 'peachy', 'sports-watch', 'dokujo-tsushin', 'smax']

    datasets = pd.DataFrame(columns=["title", "category"])
    for cat in categories:
        path = rpath + cat + "/*.txt"
        files = glob.glob(path)
        for text_name in files:
            title = linecache.getline(text_name, 3)
            s = pd.Series([title, cat], index=datasets.columns)
            datasets = datasets.append(s, ignore_index=True)

    # データフレームシャッフル
    #引数fracで抽出する行・列の割合を指定できる。1だと100%
    datasets = datasets.sample(frac=1).reset_index(drop=True)
    return categories,datasets

def load_dotcom(rpath = "../data/sltm_dotcom/title_cat.csv"):

    datasets = pd.read_csv(rpath, header=None, names=["title", "category"])
    #1:nを1:1に重複を削除
    datasets=datasets.drop_duplicates(subset=["title"],keep='first')
    # print(datasets)
    cats=datasets.drop_duplicates(subset=["category"],keep='first')
    # カテゴリを配列で取得
    categories = cats["category"].to_numpy()
    print(categories)
    # print(categories["category"].to_numpy())

    # データフレームシャッフル
    #引数fracで抽出する行・列の割合を指定できる。1だと100%
    datasets = datasets.sample(frac=1).reset_index(drop=True)
    return categories,datasets

#datasets.head()
#title  category
#0  兼用アンテナ搭載の「Viewer Dock」が同梱！シャープのドコモ向けハイエンドエンタメ系... smax
#1  女は“愛嬌”、男も“愛嬌”-人事担当者がこっそり教える採用ウラ話 vol.6\n  livedoor-homme
#2  社会貢献×ファッションがカッコイイ、今年の春旋風を巻き起こしたMODE for Charit...  peachy
#3  今でも、後でも読めるニュースがここにある！スマホでもタブレットでも読みやすいITニュース活用...   it-life-hack
#4  被災地の缶詰を途上国に…「正気じゃない。人殺しだ!!」\n topic-news
#==============================================================

#わかち書きをする
tagger = MeCab.Tagger("-O wakati")

def make_wakati(sentence):
    # MeCabで分かち書き
    sentence = tagger.parse(sentence)
    # 半角全角英数字除去
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # 記号もろもろ除去
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    # スペースで区切って形態素の配列へ
    wakati = sentence.split(" ")
    # 空の要素は削除
    wakati = list(filter(("").__ne__, wakati))
    return wakati

# テスト
#test = "【人工知能】は「人間」の仕事を奪った"
#print(make_wakati(test))
# ['人工', '知能', 'は', '人間', 'の', '仕事', 'を', '奪っ', 'た']

# 単語ID辞書を作成する
def word2index(titels):
    word2index = {}
    # 系列を揃えるためのパディング文字列<pad>を追加
    # パディング文字列のIDは0とする
    word2index.update({"<pad>":0})

    for title in titels:
        wakati = make_wakati(title)
        for word in wakati:
            if word in word2index: continue
            word2index[word] = len(word2index)
    return word2index
    print("vocab size : ", len(word2index))
# vocab size :  13229

# 文章を単語IDの系列データに変換
# PyTorchのLSTMのインプットになるデータなので、もちろんtensor型で
def sentence2index(sentence, word2idx):
    wakati = make_wakati(sentence)
    return torch.tensor([word2idx[w] for w in wakati], dtype=torch.long)  # @UndefinedVariable

# テスト
#test = "例のあのメニューも！ニコニコ超会議のフードコートメニュー14種類紹介（前半）"
#print(sentence2index(test))
# tensor([11320,     3,   449,  5483,    26,  3096,  1493,  1368,     3, 11371, 7835,   174,  8280])

def category2index(categories):
    category2index = {}
    for cat in categories:
        if cat in category2index: continue
        category2index[cat] = len(category2index)
    print(category2index)
    return category2index
#{'movie-enter': 0, 'it-life-hack': 1, 'kaden-channel': 2, 'topic-news': 3, 'livedoor-homme': 4, 'peachy': 5, 'sports-watch': 6, 'dokujo-tsushin': 7, 'smax': 8}

def category2tensor(cat, category2idx):
    return torch.tensor([category2idx[cat]], dtype=torch.long)  # @UndefinedVariable

#print(category2tensor("it-life-hack"))
# tensor([1])


# データをバッチでまとめるための関数
def train2batch(title, category, batch_size=CONFIG.batch_size):
    title_batch = []
    category_batch = []
    title_shuffle, category_shuffle = shuffle(title, category)
    for i in range(0, len(title), batch_size):
        title_batch.append(title_shuffle[i:i+batch_size])
    category_batch.append(category_shuffle[i:i+batch_size])
    return title_batch, category_batch