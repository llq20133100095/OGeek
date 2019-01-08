#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:01:55 2018

@author: llq
"""
import pandas as pd
import numpy as np
from pyhanlp import HanLP
from gensim.models import Word2Vec, KeyedVectors
from text_sim.hownet import similar_word
# 词向量长度
DIM = 300

def read_file(path,is_test=False):
    fp = open(path)
    dataset = []
    for line in fp.readlines():
        line = line.strip().split('\t')
        if is_test:
            line.append(-1)
        dataset.append(line)
    data = pd.DataFrame(dataset)
    data.columns = ['prefix', 'query_prediction', 'title', 'tag', 'label']
    return data

def tag_onehot(train_data, test_data):
    """
        tag 向量化
    Args:
        train_data:
        val_data:
        test_data:
    Returns:
    """
    tag = pd.concat([train_data['tag'], test_data['tag']])
    onehot = pd.get_dummies(tag)
    onehot.reset_index(drop=True, inplace=True)
    train_tag = onehot.iloc[:train_data.shape[0]]
    test_tag = onehot.iloc[-test_data.shape[0]:]
    test_tag.reset_index(drop=True, inplace=True)
    train_data=train_data.drop(['query_prediction'],axis=1)
    test_data=test_data.drop(['query_prediction'],axis=1)
    train_data = pd.concat([train_data['label'], train_data[train_data.columns[:-2]], train_tag], axis=1)    
    test_data = pd.concat([test_data['label'], test_data[test_data.columns[:-2]], test_tag], axis=1)
    print(train_data.columns)
    return train_data, test_data

def generate_feature(data, word2vec):
    """
        生成特征向量
    Args:
        data: 数据集
        word2vec:
    Returns:
    """
    features = []
    end = len(data.columns)
    for idx, row in data.iterrows():
        prefix_vec = np.zeros(DIM)
        title_vec = np.zeros(DIM)
        tag_vec = row.iloc[3: end]
        count = 0
        try:
            for word in HanLP.segment(row['prefix']):
                word = str(word).split('/')[0]
                word=unicode(word, 'utf-8')
                try:
                    prefix_vec += word2vec[word]
                    count += 1
                except:
                    print('word %s not in vocab' % word)
            if count > 0:
                prefix_vec = np.true_divide(prefix_vec, count)
            count = 0
            for word in HanLP.segment(row['title']):
                word = str(word).split('/')[0]
                word=unicode(word, 'utf-8')
                try:
                    title_vec += word2vec[word]
                    count += 1
                except:
                    print('word %s not in vocab' % word)
            if count > 0:
                title_vec = np.true_divide(title_vec, count)
        except Exception as e:
            print(e)
        feature = np.concatenate((prefix_vec, title_vec, tag_vec))
        features.append(feature)
        
    return pd.DataFrame(features)

"""
baseline feature
"""
train_data = read_file('./data/oppo_round1_train_20180929.txt').astype(str)

val_data = read_file('./data/oppo_round1_vali_20180929.txt').astype(str)
test_data = read_file('./data/oppo_round1_test_A_20180929.txt',True).astype(str)

train_data = train_data[train_data['label'] != '音乐' ]
test_data['label'] = -1

train_data = pd.concat([train_data,val_data])
train_data.reset_index(drop=True, inplace=True)
train_data['label'] = train_data['label'].apply(lambda x: int(x))
test_data['label'] = test_data['label'].apply(lambda x: int(x))
items = ['prefix', 'title', 'tag']


#tag to one-hot
train_data, test_data=tag_onehot(train_data,test_data)

# 加载外部词向量
vec_path='./data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
word2vec = KeyedVectors.load_word2vec_format(vec_path, binary=False)

# 生成特征向量
X_train=generate_feature(train_data[:50000], word2vec)
for i in range(10):
#    X_train = generate_feature(train_data[:200000], word2vec)
    X_a = generate_feature(train_data[50000+i*200000:50000+(i+1)*200000], word2vec)
    X_train=pd.concat([X_train,X_a])
    
X_test = generate_feature(test_data, word2vec)
#y_train=train_data[['label']]
y_test=test_data[['label']]
#
## 保存处理后的数据集
#data = dict(
#    X_train=X_train,
#    y_train=y_train,
#)
#np.savez('./data/train_vector.npz', **data)

data = dict(
    X_test=X_test,
    y_test=y_test,
)
np.savez('./data/test_vector.npz', **data)
