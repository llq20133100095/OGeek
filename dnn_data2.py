#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:49:52 2018

@author: llq
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import sparse

class PrefixProcessing(object):
    @staticmethod
    def _is_in_title(item):
        prefix = item["prefix"]
        title = item["title"]

        if not isinstance(prefix, str):
            prefix = "null"

        if prefix in title:
            return 1
        return 0

    @staticmethod
    def _levenshtein_distance(item):
        str1 = item["prefix"]
        str2 = item["title"]

        if not isinstance(str1, str):
            str1 = "null"

        x_size = len(str1) + 1
        y_size = len(str2) + 1

        matrix = np.zeros((x_size, y_size), dtype=np.int_)

        for x in range(x_size):
            matrix[x, 0] = x

        for y in range(y_size):
            matrix[0, y] = y

        for x in range(1, x_size):
            for y in range(1, y_size):
                if str1[x - 1] == str2[y - 1]:
                    matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1)
                else:
                    matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1)

        return matrix[x_size - 1, y_size - 1]

    @staticmethod
    def _distince_rate(item):
        str1 = item["prefix"]
        str2 = item["title"]
        leven_distance = item["leven_distance"]

        if not isinstance(str1, str):
            str1 = "null"

        length = max(len(str1), len(str2))

        return float(leven_distance) / (length + 5.0)  # 平滑

    def get_prefix_df(self, df):
        prefix_df = pd.DataFrame()

        prefix_df[["prefix", "title"]] = df[["prefix", "title"]]
        prefix_df["is_in_title"] = prefix_df.apply(self._is_in_title, axis=1)
        prefix_df["leven_distance"] = prefix_df.apply(self._levenshtein_distance, axis=1)
        prefix_df["distance_rate"] = prefix_df.apply(self._distince_rate, axis=1)
        prefix_df=prefix_df.drop(['prefix','title'],axis=1)
        return prefix_df
    
def length_feature(data):
    '''
    对vector feature构造长度特征
    '''
    vec_feature = ['query_prediction']
    
    for co in vec_feature:
        value = []
        lis = list(data[co].values)
        for i in range(len(lis)):
            value.append(len(lis[i].split(',')))
        col_name = co+'_length'
        data[col_name]=value
    return data

def tag_group(train_data,val_data,test_data):
    """
    针对tag标签进行分割
    """
    data=pd.concat([train_data,val_data,test_data])
    items = ['prefix', 'title']
    target=['tag']
    for i in items:
        temp=data.groupby(i,as_index = False)[target[0]].agg({i+"_"+target[0]+"_nunique":'count'})
        train_data = pd.merge(train_data, temp, on=i, how='left')
        val_data = pd.merge(val_data, temp, on=i, how='left')
        test_data = pd.merge(test_data, temp, on=i, how='left')
    
#    temp=data.groupby(target[0],as_index = False)[items[0]].agg({target[0]+"_"+items[0]+"_nunique":'count'})
#    train_data = pd.merge(train_data, temp, on=target[0], how='left')
#    test_data = pd.merge(test_data, temp, on=target[0], how='left')
#
#    temp=data.groupby(target[0],as_index = False)[items[1]].agg({target[0]+"_"+items[1]+"_nunique":'count'})
#    train_data = pd.merge(train_data, temp, on=target[0], how='left')
#    test_data = pd.merge(test_data, temp, on=target[0], how='left')
    
    return train_data,val_data,test_data
    
def char_lower(char):
    return char.lower()
    
def get_data():
    
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

    """
    baseline feature
    """
    train_data = read_file('./data/oppo_round1_train_20180929.txt').astype(str)
    
    val_data = read_file('./data/oppo_round1_vali_20180929.txt').astype(str)
    test_data = read_file('./data/oppo_round1_test_A_20180929.txt',True).astype(str)
    
    train_data = train_data[train_data['label'] != '音乐' ]
    test_data['label'] = -1
    
#    train_data = pd.concat([train_data,val_data])
#    train_data.reset_index(drop=True, inplace=True)
    train_data['label'] = train_data['label'].apply(lambda x: int(x))
    val_data['label'] = val_data['label'].apply(lambda x: int(x))
    test_data['label'] = test_data['label'].apply(lambda x: int(x))
    items = ['prefix', 'title', 'tag']
    
    #lower
    train_data["prefix"] = train_data["prefix"].apply(char_lower)
    train_data["title"] = train_data["title"].apply(char_lower)
    val_data['prefix'] = val_data['prefix'].apply(char_lower)
    val_data['title'] = val_data['title'].apply(char_lower)
    test_data["prefix"] = test_data["prefix"].apply(char_lower)
    test_data['title'] = test_data['title'].apply(char_lower)
    
    #feature:is_in_title  leven_distance  distance_rate
    prefix_processing = PrefixProcessing()
    prefix_train_data = prefix_processing.get_prefix_df(train_data)
    prefix_val_data = prefix_processing.get_prefix_df(val_data)
    prefix_test_data = prefix_processing.get_prefix_df(test_data)
    
    #构造长度特征
    train_data=length_feature(train_data)
    val_data=length_feature(val_data)
    test_data=length_feature(test_data)
    
    #单个特征
    for item in items:
        #通过item对label进行分割
        temp = train_data.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
        #统计item的点击次数
        temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count']+0.001)
        train_data = pd.merge(train_data, temp, on=item, how='left')
        val_data = pd.merge(val_data, temp, on=item, how='left')
        test_data = pd.merge(test_data, temp, on=item, how='left')
    #交叉特征
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            item_g = [items[i], items[j]]
            temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
            temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+0.001)
            train_data = pd.merge(train_data, temp, on=item_g, how='left')
            val_data = pd.merge(val_data, temp, on=item_g, how='left')
            test_data = pd.merge(test_data, temp, on=item_g, how='left')
    #3 feature across
    item_g = [items[0], items[1], items[2]]
    temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
    temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
    train_data = pd.merge(train_data, temp, on=item_g, how='left')
    val_data = pd.merge(val_data, temp, on=item_g, how='left')
    test_data = pd.merge(test_data, temp, on=item_g, how='left')
    
    #针对tag标签进行分割
    train_data,val_data,test_data=tag_group(train_data,val_data,test_data)
    
    train_data=pd.concat([train_data,prefix_train_data],axis=1)
    val_data=pd.concat([val_data,prefix_val_data],axis=1)
    test_data=pd.concat([test_data,prefix_test_data],axis=1)
    
    #delete the column 
    train_data_ = train_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
    val_data_ = val_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
    test_data_ = test_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
    #    train_data_=train_data_[['prefix_ctr','title_ctr','tag_ctr','prefix_title_ctr','prefix_tag_ctr','title_tag_ctr','label']]
    #    test_data_=test_data_[['prefix_ctr','title_ctr','tag_ctr','prefix_title_ctr','prefix_tag_ctr','title_tag_ctr','label']]
    #    train_data_=train_data_[['tag_ctr','label']]
    #    test_data_=test_data_[['tag_ctr','label']]
    
    #    X = np.array(train_data_.drop(['label'], axis = 1))
    #    y = np.array(train_data_['label'])
    #    X_test_ = np.array(test_data_.drop(['label'], axis = 1))
    
    #    return X,y,X_test_,test_data_
    return train_data_,val_data_,test_data_
    
    
train_data_,val_data_,test_data_=get_data()

y_train=train_data_[['label']]
y_val=val_data_[['label']]
y_test=test_data_[['label']]
train_data_=train_data_.drop(['label'],axis=1)
val_data_=val_data_.drop(['label'],axis=1)
test_data_=test_data_.drop(['label'],axis=1)

# 保存处理后的数据集
data = dict(
    train_data=train_data_,
    y_train=y_train,
)
np.savez('./data_word2vec_feature/train_dnn_data2_feature.npz', **data)

# 保存处理后的数据集
data = dict(
    val_data=val_data_,
    y_val=y_val,
)
np.savez('./data_word2vec_feature/val_dnn_data2_feature.npz', **data)

data = dict(
    test_data=test_data_,
    y_test=y_test,
)
np.savez('./data_word2vec_feature/test_dnn_data2_feature.npz', **data)