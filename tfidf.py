#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:43:37 2018

@author: llq
"""
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
X_test = ['没有 你 的 地方 都是 他乡','没有 你 的 旅行 都是 流浪']
stopword = [u'都是'] #自定义一个停用词表，如果不指定停用词表，则默认将所有单个汉字视为停用词；
#但可以设token_pattern=r"(?u)\b\w+\b"，即不考虑停用词

tfidf=TfidfVectorizer()
weight=tfidf.fit_transform(X_test).toarray()
word=tfidf.get_feature_names()

print 'vocabulary list:\n'
for key,value in tfidf.vocabulary_.items():
    print key,value
    
train_data = pd.read_table('./data/oppo_round1_train_20180929.txt', 
        names= ['prefix','query_prediction','title','tag','label'], header= None).astype(str)
val_data = pd.read_table('./data/oppo_round1_vali_20180929.txt', 
        names = ['prefix','query_prediction','title','tag','label'], header = None).astype(str)
test_data = pd.read_table('./data/oppo_round1_test_A_20180929.txt',
        names = ['prefix','query_prediction','title','tag'],header = None).astype(str)

train_data = train_data[train_data['label'] != '音乐' ]
test_data['label'] = -1

train_data = pd.concat([train_data,val_data])
train_data['label'] = train_data['label'].apply(lambda x: int(x))
test_data['label'] = test_data['label'].apply(lambda x: int(x))
items = ['prefix', 'title', 'tag']

data=pd.concat([train_data,test_data])
data_query=data['query_prediction']
data_query_list=[]
i=0
for query in data_query:
    query=query.replace("{","").replace("}","").replace('"','')
    query=query.split(",")
    query_str=""
    for word_fep in query:
        word=word_fep.split(":")[0].replace(" ","")
        seg_list=jieba.cut(word)  #cut word
        query_str=query_str+" ".join(seg_list)
    data_query_list.append(query_str)
    i+=1
    print i
    
tfidf=TfidfVectorizer()
weight=tfidf.fit_transform(data_query_list).toarray()
word=tfidf.get_feature_names()



