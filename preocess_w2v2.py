#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 09:14:21 2018

@author: llq
"""
import pandas as pd
import numpy as np
from pyhanlp import HanLP
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from dnn_data3 import char_lower
import re

pat="[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8")

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

def extract_prob(pred):
    '''
    split the "query_prediction"
    '''
    pred = eval(pred)
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    pred_prob_lst=[]
    for i in range(10):
        if len(pred)<i+1:
            pred_prob_lst.append("pad")
        else:
            pred_prob_lst.append(pred[i][0].lower())
    return pred_prob_lst

def split_query_prediction(data):
    for i in range(10):
        data['text'+str(i)]=data.pred_prob_lst.apply(lambda x:str(x[i]))
    return data

stop_words=[]
def load_stopwords(path):
    with open(path,"r") as f:
        for lines in f.readlines():
            stop_words.append(lines.strip().replace('\n',''))

    
def get_text_vector(text, text2vec, word2vec):
    if text in text2vec:
        return  
    words=HanLP.segment(text)
    M = []
    for w in words:
        w = str(w).split('/')[0]
        if((w in stop_words) or (w==' ')):
            continue
        w=unicode(w, 'utf-8')
        if w in word2vec:
            M.append(word2vec[w])
        else:
            continue

    if not M:
        text2vec[text] = (0, [0.] * 300)
        return

    M = np.array(M)
    v = M.sum(axis=0)
    text2vec[text] = (1, v / np.sqrt((v ** 2).sum()))
    return 

def get_distance(q, t, distance_func):
    if q == "pad" or t == "pad":
        return -1
    flag_q, q_vector = text2vec[q]
    flag_t, t_vector = text2vec[t]

    if flag_q and flag_t:
        return distance_func(q_vector, t_vector)
    else:
        return -1

def get_dot(q,t):
    if q == "pad" or t == "pad":
        return -1
    flag_q, q_vector = text2vec[q]
    flag_t, t_vector = text2vec[t]

    if flag_q and flag_t:
        return np.dot(q_vector,t_vector)
    else:
        return -1

def get_pairwise_feature(data, q, t):
    for distance_func in [cosine]:
        data[q+"_"+t+"_"+distance_func.__name__+"_distance"] = data[[q, t]].apply(lambda x:get_distance(x[q], x[t], distance_func), axis=1)
        data[q+"_"+t+"_"+"_dot"] = data[[q, t]].apply(lambda x:get_dot(x[q], x[t]), axis=1)
        
    return data


train_data = read_file('./data/oppo_round1_train_20180929.txt').astype(str)
val_data = read_file('./data/oppo_round1_vali_20180929.txt').astype(str)
test_data = read_file('./data/oppo_round1_test_A_20180929.txt',True).astype(str)
train_data=pd.concat([train_data,val_data])
del val_data

#lower
train_data["prefix"] = train_data["prefix"].apply(char_lower)
train_data["title"] = train_data["title"].apply(char_lower)
test_data["prefix"] = test_data["prefix"].apply(char_lower)
test_data['title'] = test_data['title'].apply(char_lower)
 

#split the query_prediction
train_data['pred_prob_lst']=train_data['query_prediction'].apply(extract_prob)
train_data=split_query_prediction(train_data)
test_data['pred_prob_lst']=test_data['query_prediction'].apply(extract_prob)
test_data=split_query_prediction(test_data)



#embedding
print("Text2vec:embedding....")
#load stop_words
path="./data/中文停用词表.txt"
load_stopwords(path)

text2vec={}
# 加载外部词向量
vec_path='./data/sgns.weibo.word'
word2vec = KeyedVectors.load_word2vec_format(vec_path, binary=False)

items=["prefix", "title", "text0", "text1", "text2", "text3", "text4", "text5", "text6", "text7", "text8", "text9"]
for col in items:
    train_data[col].apply(lambda x: get_text_vector(x, text2vec, word2vec))
    test_data[col].apply(lambda x: get_text_vector(x, text2vec, word2vec))

#pairwise feature
print("pairwise feature")
for col in ["prefix", "text0", "text1", "text2", "text3", "text4", "text5", "text6", "text7", "text8", "text9"]:
    train_data = get_pairwise_feature(train_data, col, "title")
    test_data = get_pairwise_feature(test_data, col, "title")

y_train=train_data[['label']]
y_test=test_data[['label']]
train_data=train_data.drop(['prefix','query_prediction','title','tag','label','pred_prob_lst','text0','text1','text2','text3','text4',\
       'text5','text6','text7','text8','text9'],axis=1)
test_data=test_data.drop(['prefix','query_prediction','title','tag','label','pred_prob_lst','text0','text1','text2','text3','text4',\
       'text5','text6','text7','text8','text9'],axis=1)

# 保存处理后的数据集
data = dict(
    train_data=train_data,
    y_train=y_train,
)
np.savez('./data_word2vec_feature/train_vector_lower.npz', **data)

data = dict(
    test_data=test_data,
    y_test=y_test,
)
np.savez('./data_word2vec_feature/test_vector_lower.npz', **data)