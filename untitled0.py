#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:06:03 2018

@author: llq
"""

import pandas as pd
import numpy as np


def bbb(a,b):
    c = {}
    for i,j in eval(a).items():
        c[i.lower()] = j

    if b.lower() in c:
        if float(c[b.lower()])>0.1:
            return 1
        else:
            return 0
    else:
        return 0

train_data = pd.read_table('./data/oppo_round1_train_20180929.txt',names=['prefix','query_prediction','title','tag','label'],header=None).astype(str)
val_data = pd.read_table('./data/oppo_round1_vali_20180929.txt',names=['prefix','query_prediction','title','tag','label'],header=None).astype(str)
test_data = pd.read_table('./data/oppo_round1_test_A_20180929.txt',names=['prefix','query_prediction','title','tag','label'],header=None).astype(str)
train_data = train_data[train_data['label']!='音乐']
train_data = pd.concat([train_data,val_data])
train_data['label'] = train_data['label'].apply(lambda x:int(x))
val_data['label'] = val_data['label'].apply(lambda x:int(x))
train_data['prefix_num'] = train_data['prefix'].apply(lambda x:len(x))
train_data['title_num'] = train_data['title'].apply(lambda x:len(x))
test_data['prefix_num'] = test_data['prefix'].apply(lambda x:len(x))
test_data['title_num'] = test_data['title'].apply(lambda x:len(x))

test_data['in_query_big'] = test_data.apply(lambda x:bbb(x['query_prediction'],x['title']),axis=1)


temp1 = train_data.groupby(['prefix','title','tag'],as_index=False)['label'].agg({'click1':'sum','count1':'count','ctr1':'mean'})
temp2 = train_data.groupby(['prefix','title'],as_index=False)['label'].agg({'click2':'sum','count2':'count','ctr2':'mean'})
temp3 = train_data.groupby(['prefix','tag'],as_index=False)['label'].agg({'click3':'sum','count3':'count','ctr3':'mean'})
temp4 = train_data.groupby(['title','tag'],as_index=False)['label'].agg({'click4':'sum','count4':'count','ctr4':'mean'})
temp5 = train_data.groupby('prefix',as_index=False)['label'].agg({'click5':'sum','count5':'count','ctr5':'mean'})
temp6 = train_data.groupby('title',as_index=False)['label'].agg({'click6':'sum','count6':'count','ctr6':'mean'})
temp7 = train_data.groupby('tag',as_index=False)['label'].agg({'click7':'sum','count7':'count','ctr7':'mean'})
temp8 = train_data.groupby(['prefix_num','title_num','tag'],as_index=False)['label'].agg({'click8':'sum','count8':'count','ctr8':'mean'})




a = pd.merge(test_data,temp1,on=['prefix','title','tag'],how='left')
b = pd.merge(test_data,temp2,on=['prefix','title'],how='left')
c = pd.merge(test_data,temp3,on=['prefix','tag'],how='left')
d = pd.merge(test_data,temp4,on=['title','tag'],how='left')
e = pd.merge(test_data,temp5,on='prefix',how='left')
f = pd.merge(test_data,temp6,on='title',how='left')
g = pd.merge(test_data,temp7,on='tag',how='left')
h = pd.merge(test_data,temp8,on=['prefix_num','title_num','tag'],how='left')


a['ctr2'] = b['ctr2']
a['ctr3'] = c['ctr3']
a['ctr4'] = d['ctr4']
a['ctr5'] = e['ctr5']
a['ctr6'] = f['ctr6']
a['ctr7'] = g['ctr7']
a['ctr8'] = h['ctr8']

a['count2'] = b['count2']
a['count3'] = c['count3']
a['count4'] = d['count4']
a['count5'] = e['count5']
a['count6'] = f['count6']
a['count7'] = g['count7']

a = a.fillna(0)
a['label2'] = 0

num1 = np.where((a['ctr1']>0.4),1,0)
num4 = np.where((a['ctr1']<=0.4)&(a['ctr1']>=0.2)&(a['ctr4']>0.55),1,0)
num7 = np.where((a['count1']==0)&(a['count4']==0)&(a['in_query_big']==0)&(a['ctr8']>0.5),1,0)
num8 = np.where((a['count1']==0)&(a['count4']==0)&(a['in_query_big']==1),1,0)
num9 = np.where((a['ctr4']>0.4)&(a['count1']==0),1,0)
num10 = np.where((a['count1']==0)&(a['ctr4']<=0.4)&(a['ctr4']>=0.2)&(a['in_query_big']==1),1,0)

numall = np.array(num1)+np.array(num4)+np.array(num8)+np.array(num7)+np.array(num9)+np.array(num10)

a['label3'] = numall
a['label3'].to_csv('./submit/result.csv',index=False)
