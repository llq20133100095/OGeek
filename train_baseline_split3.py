#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:03:05 2018

@author: llq
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest,chi2,VarianceThreshold
from sklearn.metrics import f1_score
from dnn_data3 import char_lower
from dnn_data import get_data
from smote import smote
from dnn import DNN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

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
   

def split():
    """
    split the val_data and test_data in hasdata and nodata.
    """
    train_data = read_file('./data/oppo_round1_train_20180929.txt').astype(str)
    val_data = read_file('./data/oppo_round1_vali_20180929.txt').astype(str)
    test_data = read_file('./data/oppo_round1_test_A_20180929.txt',True).astype(str)
    
    train_data = train_data[train_data['label'] != '音乐' ]
    test_data['label'] = -1
    
    train_data['label'] = train_data['label'].apply(lambda x: int(x))
    val_data['label'] = val_data['label'].apply(lambda x: int(x))
    test_data['label'] = test_data['label'].apply(lambda x: int(x))
    
    #lower
    train_data["prefix"] = train_data["prefix"].apply(char_lower)
    train_data["title"] = train_data["title"].apply(char_lower)
    val_data['prefix'] = val_data['prefix'].apply(char_lower)
    val_data['title'] = val_data['title'].apply(char_lower)
    test_data["prefix"] = test_data["prefix"].apply(char_lower)
    test_data['title'] = test_data['title'].apply(char_lower)
    
    #split: has data in train_data or no data in train_data
    items=['prefix', 'title', 'tag']
   
    train_data=train_data.drop_duplicates(items)
    val_merge=pd.merge(val_data,train_data,on=items,how='left')
    test_merge=pd.merge(test_data,train_data,on=items,how='left')
    
    val_hasdata=val_merge[pd.isnull(val_merge['label_y'])==False]
    test_hasdata=test_merge[pd.isnull(test_merge['label_y'])==False]
    
    val_notdata=val_merge[pd.isnull(val_merge['label_y'])]
    test_notdata=test_merge[pd.isnull(test_merge['label_y'])]
    
    val_hasdata_index=list(val_hasdata.index)
    test_hasdata_index=list(test_hasdata.index)
    
    val_notdata_index=list(val_notdata.index)
    test_notdata_index=list(test_notdata.index)
    
    return val_hasdata_index,test_hasdata_index,val_notdata_index,test_notdata_index,val_hasdata,val_notdata

def lgbm(X,y,X_val_,y_val,X_test_,val_save,test_save,feature_names):    
    print('train beginning')

    print('================================')
    print(X.shape)
    print(y.shape)
    print(X_val_.shape)
    print('================================')
    
    
    xx_logloss = []
    xx_val=[]
    xx_submit = []
    N = 5
    skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
    
    params = {
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "max_bin": 600,
        "subsample_for_bin": 20000,
        "objective": 'binary',
        "min_split_gain": 0,
        "min_child_weight": 0.001,
        "min_child_samples": 20,
        "subsample_freq": 1,
        "reg_alpha": 3,
        "reg_lambda": 5,
        "seed": 2018,
        "verbose": 1,
#        'feature_fraction': 0.9,
#        'bagging_fraction': 0.8,
        'drop_rate':0.5,
    }
    
    best_f1_score=0
    for k, (train_in, test_in) in enumerate(skf.split(X, y)):
        print('train _K_ flod', k)
        X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]
    #    X_train, X_test, y_train, y_test=X,X_val_,y,y_val
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=5000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=100,
                        verbose_eval=50,
                        )
    #    #feature importance
    #    feature_imp=pd.DataFrame({'column': feature_names,'importance': gbm.feature_importance()})
        
        val_test=gbm.predict(X_test, num_iteration=gbm.best_iteration)
        f1_sco=f1_score(y_test, np.where(val_test>0.38, 1,0))
        
        print("f1 in has_data:%f" % (f1_sco))
        if(best_f1_score<f1_sco):
            best_f1_score=f1_sco
            
        xx_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
        
        #val
        val_pre=gbm.predict(X_val_, num_iteration=gbm.best_iteration)
        xx_val.append(val_pre)
        
        #test
        test_pre=gbm.predict(X_test_, num_iteration=gbm.best_iteration)
        xx_submit.append(test_pre)
    
    print(best_f1_score)
    print('train_logloss:', np.mean(xx_logloss))
    
    s = 0
    for i in xx_val:
        s = s + i
    s=list(s/N)
    val_save['label']=s
    val_save['label']=np.where(val_save[['label']]>0.38,1,0)

    s = 0
    for i in xx_submit:
        s = s + i
        s=list(s/N)
    test_save['label']=np.where(test_pre>0.38, 1,0)
    print('test_logloss:', np.mean(test_save.label))
    return feature_names,val_save,test_save

def lgbm2(X,y,X_val_,y_val,X_test_,val_save,test_save,feature_names):    
    print('train beginning')

    print('================================')
    print(X.shape)
    print(y.shape)
    print(X_val_.shape)
    print('================================')
    
    
    xx_logloss = []
    xx_submit = []
#    N = 5
#    skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
    
    params = {
        "num_leaves": [x for x in range(50,150,20)],
#        "max_depth": [x for x in range(1,10,1)],
        "learning_rate": np.linspace(0.05,0.25,5),
        "n_estimators": [x for x in range(100,500,50)],
    }
    
    best_f1_score=0
    X_train, X_test, y_train, y_test=X,X_val_,y,y_val

    gbm=lgb.LGBMClassifier(
        boosting_type="gbdt",
        num_leaves=50, 
        max_depth=-1, 
        learning_rate=0.05, 
        n_estimators=100, 
        subsample_for_bin=200000, 
        objective='binary', 
#        class_weight=None,
        is_unbalance=True,
        min_split_gain=0.0, 
        min_child_weight=0.001, 
        min_child_samples=20, 
        subsample=1.0, 
        subsample_freq=1, 
        colsample_bytree=1.0, 
        reg_alpha=3.0, 
        reg_lambda=5.0, 
        random_state=None, 
        n_jobs=-1, 
        silent=True,
        seed=2018,
        verbose=1,
        early_stopping_rounds=50)
    lgb_eval=(X_test, y_test)
    gbm.fit(X_train,y_train,eval_set=lgb_eval,verbose=50)

#    grid = GridSearchCV(gbm, params, cv=5, scoring="f1")
#    grid.fit(X_train,y_train)
#    print grid.best_score_    #查看最佳分数(此处为f1_score)
#    print("params:")
#    print grid.best_params_   #查看最佳参数
    
    
#    #feature importance
#    feature_imp=pd.DataFrame({'column': feature_names,'importance': gbm.feature_importance()})
    
    val_test=gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    f1_sco=f1_score(y_test, np.where(val_test>0.38, 1,0))
    val_save['label']=np.where(val_test>0.38, 1,0)
    
    print("f1 in has_data:%f" % (f1_sco))
    if(best_f1_score<f1_sco):
        best_f1_score=f1_sco
        
    xx_logloss.append(gbm.best_score_['valid_0']['binary_logloss'])
    test_pre=gbm.predict(X_test_, num_iteration=gbm.best_iteration_)
    xx_submit.append(test_pre)
    
        
    print(best_f1_score)
    print('train_logloss:', np.mean(xx_logloss))
    
    s = 0
    for i in xx_submit:
        s = s + i
    
#    test_data_['label'] = list(s / 1)
#    test_data_['label'] = test_data_['label'].apply(lambda x: round(x))
#    test_data_['label']=test_data_['label'].apply(lambda x:int(x))
#    test_data_['label'].to_csv('./submit/split_data/result_baseline.csv',index = True)
    
    test_save['label']=np.where(test_pre>0.38, 1,0)
    print('test_logloss:', np.mean(test_save.label))
    
    return feature_names,val_save,test_save

def gbdt(X,y,X_val_,y_val,X_test_,val_save,test_save,feature_names):    
    print('train beginning')

    print('================================')
    print(X.shape)
    print(y.shape)
    print(X_val_.shape)
    print('================================')
    
    
    xx_logloss = []
    xx_submit = []
    
    best_f1_score=0
    X_train, X_test, y_train, y_test=X,X_val_,y,y_val
    gbdt=GradientBoostingClassifier(
      learning_rate=0.1,
      n_estimators=100,
      subsample=0.8,
      min_samples_split=2,
      min_samples_leaf=1,
      max_depth=8,
      init=None,
      random_state=2018,
      max_features='sqrt',
      verbose=1,
      max_leaf_nodes=None,
      warm_start=False,
    )

    gbdt.fit(X_train,y_train)
    val_test=gbdt.predict_proba(X_test)[:,1]
    f1_sco=f1_score(y_test, np.where(val_test>0.38, 1,0))
    val_save['label']=np.where(val_test>0.38, 1,0)
    print("f1 in has_data:%f" % (f1_sco))
    if(best_f1_score<f1_sco):
        best_f1_score=f1_sco
        
    xx_logloss.append(gbdt.loss)
    test_pre=gbdt.predict_proba(X_test_)[:,1]
    xx_submit.append(test_pre)
    
        
    print(best_f1_score)
    print('train_logloss:', np.mean(xx_logloss))
    
    s = 0
    for i in xx_submit:
        s = s + i
        
    test_save['label']=np.where(test_pre>0.38, 1,0)
    print('test_logloss:', np.mean(test_save.label))
    return feature_names,val_save,test_save


def select_kbest(X,X_val_,X_test_,K):
    #选择K个最好的特征，返回选择特征后的数据
    skb = SelectKBest(chi2, k=K)
    X_val_=skb.fit_transform(X_val_, y_val)
    X=X[:,skb.get_support(True)]
    X_test_=X_test_[:,skb.get_support(True)]
    return X,X_val_,X_test_
    

def merge_val_test_data(val_save,test_save,val_nodata_save,test_nodata_save):
    
    val_save=pd.concat([val_save,val_nodata_save],axis=0)
    test_save=pd.concat([test_save,test_nodata_save],axis=0)
    
    #set and sort index
    val_save.set_index(['data_index'],inplace=True)
    val_save.sort_index(inplace=True)
    test_save.set_index(['data_index'],inplace=True)
    test_save.sort_index(inplace=True)
    
    #f1
    _,y,_,_=get_data()
    val_f1=f1_score(y[-50000:],val_save[['label']])
    print("finally val_f1:%f" %(val_f1))
    
    return val_save,test_save

if __name__=='__main__':
    #hasdata and notdata in val_data and test_data
    val_hasdata_index,test_hasdata_index,val_notdata_index,test_notdata_index,val_hasdata,val_notdata=split()
    
    """
    1.has data
    """
    #statis feature
    X=np.load('./data_word2vec_feature/train_dnn_data3_feature.npz')
    X_val_=np.load('./data_word2vec_feature/val_dnn_data3_feature.npz')
    X_test_=np.load('./data_word2vec_feature/test_dnn_data3_feature.npz')
    
    y=np.reshape(np.array(X['y_train']),(-1,))
    y_val=np.reshape(np.array(X_val_['y_val']),(-1,))
    y_test=np.reshape(np.array(X_test_['y_test']),(-1,))
    X=np.array(X['train_data'])
    X_val_=np.array(X_val_['val_data'])
    X_test_=np.array(X_test_['test_data'])
    
    #get the "has_data"
    y_val=y_val[val_hasdata_index]
    y_test=y_test[test_hasdata_index]
    X_val_=X_val_[val_hasdata_index]
    X_test_=X_test_[test_hasdata_index]    
    
    
#    X=X[:,[20,23,24,25]]
#    X_val_=X_val_[:,[20,23,24,25]]
#    X_test_=X_test_[:,[20,23,24,25]]
    
#    #选择K个最好的特征，返回选择特征后的数据
#    K=35
#    skb = SelectKBest(chi2, k=K)
#    X_val_=skb.fit_transform(X_val_, y_val)
#    X=X[:,skb.get_support(True)]
#    X_test_=X_test_[:,skb.get_support(True)]
    
#    #smote
#    X_smote=np.load('./data/smote/train_data_20w.npz')
#    X=np.array(X_smote['train_data'])[:-150000]
#    y=np.reshape(np.array(X_smote['y_train'])[:-150000],(-1,))
    
    #add new feature:embedding similarity
    X_train=np.load('./data_word2vec_feature/train_vector_lower4.npz')
    X_test=np.load('./data_word2vec_feature/test_vector_lower4.npz')
    
    X2=np.reshape(np.array(X_train['train_data'][:-50000])[:,24],(-1,1))  
    X_val_2=np.reshape(np.array(X_train['train_data'][-50000:])[:,24],(-1,1))  
    X_test_2=np.reshape(np.array(X_test['test_data'])[:,24],(-1,1))  
    
    #get the "has_data"
    X_val_2=X_val_2[val_hasdata_index]
    X_test_2=X_test_2[test_hasdata_index]
    
    X=np.concatenate((X,X2),axis=1)
    X_val_=np.concatenate((X_val_,X_val_2),axis=1)
    X_test_=np.concatenate((X_test_,X_test_2),axis=1)
    
    val_save=pd.DataFrame(val_hasdata_index,columns=['data_index'])
    test_save=pd.DataFrame(test_hasdata_index,columns=['data_index'])
    
    #train lgb
#    feature_names=['prefix_cut_in_title', 'prefix_cut_in_start_title',
#       'query_prediction_length', 'prefix_count', 'prefix_click',
#       'prefix_ctr', 'title_click', 'title_count', 'title_ctr',
#       'tag_click', 'tag_count', 'tag_ctr', 'prefix_title_click',
#       'prefix_titlecount', 'prefix_title_ctr', 
#       'prefix_tag_click',
#       'prefix_tagcount', 'prefix_tag_ctr', 
#       'title_tag_click',
#       'title_tagcount', 'title_tag_ctr', 
#       'prefix_title_tag_click',
#       'prefix_title_tagcount', 'prefix_title_tag_ctr',
#       'prefix_tag_nunique', 'title_tag_nunique', 'is_in_title',
#       'leven_distance', 'distance_rate',]

    feature_names=['5','6','7','8','9']


    feature_imp,val_save,test_save=lgbm2(X,y,X_val_,y_val,X_test_,val_save,test_save,feature_names)
    
    
    """
    2.not data
    """
    #add new feature:embedding similarity
    X_train=np.load('./data_word2vec_feature/train_vector_lower3.npz')
    X_test=np.load('./data_word2vec_feature/test_vector_lower3.npz')
    
    X=np.array(X_train['train_data'][:-50000])[:,:-3]
    y=np.reshape(np.array(X_train['y_train'][:-50000],dtype=np.float),(-1,))
    
    X_val_=np.array(X_train['train_data'][-50000:])[:,:-3]
    y_val=np.reshape(np.array(X_train['y_train'][-50000:],dtype=np.float),(-1,))
    
    X_test_=np.array(X_test['test_data'])[:,:-3]
    y_test=np.reshape(np.array(X_test['y_test'],dtype=np.float),(-1,))

    #get the "no_data"
    y_val=y_val[val_notdata_index]
    X_val_=X_val_[val_notdata_index]
    X_test_=X_test_[test_notdata_index]

    #选择K个最好的特征，返回选择特征后的数据
    X=np.where(X>=0,X,float('nan'))
    for i in range(len(X[0])):
        pad=np.nanmean(X[:,i])
        X[:,i]=np.where(X[:,i]>=0,X[:,i],pad)
        X_val_[:,i]=np.where(X_val_[:,i]>=0,X_val_[:,i],pad)
        X_test_[:,i]=np.where(X_test_[:,i]>=0,X_test_[:,i],pad)
#    X=np.where(X>=0,X,0)
#    X_val_=np.where(X_val_>=0,X_val_,0)
#    X_test_=np.where(X_test_>=0,X_test_,0)
#
#    X,X_val_,X_test_=select_kbest(X,X_val_,X_test_,20)
        
    feature_names=['prefix_title_sim', 'prefix_title_dot',
           'text0_title_sim', 'text0_title_dot',
           'text1_title_sim', 'text1_title_dot',
           'text2_title_sim', 'text2_title_dot', 
           'text3_title_sim', 'text3_title_dot',
           'text4_title_sim', 'text4_title_dot',
           'text5_title_sim', 'text5_title_dot',
           'text6_title_sim', 'text6_title_dot',
           'text7_title_sim', 'text7_title_dot',
           'text8_title_sim', 'text8_title_dot',
           'text9_title_sim', 'text9_title_dot',
           u'max_similar',u'mean_similar', u'weight_similar',
           
           
           'prefix_cut_in_title', 'prefix_cut_in_start_title',
           'query_prediction_length','is_in_title',
           'leven_distance', 'distance_rate',
           'title_count','tag_count',
           'tag_click', 'tag_ctr',
           'title_tagcount',
           
           u'tag', u'prefix_len', 
           u'prediction0',u'similarity0', u'equal_rate0', 
           u'prediction1', u'similarity1',u'equal_rate1', 
           u'prediction2', u'similarity2', u'equal_rate2',
           u'prediction3', u'similarity3', u'equal_rate3', 
           u'prediction4',u'similarity4', u'equal_rate4', 
#           u'prediction5', u'similarity5',u'equal_rate5', 
           u'prediction6', u'similarity6', u'equal_rate6',]
#           u'prediction7', u'similarity7', u'equal_rate7', 
#           u'prediction8',u'similarity8', u'equal_rate8', 
#           u'prediction9', u'similarity9',u'equal_rate9', u'prefix_title_sim']
    
    #statis feature+ leven_distance feature
    X2=np.load('./data_word2vec_feature/train_dnn_data3_feature.npz')
    X_val_2=np.load('./data_word2vec_feature/val_dnn_data3_feature.npz')
    X_test_2=np.load('./data_word2vec_feature/test_dnn_data3_feature.npz')
    
    X2=X2['train_data']
    X_val_2=X_val_2['val_data']
    X_test_2=X_test_2['test_data']

    X_val_2=X_val_2[val_notdata_index]
    X_test_2=X_test_2[test_notdata_index]
    
    for i in [0,1,2,26,27,28,7,10,9,11,19]:#[3,4,5,29,30,31,10,13,12,14,22]:
        X=np.concatenate((X,np.reshape(X2[:,i],(-1,1))),axis=1)
        X_val_=np.concatenate((X_val_,np.reshape(X_val_2[:,i],(-1,1))),axis=1)
        X_test_=np.concatenate((X_test_,np.reshape(X_test_2[:,i],(-1,1))),axis=1)


    #query length sim
    data=pd.read_csv('./data/data.csv')
    X3=data[data['flag']==0]
    X_val_3=data[data['flag']==1]
    X_test_3=data[data['flag']==2]
    
    X3=X3.drop(['flag','label'],axis=1)
    X3=np.array(X3)
    X_val_3=X_val_3.drop(['flag','label'],axis=1)
    X_val_3=np.array(X_val_3)
    X_test_3=X_test_3.drop(['flag','label'],axis=1)
    X_test_3=np.array(X_test_3)
    
    #选择K个最好的特征，返回选择特征后的数据
    X_val_3=X_val_3[val_notdata_index]
    X_test_3=X_test_3[test_notdata_index]
        
    X3,X_val_3,X_test_3=select_kbest(X3,X_val_3,X_test_3,20)

    #concate
    X=np.concatenate((X,X3),axis=1)
    X_val_=np.concatenate((X_val_,X_val_3),axis=1)
    X_test_=np.concatenate((X_test_,X_test_3),axis=1)
    
    
    
    val_nodata_save=pd.DataFrame(val_notdata_index,columns=['data_index'])
    test_nodata_save=pd.DataFrame(test_notdata_index,columns=['data_index'])
    
    #train
    feature_imp2,val_nodata_save,test_nodata_save=lgbm2(X,y,X_val_,y_val,X_test_,val_nodata_save,test_nodata_save,feature_names)
    
#    print("begin training")
#    y_test=np.reshape(np.array([-1]*len(test_notdata_index)),(-1,1))
#    dnn=DNN(K=len(X[0]))
#    val_nodata_save,test_nodata_save=dnn.dnn_model_split(X, y, X_test_, y_test, X_val_, y_val,val_nodata_save,test_nodata_save)
    
    """
    3.merge
    """
    val_save_fin,test_save_fin=merge_val_test_data(val_save,test_save,val_nodata_save,test_nodata_save)
    
    test_save_fin['label']=test_save_fin['label'].apply(lambda x:int(x))
    test_save_fin['label'].to_csv('./submit/split_data/result1.csv',index = False)
    
#    val_save_fin['label']=val_save_fin['label'].apply(lambda x:int(x))
#    val_save_fin['label'].to_csv('./submit/split_data/result_val.csv',index = False)
    
#    val_save.to_csv('./submit/split_data/val_save.csv',index = False)
    