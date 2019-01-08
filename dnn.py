#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:20:47 2018

@author: llq
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from dnn_data import get_data,process_data,onehot_feature,iterate_minibatches,iterate_minibatches2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
tf.reset_default_graph()

class DNN():
    def __init__(self,voc_size=1,tag_size=24,K=None):
        self.voc_size=voc_size
        self.tag_size=tag_size
        self.embed_size=22
        self.logdir='./summary'
        self.epochs=30
        self.batch_size=600
        self.N=5
        self.model_name="dnn_"+str(self.batch_size)
        self.submit_file="./submit/dnn/result_dnn.csv"
        self.regularizer=0.001
        self.K=K
        
        with tf.name_scope("inputs"):
#            self.prefix=tf.placeholder(dtype=tf.int32,shape=(None,1),name='prefix')
#            self.title = tf.placeholder(dtype=tf.int32, shape=(None, 1), name="title")
#            self.tag = tf.placeholder(tf.float32, shape=(None, self.tag_size), name="tag")
            self.target=tf.placeholder(tf.float32, shape=(None, 1), name="target")
            
            self.feature=tf.placeholder(tf.float32, shape=(None, self.K), name="input")
            self.dropout=tf.placeholder(tf.float32, name="dropout")
            
    def dnn_model1(self, X_prefix, X_title, X_tag, y, X_prefix_test, X_title_test, X_tag_test, y_test):
        """
        prefix and title -> id
        tag -> one-hot
        """
        with tf.name_scope("embedding"):
            embeddings = tf.get_variable("word_embeddings",[self.voc_size,self.embed_size])
            prefix_feature=tf.reshape(tf.gather(embeddings, self.prefix),(-1,self.embed_size))
            title_feature=tf.reshape(tf.gather(embeddings, self.title),(-1,self.embed_size))
            feature=tf.concat([prefix_feature,title_feature,self.tag],1)
        
        with tf.name_scope("dense"):
            dense = tf.layers.dense(inputs=feature, units=128, activation=tf.nn.relu,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            drop_dense=tf.layers.dropout(dense, rate=self.dropout, training=True)
            y_pred=tf.layers.dense(inputs=drop_dense,units=1,activation=tf.nn.sigmoid,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
        
        with tf.name_scope("Loss"):
            logloss=-tf.reduce_mean(tf.log(y_pred)*self.target+(1-self.target)*tf.log(1-y_pred))\
                 +tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', logloss)
        
        with tf.name_scope("training_op"):
            optimizer = tf.train.AdamOptimizer()
            training_op = optimizer.minimize(logloss)
            
        # Summary
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
        
        #init
        init=tf.global_variables_initializer()
    
        #split train_data: train and val
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        for (train_in, test_in) in skf.split(X_prefix, y):
            X_prefix_train, X_title_train, X_tag_train, \
            X_prefix_val, X_title_val, X_tag_val, \
            y_train, y_val\
            = X_prefix[train_in], X_title[train_in], X_tag[train_in],\
            X_prefix[test_in], X_title[test_in], X_tag[test_in], \
            y[train_in], y[test_in]
            break
        
        best_f1_score=0
        best_epoch=0
        ypred_test=0
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                start=time.time()
                #train
                loss_epoch=[]
                for batch in iterate_minibatches(X_prefix_train, X_title_train, X_tag_train, y_train, self.batch_size, shuffle=True):
                    prefix, title, tag, y= batch
                    feed_dict_train = {
                        self.prefix: prefix,
                        self.title: title,
                        self.tag: tag,
                        self.target: y,
                        self.dropout:0.5,
                    }
                    
                    sess.run(training_op,feed_dict_train)
                    e,summary=sess.run([logloss,merged_summary],feed_dict_train)
                    loss_epoch.append(e)
                    
                summary_writer.add_summary(summary,epoch)
                print("epoch: %d" % epoch)
                print("train loss: %f" % np.mean(loss_epoch))
                
                #val:
                loss_epoch=[]
#                for batch in iterate_minibatches(X_prefix_val, X_title_val, X_tag_val, y_val, self.batch_size, shuffle=False):
                prefix, title, tag, y= batch
                feed_dict_val = {
                    self.prefix: X_prefix_val,
                    self.title: X_title_val,
                    self.tag: X_tag_val,
                    self.target: y_val,
                    self.dropout:0.0,
                }
                    
                ff=feature.eval(feed_dict=feed_dict_val)
                e=logloss.eval(feed_dict=feed_dict_val)
                loss_epoch.append(e)
                ypred_val=sess.run(y_pred,feed_dict=feed_dict_val)
                f1_sco=f1_score(y_val,np.where(ypred_val>0.4,1,0))
                print("val loss: %f" % np.mean(loss_epoch))
                print("f1_score: %f" % f1_sco)
                
                #test
                if(best_f1_score<f1_sco):
                    best_f1_score=f1_sco
                    best_epoch=epoch
                    feed_dict_test = {
                        self.prefix: X_prefix_test,
                        self.title: X_title_test,
                        self.tag: X_tag_test,
                        self.target: y_test,
                        self.dropout:0.0
                    }
                    ypred_test=sess.run(y_pred,feed_dict=feed_dict_test)
                
                saver = tf.train.Saver()
                saver.save(sess, "./checkpoint_dir/MyModel_"+self.model_name, global_step=epoch)
                print("time: %f s" % (time.time()-start))
                
            print("best_f1_score: %f" % best_f1_score)
            print("best_epoch: %d" % best_epoch)
            
            ypred_test=np.where(ypred_test>0.4,1,0)
            ypred_test=pd.DataFrame(ypred_test,columns=['label'])
            ypred_test['label']=ypred_test['label'].apply(lambda x:int(x))
            ypred_test['label'].to_csv(self.submit_file,index = False)
            
            return ff
    
    def dnn_model2(self,X_train,y_train,X_test_,y_test, X_val, y_val):
        """
        1.dnn
        2.dropout+l2
        """
        with tf.name_scope("dense"):
            dense = tf.layers.dense(inputs=self.feature, units=300, activation=tf.nn.relu6,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#            dense=tf.layers.dropout(dense, rate=self.dropout, training=True)
#            dense = tf.layers.dense(inputs=dense, units=200, activation=tf.nn.relu6,\
#                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#            dense = tf.layers.dense(inputs=dense, units=300, activation=tf.nn.relu6,\
#                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#            dense = tf.layers.dense(inputs=dense, units=200, activation=tf.nn.relu6,\
#                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#            dense = tf.layers.dense(inputs=dense, units=100, activation=tf.nn.relu6,\
#                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
        
#        with tf.name_scope("attention"):
#            w_att = tf.get_variable('wight_att', [1, 100], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), \
#                regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#            b_att = tf.get_variable('b_att', [100], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), \
#                regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#            u_att = tf.get_variable('u_att', [100,1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), \
#                regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#
#            H=tf.reshape(dense,(-1,1))
#            M=tf.tanh(tf.matmul(H, w_att)+b_att)
#            alpha=tf.nn.softmax(tf.reshape(tf.matmul(M,u_att),(-1,dense.shape[1])),dim=1)
#            y_pred=tf.reshape(tf.reduce_sum(dense * alpha, axis=1),(-1,1))
            
            y_pred=tf.layers.dense(inputs=dense,units=1,activation=tf.nn.sigmoid,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
        
        with tf.name_scope("Loss"):
            logloss=-tf.reduce_mean(tf.log(y_pred)*self.target+(1-self.target)*tf.log(1-y_pred)) \
                +tf.losses.get_regularization_loss()
#            label_onehot=tf.one_hot(indices=self.target, depth=2, dtype=tf.float32)
#            logloss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot, logits=y_pred)) \
#                +tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', logloss)
        
        with tf.name_scope("training_op"):
            global_step = tf.placeholder(dtype=tf.int32)
            learning_rate = tf.train.exponential_decay(1e-4,global_step,decay_steps=2,decay_rate=1.0,staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            training_op = optimizer.minimize(logloss)
        
        # Summary
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
        
        #init
        init=tf.global_variables_initializer()
    
#        #split train_data: train and val:42
#        X=np.concatenate((X_train,X_val),axis=0)
#        y=np.concatenate((y_train,y_val),axis=0)
#        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
#        for (train_in, test_in) in skf.split(X, y):
#            X_train, X_val, y_train, y_val = X[train_in], X[test_in], y[train_in], y[test_in]
#            break
        
        best_f1_score=0
        best_epoch=0
        ypred_test=0
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                start=time.time()
                #train
                loss_epoch=[]
                for batch in iterate_minibatches2(X_train, y_train, self.batch_size, shuffle=True):
                    train, traget = batch
                    feed_dict_train = {
                        self.feature: train,
                        self.dropout:0.5,
                        self.target: traget,
                        global_step:epoch,
                    }
                    
                    sess.run(training_op,feed_dict_train)
                    e,summary,lr,out=sess.run([logloss,merged_summary,learning_rate,y_pred],feed_dict_train)
                    loss_epoch.append(e)
                    
                summary_writer.add_summary(summary,epoch)
                print("epoch: %d | lr: %f" % (epoch,lr))
                print("train loss: %f" % np.mean(loss_epoch))
                
                
                
                #val:
                loss_epoch=[]
#                y_pre_val=np.array([[0]])
#                for batch in iterate_minibatches2(X_val, y_val, self.batch_size, shuffle=False):
#                    train, traget = batch
                feed_dict_val = {
                    self.feature: X_val,
                    self.dropout:0.0,
                    self.target: y_val,
                }
                    
                e=logloss.eval(feed_dict=feed_dict_val)
                loss_epoch.append(e)
                ypred_val=sess.run(y_pred,feed_dict=feed_dict_val)
#                y_pre_val=np.concatenate((y_pre_val,np.reshape(ypred_val,(-1,1))),axis=0)
                
#                y_pre_val=y_pre_val[1:]
                f1_sco=f1_score(y_val,np.where(ypred_val>0.4,1,0))
                print("val loss: %f" % np.mean(loss_epoch))
                print("f1_score: %f" % f1_sco)
                
                #test
                if(best_f1_score<f1_sco):
                    best_f1_score=f1_sco
                    best_epoch=epoch
                    feed_dict_test = {
                        self.feature: X_test_,
                        self.dropout:0.0,
                        self.target: y_test,
                    }
                    ypred_test1=sess.run(y_pred,feed_dict=feed_dict_test)
                    ypred_test=np.where(ypred_test1>0.4,1,0)
                    ypred_test=pd.DataFrame(ypred_test,columns=['label'])
                    ypred_test['label']=ypred_test['label'].apply(lambda x:int(x))
                    ypred_test['label'].to_csv(self.submit_file,index = False)
                    
                saver = tf.train.Saver()
                saver.save(sess, "./checkpoint_dir/MyModel_"+self.model_name, global_step=epoch)
                print("time: %f s" % (time.time()-start))
                
            print("best_f1_score: %f" % best_f1_score)
            print("best_epoch: %d" % best_epoch)
               
            return ypred_test1,ypred_val
        
    def dnn_model_split(self,X_train,y_train,X_test_,y_test, X_val, y_val, val_nodata_save, test_nodata_save):
        """
        1.dnn
        2.dropout+l2
        """
        with tf.name_scope("dense"):
            dense = tf.layers.dense(inputs=self.feature, units=300, activation=tf.nn.relu6,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))            
            dense=tf.layers.dropout(dense, rate=self.dropout, training=True)

            dense = tf.layers.dense(inputs=dense, units=200, activation=tf.nn.relu6,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            dense=tf.layers.dropout(dense, rate=self.dropout, training=True)

            dense = tf.layers.dense(inputs=dense, units=300, activation=tf.nn.relu6,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            dense=tf.layers.dropout(dense, rate=self.dropout, training=True)

            dense = tf.layers.dense(inputs=dense, units=200, activation=tf.nn.relu6,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            dense=tf.layers.dropout(dense, rate=self.dropout, training=True)

            dense = tf.layers.dense(inputs=dense, units=100, activation=tf.nn.relu6,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            dense=tf.layers.dropout(dense, rate=self.dropout, training=True)

            y_pred=tf.layers.dense(inputs=dense,units=1,activation=tf.nn.sigmoid,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
        
        with tf.name_scope("Loss"):
            logloss=-tf.reduce_mean(tf.log(y_pred)*self.target+(1-self.target)*tf.log(1-y_pred)) \
                +tf.losses.get_regularization_loss()
#            label_onehot=tf.one_hot(indices=self.target, depth=2, dtype=tf.float32)
#            logloss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot, logits=y_pred)) \
#                +tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', logloss)
        
        with tf.name_scope("training_op"):
            global_step = tf.placeholder(dtype=tf.int32)
            learning_rate = tf.train.exponential_decay(1e-4,global_step,decay_steps=2,decay_rate=1.0,staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            training_op = optimizer.minimize(logloss)
        
        # Summary
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
        
        #init
        init=tf.global_variables_initializer()
    
#        #split train_data: train and val:42
#        X=np.concatenate((X_train,X_val),axis=0)
#        y=np.concatenate((y_train,y_val),axis=0)
#        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
#        for (train_in, test_in) in skf.split(X, y):
#            X_train, X_val, y_train, y_val = X[train_in], X[test_in], y[train_in], y[test_in]
#            break
        
        best_f1_score=0
        best_epoch=0
        ypred_test=0
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                start=time.time()
                #train
                loss_epoch=[]
                for batch in iterate_minibatches2(X_train, y_train, self.batch_size, shuffle=True):
                    train, traget = batch
                    feed_dict_train = {
                        self.feature: train,
                        self.dropout:0.5,
                        self.target: traget,
                        global_step:epoch,
                    }
                    
                    sess.run(training_op,feed_dict_train)
                    e,summary,lr,out=sess.run([logloss,merged_summary,learning_rate,y_pred],feed_dict_train)
                    loss_epoch.append(e)
                    
                summary_writer.add_summary(summary,epoch)
                print("epoch: %d | lr: %f" % (epoch,lr))
                print("train loss: %f" % np.mean(loss_epoch))
                
                
                
                #val:
                loss_epoch=[]
#                y_pre_val=np.array([[0]])
#                for batch in iterate_minibatches2(X_val, y_val, self.batch_size, shuffle=False):
#                    train, traget = batch
                feed_dict_val = {
                    self.feature: X_val,
                    self.dropout:0.0,
                    self.target: y_val,
                }
                    
                e=logloss.eval(feed_dict=feed_dict_val)
                loss_epoch.append(e)
                ypred_val=sess.run(y_pred,feed_dict=feed_dict_val)
#                y_pre_val=np.concatenate((y_pre_val,np.reshape(ypred_val,(-1,1))),axis=0)
                
#                y_pre_val=y_pre_val[1:]
                f1_sco=f1_score(y_val,np.where(ypred_val>0.4,1,0))
                print("val loss: %f" % np.mean(loss_epoch))
                print("f1_score: %f" % f1_sco)
                
                #test
                if(best_f1_score<f1_sco):
                    best_f1_score=f1_sco
                    best_epoch=epoch
                    #save val
                    val_nodata_save['label']=np.where(ypred_val>0.4, 1,0)
                    
                    feed_dict_test = {
                        self.feature: X_test_,
                        self.dropout:0.0,
                        self.target: y_test,
                    }
                    ypred_test1=sess.run(y_pred,feed_dict=feed_dict_test)
                    ypred_test=np.where(ypred_test1>0.4,1,0)
                    ypred_test=pd.DataFrame(ypred_test,columns=['label'])
                    ypred_test['label']=ypred_test['label'].apply(lambda x:int(x))
                    ypred_test['label'].to_csv(self.submit_file,index = False)
                    #save test
                    test_nodata_save['label']=np.where(ypred_test1>0.4, 1,0)
                    
                saver = tf.train.Saver()
                saver.save(sess, "./checkpoint_dir/MyModel_"+self.model_name, global_step=epoch)
                print("time: %f s" % (time.time()-start))
                
            print("best_f1_score: %f" % best_f1_score)
            print("best_epoch: %d" % best_epoch)
               
            return val_nodata_save,test_nodata_save
        
if __name__=='__main__':
    #
    #"""
    #1.
    #"""
    #train_data,test_data,voc_size,tag_size=process_data()
    #
    ##delete the column 
    #train_data_ = train_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
    #test_data_ = test_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
    #
    ##title_id to one-hot
    #X_tag,X_test_tag=onehot_feature(train_data_,test_data_)
    #
    #X_prefix=np.array(train_data_[['prefix_id']])
    #X_title=np.array(train_data_[['title_id']])
    #X_tag=X_tag.toarray()
    #y=np.array(train_data_[['label']])
    #
    #X_test_prefix=np.array(test_data_[['prefix_id']])
    #X_test_title=np.array(test_data_[['title_id']])
    #X_test_tag=X_test_tag.toarray()
    #y_test=np.array(test_data_[['label']])
    #
    #print("begin training")
    #
    #dnn=DNN(voc_size,tag_size)
    #ff=dnn.dnn_model1(X_prefix, X_title, X_tag, y, X_test_prefix, X_test_title, X_test_tag, y_test)
    
    #"""
    #2.only use the "ctr" feature
    #"""
    #feature=[2,5,8,11,14,17]
    #X,y,X_test_,test_data_=get_data()
    #
    #X_train=np.reshape(X[:,feature[0]],(-1,1))
    #X_test=np.reshape(X_test_[:,feature[0]],(-1,1))
    #for i in feature[1:]:
    #    X_train=np.concatenate((X_train,np.reshape(X[:,i],(-1,1))),axis=1)
    #    X_test=np.concatenate((X_test,np.reshape(X_test_[:,i],(-1,1))),axis=1)
    ##chang nan to 0
    #X_train=np.nan_to_num(X_train) 
    #X_test=np.nan_to_num(X_test)
    #
    #y=np.reshape(y,(-1,1))
    #y_test=np.array(test_data_[['label']])
    #
    #print("begin training")
    #dnn=DNN()
    #dnn.dnn_model2(X_train, y, X_test, y_test)
    
    #"""
    #3.all feature -> normalization(0,1)
    #"""
    ##feature=[2,5,8,11,14,17]
    ##feature=[3,6,9,12,15,18,21]
    #X,y,X_test_,test_data_=get_data()
    #
    ##chang nan to 0
    #X=np.nan_to_num(X) 
    #X_test_=np.nan_to_num(X_test_)
    #
    ##normalization
    #for i in range(len(X[0])):
    #    if(X[0,i]>1):
    #        se=X[:,i]
    #        semax=se.max()
    #        semin=se.min()
    #        X[:,i]=(se-semin)/(semax-semin)
    #        se_text=X_test_[:,i]
    #        X_test_[:,i]=(se_text-se_text.min())/(se_text.max()-se_text.min())
    #
    #y=np.reshape(y,(-1,1))
    #y_test=np.array(test_data_[['label']])
    #
    #print("begin training")
    #dnn=DNN(K=24)
    #out=dnn.dnn_model2(X, y, X_test_, y_test)
    
    
    #"""
    #4.new feature
    #"""
    #data=pd.read_csv('./data/data.csv')
    #
    #X=data[data['flag']==0]
    #X_val=data[data['flag']==1]
    #X_test=data[data['flag']==2]
    #
    #X=pd.concat([X,X_val])
    #y=np.array(X['label'])
    #y=np.reshape(y,(-1,1))
    #y_test=np.array(X_test['label'])
    #y_test=np.reshape(y_test,(-1,1))
    #
    #
    #X=X.drop(['flag','label'],axis=1)
    #X=np.array(X)
    #
    #X_test=X_test.drop(['flag','label'],axis=1)
    #X_test=np.array(X_test)
    #
    ##chang nan to 0
    #X=np.nan_to_num(X) 
    #X_test=np.nan_to_num(X_test)
    #
    #print("begin training")
    #dnn=DNN()
    #out=dnn.dnn_model2(X, y, X_test, y_test)
    
    #"""
    #5.embedding
    #"""
    #X_train=np.load('./data/train_vector.npz')
    #X_test=np.load('./data/test_vector.npz')
    #
    #X=X_train['X_train']
    #y=X_train['y_train']
    #
    #X_test_=X_test['X_test']
    #y_test=X_test['y_test']
    #
    ##X2,_,X_test_2,_=get_data()
    ##for i in range(len(X2[0])):
    ##    if(X2[0,i]>1):
    ##        se=X2[:,i]
    ##        semax=se.max()
    ##        semin=se.min()
    ##        X2[:,i]=(se-semin)/(semax-semin)
    ##        se_text=X_test_2[:,i]
    ##        X_test_2[:,i]=(se_text-se_text.min())/(se_text.max()-se_text.min())
    #
    #X=np.concatenate((X[:,:600],X2),axis=1)
    #X_test_=np.concatenate((X_test_[:,:600],X_test_2),axis=1)
    #del X2,X_test_2
    #
    #print("begin training")
    #dnn=DNN(K=622)
    #out=dnn.dnn_model2(X, y, X_test_, y_test)
                
    #"""
    #6.feature choose:KBest
    #"""
    #X,y,X_test_,test_data_=get_data()
    #
    ##add new feature:simarity
    #data=pd.read_csv('./data/data.csv')
    #X2=data[data['flag']==0]
    #X_val2=data[data['flag']==1]
    #X_test2=data[data['flag']==2]
    #X2=pd.concat([X2,X_val2])
    #
    #X2=X2.drop(['flag','label','tag','prefix_len'],axis=1)
    ##X2=X2[['prefix_title_sim']]
    #X2=np.array(X2)
    #X_test2=X_test2.drop(['flag','label','tag','prefix_len'],axis=1)
    ##X_test2=X_test2[['prefix_title_sim']]
    #X_test2=np.array(X_test2)
    #
    #X=np.concatenate((X,X2),axis=1)
    #X_test_=np.concatenate((X_test_,X_test2),axis=1)
    #del X2,X_test2
    #
    ##chang nan to 0
    #X=np.nan_to_num(X) 
    #X_test_=np.nan_to_num(X_test_)
    #
    ##normalization
    #for i in range(len(X[0])):
    #    if(X[0,i]>1):
    #        se=X[:,i]
    #        semax=se.max()
    #        semin=se.min()
    #        X[:,i]=(se-semin)/(semax-semin)
    #        se_text=X_test_[:,i]
    #        X_test_[:,i]=(se_text-se_text.min())/(se_text.max()-se_text.min())
    #
    ##选择K个最好的特征，返回选择特征后的数据
    #K=20
    #skb = SelectKBest(chi2, k=K)
    #X_new=skb.fit_transform(X, y)
    #X_test_new=X_test_[:,skb.get_support(True)]
    #X=X_new
    #X_test_=X_test_new
    #del X_new,X_test_new
    #
    #y=np.reshape(y,(-1,1))
    #y_test=np.array(test_data_[['label']])
    #
    #print("begin training")
    #dnn=DNN(K=K)
    #out=dnn.dnn_model2(X, y, X_test_, y_test)
                
    """
    7.distance feature+ statis feature
    """
    X_train=np.load('./data_word2vec_feature/train_vector.npz')
    X_test=np.load('./data_word2vec_feature/test_vector.npz')
    
    X=np.array(X_train['train_data'])
    y=np.array(X_train['y_train'],dtype=np.float)
    
    X_test_=np.array(X_test['test_data'])
    y_test=np.array(X_test['y_test'],dtype=np.float)
    
    #X2,_,X_test_2,_=get_data()
    X2=np.load('./data_word2vec_feature/train_dnn_data3_feature.npz')
    X_val_2=np.load('./data_word2vec_feature/val_dnn_data3_feature.npz')
    X_test_2=np.load('./data_word2vec_feature/test_dnn_data3_feature.npz')
    X2=np.array(X2['train_data'])
    X_val_2=np.array(X_val_2['val_data'])
    X_test_2=np.array(X_test_2['test_data'])
    
    #chang nan to 0
    X2=np.nan_to_num(X2) 
    X_val_2=np.nan_to_num(X_val_2) 
    X_test_2=np.nan_to_num(X_test_2)
    for i in range(len(X2[0])):
        if(X2[3,i]>1):
            se=X2[:,i]
            semax=se.max()
            semin=se.min()
            X2[:,i]=(se-semin)/(semax-semin)
            se_text=X_val_2[:,i]
            X_val_2[:,i]=(se_text-se_text.min())/(se_text.max()-se_text.min())
            se_text=X_test_2[:,i]
            X_test_2[:,i]=(se_text-se_text.min())/(se_text.max()-se_text.min())
    
    ##选择K个最好的特征，返回选择特征后的数据
    #K=6
    #skb = SelectKBest(chi2, k=K)
    #X_new=skb.fit_transform(X2, y[:-50000])
    #X_val_new=X_val_2[:,skb.get_support(True)]
    #X_test_new=X_test_2[:,skb.get_support(True)]
    #X_val_2=X_val_new
    #X2=X_new
    #X_test_2=X_test_new
    #del X_new,X_val_new,X_test_new
    
    X_val=np.concatenate((X[-50000:],X_val_2),axis=1)
    X=np.concatenate((X[:-50000],X2),axis=1)
    X_test_=np.concatenate((X_test_,X_test_2),axis=1)
    #X_val=X_val_2
    #X=X2
    #X_test_=X_test_2
    del X2,X_test_2
    
    #X_val=X[-50000:]
    #X=X[:-50000]
    y_val=y[-50000:]
    y=y[:-50000]
    
    print("begin training")
    dnn=DNN(K=len(X[0]))
    out,ypred_val=dnn.dnn_model2(X, y, X_test_, y_test, X_val, y_val)