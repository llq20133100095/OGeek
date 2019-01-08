#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:40:58 2018

@author: llq
"""
import json
import time

from pyhanlp import HanLP
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def char_lower(char):
    char = char.lower()
    return char

def char_list_cheaner(char_list, stop_words=None):
    new_char_list = list()
    for char in char_list:
        if len(char) <= 1:
            continue
        if stop_words and char in stop_words:
            continue
        new_char_list.append(char)

    return new_char_list

def get_sentence(file_path="train"):
    with open(file_path, "r") as f:
        line = f.readline()

        while line:
            line_arr = line.split("\t")
            
            prefix = line_arr[0]
            yield char_lower(prefix)
            
            query_prediction = line_arr[1]
            sentences = json.loads(query_prediction)
            for sentence in sentences:
                yield char_lower(sentence)

            title = line_arr[2]
            yield char_lower(title)

            line = f.readline()


class MySentence(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for sentence in get_sentence(self.fname):
            seg_list = HanLP.segment(sentence)
            for i,word in enumerate(seg_list):
                seg_list[i]=str(word).split('/')[0]
            
#            print str(seg_list)
#            seg_list = char_list_cheaner(seg_list)
            if seg_list:
                yield seg_list

def hanlp_cut(text,save_file):
    seg_list = HanLP.segment(text)
    for i,word in enumerate(seg_list):
        word=str(word).split('/')[0]
        save_file.write(word+" ")
    save_file.write("\n")

def cut_txt(train_file,val_file,text_file,save_file):
    save_file=open(save_file,'w')
    
    file_list=[train_file,val_file,text_file]
    
    for file_name in file_list:
        with open(file_name, "r") as f:
            line = f.readline()
    
            while line:
                line_arr = line.split("\t")
                
                prefix = line_arr[0]
                prefix = char_lower(prefix)
                hanlp_cut(prefix,save_file)
                
                query_prediction = line_arr[1]
                sentences = json.loads(query_prediction)
                for sentence in sentences:
                    sentence=char_lower(sentence)
                    hanlp_cut(sentence,save_file)
    
                title = line_arr[2]
                title=char_lower(title)
                hanlp_cut(title,save_file)
    
                line = f.readline()
    
    save_file.close()

def build_model(fname):
    model_name = "../data/w2v.bin"
    my_model = Word2Vec(LineSentence(fname), size=300, window=5, sg=1, hs=1, min_count=2)
    my_model.wv.save_word2vec_format(model_name, binary=False)

if __name__ == "__main__":
    t0 = time.time()
    save_file='../data/origin_cut.txt'
#    cut_txt("../data/oppo_round1_train_20180929.txt","../data/oppo_round1_vali_20180929.txt","../data/oppo_round1_test_A_20180929.txt",save_file)
    build_model(save_file)

    print(time.time() - t0)
