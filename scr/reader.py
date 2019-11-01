# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:19:04 2018

@author: NLP
"""
import re
import numpy as np
from keras.utils import to_categorical
import pickle

def read_data(path):
    f = open(path,'r',encoding='utf-8')
    data = f.readlines()
    target = []
    context = []
    polarity = []
    for j,t in enumerate(data):
        tokens = t.strip().split()
        find_label = False
        one_target=[]
        one_context = []
        for i,word in enumerate(tokens):
            if '/p' in word or '/n' in word or '/0' in word:
                end = 'xx'
                y = 0
                if '/p' in t:
                    end = '/p'
                    y = 1
                elif '/n' in t:
                    end = '/n'
                    y = 0
                elif '/0' in t:
                    end = '/0'
                    y = 2
                one_target.append(word.strip(end))
                if not find_label:
                    find_label = True
                    polarity.append(y)
                if 'aspect_term' not in one_context:
                    one_context.append('aspect_term')
            else:
                one_context.append(word)
        context.append(one_context)
        target.append(one_target)
    return target, context, polarity

def read_vec(path):
    f = open(path,'rb')
    dict_name = pickle.load(f)
    return dict_name


def get_dataset(path):
    target,context,polarity = read_data(path)
    label = np.array(to_categorical(polarity))
    maxlen = 0
    for i in range(len(target)):
        length = len(target[i]) + len(context[i]) - 1
        if length > maxlen:
            maxlen = length
    return target,context,label, maxlen
    

           
            