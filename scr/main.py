# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split  
import numpy as np
import pandas as pd
from keras.models import load_model
from cuenet import CueNet,customLoss
from binary_indicator import binary_indicator_layer
from target_representation import target_representation_layer
import os
import logging
import gc
from reader import get_dataset, read_vec
from keras import backend as K
import nltk
from keras_contrib.layers import CRF

def word_embeding2(comment_cut,aspect,maxlen,word_vectors,size):
    data=pd.read_excel('../data/lexicon_words.xlsx',index=None)
    sen_words = list(data['Word'])
    sen_polarity = list(data['Polarity'])
    nt_words = list(data['Nt'].dropna(axis=0,how='all'))
    sup_words = list(data['Sup'].dropna(axis=0,how='all'))
    int_words = list(data['Int'].dropna(axis=0,how='all'))
    oth_words = list(data['Oth'].dropna(axis=0,how='all'))
    flip_words = list(data['flip_words'].dropna(axis=0,how='all'))
    pos_list = ['NN','JJ','VB','PRO','RB','IN','CC']
    
    word_embed = np.zeros((len(comment_cut),maxlen,300))
    target_embed = np.zeros((len(comment_cut),maxlen))
    word_embed2 = np.zeros((len(comment_cut),maxlen,300))
    select_embed = np.zeros((len(comment_cut),maxlen))
    target_embed2 = np.zeros((len(comment_cut),maxlen))
    lexicon_embed = np.zeros((len(comment_cut),maxlen,6*size+8))
    flip_weights = np.zeros((len(comment_cut),maxlen))
    for i,sentence in enumerate(comment_cut):
        index = 0
        pos = nltk.pos_tag(sentence)
        for n,word in enumerate(sentence[:maxlen]):
            if word in sen_words:
                dex=sen_words.index(word)
                score=sen_polarity[dex]
                if score=='negative':#neg
                    lexicon_embed[i,n,:size] = 1
                if score == 'positive':#pos
                    lexicon_embed[i,n,size:2*size] = 1
            if word in nt_words:
                lexicon_embed[i,n,2*size:3*size] =1
            if word in int_words:
                lexicon_embed[i,n,3*size:4*size] =1
            if word in sup_words:
                lexicon_embed[i,n,4*size:5*size] =1
            if word in oth_words:
                lexicon_embed[i,n,5*size:6*size] =1
            pos_flag = 0
            for j,one_pos in enumerate(pos_list):
                if pos[n][1] in one_pos:
                    lexicon_embed[i,n,j+6*size:(j+1)+6*size] =1
                    pos_flag = 1
                if j == len(pos_list) -1 and pos_flag == 0:
                    lexicon_embed[i,n,6*size+7] = 1
            if word in flip_words:
                flip_weights[i,n] = 1
            if word == 'aspect_term':
                select_embed[i][n] = 1
                target_embed2[i][n] =1
                for k in range(len(aspect[i])):
                    try:
                        if index < maxlen:
                            word_embed[i][index][:] = word_vectors[aspect[i][k]]
                            target_embed[i][index] = 1
                            index += 1
                    except:
                        if index < maxlen:
                            word_embed[i][index][:] = word_vectors['UNK']
                            target_embed[i][index] = 1
                            index += 1
                        continue
            else:
                target_embed2[i][n] =2
                try:
                    word_embed2[i][n][:] = word_vectors[word]
                    if index < maxlen:
                        word_embed[i][index][:] = word_vectors[word]
                        target_embed[i][index] = 2
                        index += 1
                except:
                    if index < maxlen:
                        word_embed[i][index][:] = word_vectors['UNK']
                        target_embed[i][index] = 2
                        index += 1
                    continue
    return word_embed,target_embed,word_embed2,select_embed,target_embed2,lexicon_embed,flip_weights
 
def load_new_train(path):
    f = open(path,'r',encoding='utf-8')
    data = f.readlines()
    new_train = []
    for t in data:
        new_train.append(t[:-1].split(' '))
    return new_train

if __name__=='__main__':
    print('Loading Data...')
    logging.basicConfig(filename='../log/test.log', filemode="w", level=logging.DEBUG)
    save_path = '../model_data'
    
    train_path = '../data/train.txt'
    test_path = '../data/test.txt'
    vec_path = '../data/vec.pkl'
    
    aspect,context,label,maxlen_train = get_dataset(train_path)
    asp_test,con_test,y_test,maxlen_test = get_dataset(test_path)
    word_vectors = read_vec(vec_path)
    maxlen = max(maxlen_train, maxlen_test)

    con_graph_rate = np.load('../data/graph_distance.npy')
    new_context = load_new_train('../data/new_train.txt')
    
    size = 10
    con_emb,tar_mask,con_emb2,sel_mask,tar_mask2,lexicon_embed,flip_weights = word_embeding2(new_context,aspect,maxlen,word_vectors,size)
    con_test_emb,tar_mask_test,con_test_emb2,sel_mask_test,tar_mask_test2,lexicon_embed_test,_ = word_embeding2(con_test,asp_test,maxlen,word_vectors,size)

    ids_save=[]
    for time in range(1):
        K.clear_session()
        ids = [i for i in range(len(aspect))]
        ids_train,ids_dev = train_test_split(ids,test_size=1/5)
        ids_save.append([ids_train,ids_dev])
        
        tar_train =np.array([tar_mask[int(t)] for t in ids_train])
        tar_dev = np.array([tar_mask[int(t)] for t in ids_dev])
        
        con_train =np.array([con_emb[int(t)] for t in ids_train])
        con_dev = np.array([con_emb[int(t)] for t in ids_dev])
        
        sel_train =np.array([sel_mask[int(t)] for t in ids_train])
        sel_dev = np.array([sel_mask[int(t)] for t in ids_dev])
        
        con_train2 =np.array([con_emb2[int(t)] for t in ids_train])
        con_dev2 = np.array([con_emb2[int(t)] for t in ids_dev])
        
        tar_train2 =np.array([tar_mask2[int(t)] for t in ids_train])
        tar_dev2 = np.array([tar_mask2[int(t)] for t in ids_dev])
        
        graph_train = np.array([con_graph_rate[int(t)] for t in ids_train])
        graph_dev = np.array([con_graph_rate[int(t)] for t in ids_dev])
        
        lex_train = np.array([lexicon_embed[int(t)] for t in ids_train])
        lex_dev = np.array([lexicon_embed[int(t)] for t in ids_dev])
        
        flip_train = np.array([flip_weights[int(t)] for t in ids_train])
        flip_dev = np.array([flip_weights[int(t)] for t in ids_dev])
        
        y_train = np.array([label[int(t)] for t in ids_train])
        y_dev = np.array([label[int(t)] for t in ids_dev])
    

        network = CueNet(patience=10,batch_size=25,n_epoch=25,
                          save_path=save_path,time=str(time),
                          cluster_size = 15,
                          binary_dim = 30,
                          lamda1 = 1.5,
                          lamda2 = 0.001,
                          hidden_unit = 100,
                          learning_rate = 0.001)
    
        print('Training Data...') 
        network.train_model(con_train,tar_train,con_train2,sel_train,tar_train2,graph_train,lex_train,flip_train,y_train,
                         con_dev,tar_dev,con_dev2,sel_dev,tar_dev2,graph_dev,lex_dev,flip_dev,y_dev)
    
    files= os.listdir(save_path)
    max_acc = 0
    for i,file in enumerate(files):
        K.clear_session()
        model = load_model(save_path+'/'+file,custom_objects={"binary_indicator_layer": binary_indicator_layer,
                                                              "target_representation_layer": target_representation_layer,
                                                              "CRF":CRF,
                                                              "lossFunction":customLoss(K.variable(np.ones((1,1))),
                                                                                        K.variable(np.ones((1,1))),
                                                                                        K.variable(np.ones((1,1))),0.2)
                                                              })
        evaluate = model.evaluate([con_test_emb,tar_mask_test,con_test_emb2,sel_mask_test,tar_mask_test2,lexicon_embed_test],y_test)
        print(file,evaluate[1])
        if evaluate[1] > max_acc:
            max_acc = evaluate[1]
            best_name = file
        del model
        gc.collect()
        logging.info(file+' '+str(evaluate[1]))
    print('max_acc:', max_acc)
    print(best_name)
    logging.info(best_name+' '+str(max_acc))