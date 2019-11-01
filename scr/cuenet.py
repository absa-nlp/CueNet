# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers import LSTM ,GRU
from keras import callbacks
from keras.layers import Bidirectional,Input,concatenate,Flatten,RepeatVector,Lambda,TimeDistributed,add,multiply
from keras.layers.core import Dense, Dropout,Activation
from keras import backend as K
from keras.optimizers import Adam,RMSprop
from binary_indicator import binary_indicator_layer
from target_representation import target_representation_layer
from keras.losses import categorical_crossentropy
from keras_contrib.layers import CRF

class CueNet(object):
    def __init__(self,
                 patience=2,batch_size=32,save_path = None,time = None,
                 n_epoch=8,cluster_size =None ,binary_dim = None, 
                 lamda1 = None, lamda2 = None,
                 hidden_unit = None, learning_rate= 0.001):
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.patience = patience
        self.save_path = save_path
        self.time = time
        self.cluster_size = cluster_size
        self.binary_dim = binary_dim
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.hidden_unit = hidden_unit
        self.learning_rate = learning_rate
    
    def train_model(self,x1_train,x2_train,x3_train,x4_train,x5_train,x6_train,x8_train,x9_train,y_train,
                    x1_test,x2_test,x3_test,x4_test,x5_test,x6_test,x8_test,x9_test,y_test):

        Mydot = Lambda(lambda x: K.batch_dot(x[0],x[1]))
        crf = CRF(2, sparse_target=True,learn_mode ='marginal')
        Get_one = Lambda(lambda x: x[:,:,0])
        
        input1_shape = x1_train.shape[1:]
        input2_shape = x2_train.shape[1:]
        input3_shape = x3_train.shape[1:]
        input4_shape = x4_train.shape[1:]
        input5_shape = x5_train.shape[1:]
        input6_shape = x6_train.shape[1:]
        input8_shape = x8_train.shape[1:]
        input9_shape = x9_train.shape[1:]
        
        word = Input(shape=input1_shape)
        print('word:',np.shape(word))
        
        tar = Input(shape=input2_shape)
        
        word2 = Input(shape=input3_shape)
        print('word2:',np.shape(word2))
        sel = Input(shape=input4_shape)
        print('sel:',np.shape(sel))
        
        tar2 =  Input(shape=input5_shape)
        print('tar2:',np.shape(tar2))
        
        graph_dis = Input(shape=input6_shape)
        print('graph_dis:',np.shape(graph_dis))
        
        lexicon_emb = Input(shape=input8_shape)
        print('lexicon_emb:',np.shape(lexicon_emb))
        
        flip_weight = Input(shape=input9_shape)
        print('flip_weight:',np.shape(flip_weight))
        
        tar_mask = binary_indicator_layer(units=self.binary_dim)(tar)
        print('tar_mask:',np.shape(tar_mask))
        
        hidden = Bidirectional(GRU(units=self.hidden_unit,return_sequences=True,dropout=0.5))(word)
        AT1 = concatenate([hidden,tar_mask])
        print('AT1:',np.shape(AT1))
        
        att = Dense(1)(AT1)
        att = Flatten()(att)
        att_value = Activation('softmax')(att)
        print('att_value:',np.shape(att_value))
        
        att_vec = RepeatVector(1)(att_value)
        print('att_vec:',np.shape(att_vec))
        att_mul=Mydot([att_vec, hidden])
        print('att_mul:',np.shape(att_mul))
        
        at_done = Flatten()(att_mul)
        select_vec = Dropout(0.3)(at_done)
        select = Dense(self.cluster_size, activation='softmax')(select_vec)
        print('select:',np.shape(select))
        
        sel_embed = target_representation_layer(units=300,classes=self.cluster_size)([sel,select])
        print('sel_embed:',np.shape(sel_embed))
        
        new_embed = add([word2,sel_embed])
        
        tar2_mask = binary_indicator_layer(units=self.binary_dim)(tar2)
        
        con_emb = concatenate([new_embed,lexicon_emb])
        hidden2 = Bidirectional(GRU(units=self.hidden_unit,return_sequences=True,dropout=0.5))(con_emb)
        
        AT2 = concatenate([hidden2,tar2_mask])
        print('AT2:',np.shape(AT2))
        
        crf_out = crf(AT2)
        crf_out_1 = Get_one(crf_out)
        print('crf_out_1:',np.shape(crf_out_1))
        crf_out_trans = RepeatVector(1)(crf_out_1)
        
        att_mul2=Mydot([crf_out_trans, hidden2])
        print('att_mul2:',np.shape(att_mul2))
        
        at_done2 = Flatten()(att_mul2)
        x = Dropout(0.3)(at_done2)
        x = Dense(3)(x)
        y = Activation('softmax')(x)
        model = Model(inputs=[word,tar,word2,sel,tar2,graph_dis,lexicon_emb,flip_weight], outputs=y)
        

        RMS = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06)
        
        model.compile(loss=customLoss(crf_out_1,graph_dis,lamda1 = self.lamda1,lamda2 = self.lamda2),
                  optimizer=RMS, metrics=['accuracy'])
        saveBestModel = callbacks.ModelCheckpoint(self.save_path+'/model'+self.time+'.{epoch:04d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1, mode='auto',save_best_only=False)
        model.fit([x1_train,x2_train,x3_train,x4_train,x5_train,x6_train,x8_train,x9_train],y_train, batch_size=self.batch_size, epochs=self.n_epoch,verbose=2,
                  validation_data=([x1_test,x2_test,x3_test,x4_test,x5_test,x6_test,x8_test,x9_test],y_test),callbacks=[saveBestModel])
        
def customLoss(crf_weight=None, graph_dis=None, lamda1=0.2, lamda2=0.01):
    def lossFunction(y_true,y_pred):  
        loss1 = categorical_crossentropy(y_true, y_pred)
        loss2 =K.mean(K.mean(multiply([(K.ones_like(crf_weight) - crf_weight),graph_dis])))
        loss3 = K.sum(K.sum(crf_weight))
        loss = loss1 + lamda1 * loss2 + lamda2 * loss3
        return loss
    return lossFunction    
    
        