# -*- coding: utf-8 -*-
from keras import backend as K
import numpy as np
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
from keras.layers.core import activations
from keras.models import Model
from keras.layers.core import Dense,Activation,Masking
from keras.layers import Input,Flatten,RepeatVector,Permute,subtract,multiply,add,concatenate,Embedding,Reshape,Add,Lambda
from keras.layers import Multiply,dot


class target_representation_layer(Layer):
    
    def __init__(self,
                 units,
                 classes,
                 activation='tanh',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        
        super(target_representation_layer, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.units = units
        self.classes = classes

        
    def build(self,input_shape):
        self.weight = self.add_weight(name='kernel', 
                                      shape=(self.classes,self.units),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.built = True

    def call(self, inputs):
        Mydot = Lambda(lambda x: K.batch_dot(x[0],x[1]))
        
        tar_vec = K.dot(inputs[1],self.weight)
        
        tar_vec = RepeatVector(1)(tar_vec)
        
        sel_mask = RepeatVector(1)(inputs[0])
        sel_mask = Permute([2,1])(sel_mask)
        
        sel_embed = Mydot([sel_mask,tar_vec])

        return  sel_embed

    def compute_output_shape(self,input_shape):
        return input_shape[0] + (self.units,)
    
    def get_config(self):
        config = {'units': self.units,'classes': self.classes}
        base_config = super(target_representation_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))