# -*- coding: utf-8 -*-
from keras import backend as K
import numpy as np
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
from keras.layers.core import activations
from keras.models import Model
from keras.layers.core import Dense,Activation,Masking
from keras.layers import Input,Flatten,RepeatVector,Permute,subtract,multiply,add,concatenate,Embedding
import tensorflow as tf


class binary_indicator_layer(Layer):
    
    def __init__(self,
                 units,
                 activation='tanh',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        
        super(binary_indicator_layer, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.units = units

        
    def build(self,input_shape):
        self.w1 = self.add_weight(name='kernel', 
                                      shape=(1,self.units),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.w2 = self.add_weight(name='kernel', 
                                      shape=(1,self.units),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.built = True

    def call(self, inputs):
        
        zeros = K.zeros((1,self.units))
        weight = concatenate([zeros,self.w1],axis=0)
        weight = concatenate([weight,self.w2],axis=0)
        
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
            
        tar_embed = K.gather(weight,inputs)
        return  tar_embed

    def compute_output_shape(self,input_shape):
        return input_shape + (self.units,)
    
    def get_config(self):
        config = {'units': self.units}
        base_config = super(binary_indicator_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))