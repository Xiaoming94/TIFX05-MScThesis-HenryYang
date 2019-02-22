'''
jsonnetwork.py
Author: Henry Yang (940503-1056)

This is an exercise where A network is built from parsing a Json config file
'''

import tensorflow as tf
from keras.models import Sequential
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import keras.layers as nn_layers
import numpy as np
import matplotlib.pyplot as plt
import json

network_model1 = """
{
    "input_shape" : [2],
    "layers" : [
        {
            "type" : "Dense",
            "units" : 32,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 64,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 1,
            "activation" : "tanh"
        }
    ],
    "loss" : "bernhard_energy"
}
"""
network_model2 = """
{
    "input_shape" : [28,28,1]
    "layers" : [
        {
            "type" : "Conv2D",
            "units" : 64,
            "kernel_size" : [3,3],
            "activation" : "relu"
        },
        {
            "type" : "Conv2D",
            "units" : 32,
            "kernel_size" : [3,3],
            "activation" : "relu"
        },
        {
            "type" : "Flatten"
        },
        {
            "type" : "Dense",
            "units" : "10",
            "activation" : "softmax"
        }
    ],
    "loss" : "categorical_crossentropy"
}
"""

def setup_keras():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

def build_model(config_json):
    def build_layers(config):
 
        layers_configs = config["layers"]
        layers = [nn_layers.InputLayer(input_shape=tuple(config["input_shape"]))]
        for lc in layers_configs:
            layers += parse_layer_conf(lc)
        
        return layers
        
    config = json.loads(config_json)
    layers = build_layers(config)
    print(layers)
    model = Sequential(layers)
    model.compile(optimizer='adam', loss=loss_function(config['loss']), metrics=['accuracy'])
    return model

def loss_function(function_name):
    if function_name == "bernhard_energy":
        return bernhard_energy
    else:
        return function_name

def parse_layer_conf(layer_conf):
    layer_type = layer_conf["type"]
    if layer_type == "Dense":
        return [nn_layers.Dense(
            layer_conf["units"],
            activation = layer_conf["activation"]         
        )]
    if layer_type == "Conv2D":
        return [
            nn_layers.Conv2D(
                layer_conf["units"],
                layer_conf["kernel_size"]
            ),
            nn_layers.BatchNormalization(axis=-1),
            nn_layers.Activation(layer_conf["activation"])
        ]
    else:
        return None
    


def calc_henon(x0,y0,steps,a=1.4, b=0.3):
    x_vec = np.zeros(3+steps)
    y_vec = np.zeros(3+steps)

    x = x0
    y = y0

    for t in range(3+steps):
        x_vec[t] = x
        y_vec[t] = y
        xn = 1 - a * x * x + y
        yn = b * x
        x = xn
        y = yn

    return x_vec[3:], y_vec[3:]


def generate_data(count=60000, testsize=10000 ,ratio=0.5):
    pos_size = int(count * ratio)
    neg_size = count - pos_size

    hx,hy = calc_henon(0,0,pos_size)
    pos_data = np.transpose(np.array([hx,hy]))
    neg_data = np.transpose(np.array([-hx,hy-0.2]))

    ran_indices = np.random.permutation(int(count*ratio))
    test_pos_data = pos_data[ran_indices[0:int(testsize * ratio)]]
    test_neg_data = neg_data[ran_indices[0:int(testsize * ratio)]]
    train_pos_data = pos_data[ran_indices[int(testsize * ratio):]]
    train_neg_data = neg_data[ran_indices[int(testsize * ratio):]]
    
    Xtrain = np.concatenate([train_pos_data,train_neg_data],axis=0)
    Xtest = np.concatenate([test_pos_data,test_neg_data],axis=0)
    Ytrain = np.zeros(count-testsize) - 1
    Ytrain[0:int((count-testsize)*ratio)] = 1
    Ytest = np.zeros(testsize) - 1
    Ytest[0:int(testsize * ratio)] = 1
    return Xtrain, Ytrain, Xtest, Ytest

def bernhard_energy(y_true, y_pred):
    s = K.sum(K.square(y_true - y_pred))
    return (1/2) * s

xtrain,ytrain,xtest,ytest = generate_data(count=100000)
setup_keras()
model = build_model(network_model1)
model.fit(xtrain,ytrain,verbose=1,validation_data=(xtest,ytest),epochs=50)
