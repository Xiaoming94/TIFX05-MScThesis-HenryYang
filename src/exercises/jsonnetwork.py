'''
jsonnetwork.py
Author: Henry Yang (940503-1056)

This is an exercise where A network is built from parsing a Json config file
'''

import tensorflow as tf
from keras.models import Sequential, Model
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import keras.layers as nn_layers
import numpy as np
import matplotlib.pyplot as plt
import json
from keras.datasets import mnist

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
    ]
}
"""
network_model2 = """
{
    "input_shape" : [28,28,1],
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
            "units" : 10,
            "activation" : "softmax"
        }
    ]
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
        input_layer = nn_layers.Input(shape=tuple(config["input_shape"]))
        net_struct = input_layer
        for lc in layers_configs:
            print(lc)
            net_struct = parse_layer_conf(lc)(net_struct)
        return input_layer, net_struct
        
    config = json.loads(config_json)
    input_layer, net_struct = build_layers(config)
    model = Model(inputs=input_layer,outputs=net_struct)

    return model

def parse_layer_conf(layer_conf):
    def build_conv2D(x,layer_conf):

        x = nn_layers.Conv2D(layer_conf["units"],tuple(layer_conf["kernel_size"]))(x)
        x = nn_layers.BatchNormalization(axis=-1)(x)
        x = nn_layers.Activation(layer_conf["activation"])(x)
        return x
    layer_type = layer_conf["type"]
    if layer_type == "Dense":
        return nn_layers.Dense(layer_conf["units"],activation=layer_conf["activation"])
    if layer_type == "Conv2D":
        return lambda x: build_conv2D(x,layer_conf)
    if layer_type == 'Flatten':
        return nn_layers.Flatten()
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

def process_data(data_x, data_y):
    # Normalize the data
    ndata_x = data_x * (1/255)

    # onehot encode labels
    length_y = data_y.size
    labels_y = np.zeros([length_y,10])
    for i,j in zip(list(range(length_y)),data_y.astype(np.int32)):
        labels_y[i,j] = 1
    
    return ndata_x,labels_y

#xtrain,ytrain,xtest,ytest = generate_data(count=100000)
setup_keras()
#model = build_model(network_model1)
#model.compile(optimizer='adam', loss=bernhard_energy, metrics=['accuracy'])
#model.fit(xtrain,ytrain,verbose=1,validation_data=(xtest,ytest),epochs=50)

## LOADING THE MNIST DATA
(x_train,y_train),(x_test,y_test) = mnist.load_data()

## RESHAPING THE DATA to suitable forms for the CNN
x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)

## Processing the data, and isolates a validation set
ntrain_x, oh_trainy = process_data(x_train,y_train)

randidx = np.random.permutation(y_train.size)
val_x,val_y = ntrain_x[randidx[0:10000]],oh_trainy[randidx[0:10000]]
train_x,train_y = ntrain_x[randidx[10000:]],oh_trainy[randidx[10000:]]

model = build_model(network_model2)
print("failing to build")
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_x,train_y,verbose=1, validation_data=(val_x,val_y),epochs=5)

## Testing the Model
