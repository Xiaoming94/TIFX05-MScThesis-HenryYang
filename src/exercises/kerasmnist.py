'''
kerasmnist.py 
Author: Henry Yang

This is just a tutorial file, almost irrelevant to the thesis work.
I programmed this file with the purpose of learning Keras, and made sure that my environment works

Most of this file is inspired by the tutorial on 
https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
Add the keras documentation

This file also includes some fixes so that the program can make us of the discrete NVIDIA GPU
'''

## Importing Libs

from keras.datasets import mnist
from time import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard
import keras.layers as nn_layers
import numpy as np
import tensorflow as tf

## Session fix to have it working with the GPU

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

'''
process_data(data_x, data_y)

This function takes as input the data sets to be processed.
It normalizes each sample point to be continuous in the range [0,1]

It also takes the labels from mnist dataset and creates one hot vectors from it.
Returns the processed data, and one hot vectors without side effects
'''
def process_data(data_x, data_y):
    # Normalize the data
    ndata_x = data_x * (1/255)

    # onehot encode labels
    length_y = data_y.size
    labels_y = np.zeros([length_y,10])
    for i,j in zip(list(range(length_y)),data_y.astype(np.int32)):
        labels_y[i,j] = 1
    
    return ndata_x,labels_y
    
## LOADING THE MNIST DATA
(x_train,y_train),(x_test,y_tesst) = mnist.load_data()

## RESHAPING THE DATA to suitable forms for the CNN
x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)

## SPECIFYING THE LAYERS IN THE NEURAL NETWORK MODEL
layers_list = [
    nn_layers.Conv2D(64, (3,3), input_shape=(28,28,1)),
    nn_layers.BatchNormalization(axis=-1),
    nn_layers.Activation('relu'),
    nn_layers.Conv2D(32, (3,3)),
    nn_layers.BatchNormalization(axis=-1),
    nn_layers.Activation('relu'),
    nn_layers.Flatten(),
    nn_layers.Dense(10, activation='softmax')
]


## Processing the data, and isolates a validation set
ntrain_x, oh_trainy = process_data(x_train,y_train)

randidx = np.random.permutation(y_train.size)
val_x,val_y = ntrain_x[randidx[0:10000]],oh_trainy[randidx[0:10000]]
train_x,train_y = ntrain_x[randidx[10000:]],oh_trainy[randidx[10000:]]

## Realizing and training the data
## uses TensorBoard to monitor the progress
model = Sequential(layers_list)

tensorboard=TensorBoard(log_dir="/tmp/tensorboard/{}".format(time()))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_x,train_y,verbose=1, validation_data=(val_x,val_y),epochs=10,callbacks=[tensorboard])