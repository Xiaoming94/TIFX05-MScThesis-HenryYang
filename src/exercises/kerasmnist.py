from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.backend.tensorflow_backend import set_session
import keras.layers as nn_layers
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


def process_data(data_x, data_y):
    # Normalize the data
    ndata_x = data_x * (1/255)

    # onehot encode labels
    length_y = data_y.size
    labels_y = np.zeros([length_y,10])
    for i,j in zip(list(range(length_y)),data_y.astype(np.int32)):
        labels_y[i,j] = 1
    
    return ndata_x,labels_y
    

(x_train,y_train),(x_test,y_tesst) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)

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

ntrain_x, oh_trainy = process_data(x_train,y_train)

randidx = np.random.permutation(y_train.size)
val_x,val_y = ntrain_x[randidx[0:10000]],oh_trainy[randidx[0:10000]]
train_x,train_y = ntrain_x[randidx[10000:]],oh_trainy[randidx[10000:]]

model = Sequential(layers_list)

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_x,train_y, validation_data=(val_x,val_y),epochs=3)