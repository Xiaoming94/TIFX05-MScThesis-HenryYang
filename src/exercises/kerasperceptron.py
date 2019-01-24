'''
kerasperceptrion.py
Author: Henry Yang

This file is a Keras example i implemented on my own
Without following any guide, just as en exercise to understand keras
'''

import tensorflow as tf
from keras.models import Sequential
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import keras.layers as nn_layers
import numpy as np
import matplotlib.pyplot as plt

def setup_keras():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

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


def generate_data(count=50000, ratio=0.5):
    pos_size = int(count * ratio)
    neg_size = count - pos_size

    hx,hy = calc_henon(0,0,pos_size)
    pos_data = np.transpose(np.array([hx,hy]))
    neg_data = np.transpose(np.array([-hx,hy-0.2]))

    print(pos_data.shape)

    data = np.concatenate([pos_data,neg_data],axis=0)
    labels = np.zeros(count) - 1
    labels[0:pos_size] = 1
    return data, labels

def bernhard_energy(y_true, y_pred):
    s = K.sum(K.square(y_true - y_pred))
    return (1/2) * s

train_data_count = 100000
val_data_count = 10000

x_train,y_train = generate_data(train_data_count)
x_val, y_val = generate_data(val_data_count)

pos_x = x_val[0:int(val_data_count*0.5),:]
neg_x = x_val[int(val_data_count*0.5):,:]

plt.scatter(neg_x[:,0],neg_x[:,1],color="red")
plt.scatter(pos_x[:,0],pos_x[:,1],color="blue",marker="*")

plt.show()

print("Start creating neural Network")

setup_keras()

network_layers = [
    nn_layers.Dense(64,input_shape=(2,),activation="relu"),
    nn_layers.Dense(32,activation="relu"),
    nn_layers.Dense(1,activation="tanh")
]

model = Sequential(network_layers)
model.compile(optimizer='adam', loss=bernhard_energy, metrics=['accuracy'])
model.fit(x_train,y_train,verbose=1,validation_data=(x_val,y_val),epochs=50)

