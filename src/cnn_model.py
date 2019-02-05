import keras.layers as nn_layers
from keras.models import Sequential
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import tensorflow as tf
import os
from time import time

import utils
import numpy as np
import matplotlib.pyplot as plt


utils.setup_gpu_session()
path = os.path.join(".","images","60 Images")

imgs,labels = utils.load_images(path)

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)

x_train = utils.normalize(x_train)
y_train = utils.create_one_hot(y_train)

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


randidx = np.random.permutation(int(y_train.size / 10))
val_x,val_y = x_train[randidx[0:10000]],y_train[randidx[0:10000]]
train_x,train_y = x_train[randidx[10000:]],y_train[randidx[10000:]]

## Realizing and training the data
## uses TensorBoard to monitor the progress
model = Sequential(layers_list)

tensorboard=TensorBoard(log_dir="/tmp/tensorboard/{}".format(time()))

model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_x,train_y,verbose=1, validation_data=(val_x,val_y),epochs=5, callbacks=[tensorboard])

## Testing the Model

x_test = utils.normalize(x_test)
y_test = utils.create_one_hot(y_test)

predictions = model.predict(x_test)

correct = np.equal(np.argmax(predictions,1),np.argmax(y_test,1))
accuracy = np.mean(correct)

print("The Accuracy on the Test set is: %s" % accuracy)

(count,features) = imgs.shape

imgs = imgs.reshape(count,28,28,1)

x_img = utils.normalize(imgs)
y_img = utils.create_one_hot(labels)

predictions = model.predict(x_img)

correct = np.equal(np.argmax(predictions,1),np.argmax(y_img,1))
accuracy = np.mean(correct)

print("The Accuracy on the Test set is: %s" % accuracy)