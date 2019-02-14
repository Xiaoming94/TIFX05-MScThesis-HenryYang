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

trials = 100
mnist_acc = np.zeros(trials)
img_acc = np.zeros(trials)
num_epochs = 5



def test_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    correct = np.equal(np.argmax(predictions,1),np.argmax(test_labels,1))
    accuracy = np.mean(correct)
    return accuracy

utils.setup_gpu_session()

path = os.path.join(".","images","60 Images")

imgs,labels = utils.load_images(path)

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)

x_train = utils.normalize(x_train)
y_train = utils.create_one_hot(y_train)

x_test = utils.normalize(x_test)
y_test = utils.create_one_hot(y_test)

(count,features) = imgs.shape

imgs = imgs.reshape(count,28,28,1)

x_img = utils.normalize(imgs)
y_img = utils.create_one_hot(labels)

for i in range(trials):

    randidx = np.random.permutation(int(y_train.size / 10))
    val_x,val_y = x_train[randidx[0:10000]],y_train[randidx[0:10000]]
    train_x,train_y = x_train[randidx[10000:]],y_train[randidx[10000:]]

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

    ## Realizing and training the data
    ## uses TensorBoard to monitor the progress
    model = Sequential(layers_list)

    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_x,train_y,verbose=1, validation_data=(val_x,val_y),epochs=num_epochs)

    mnist_acc[i] = test_model(model, x_test,y_test)
    img_acc[i] = test_model(model, x_img, y_img)

Xaxis = np.arange(trials)

plt.figure()
plt.subplot(2,1,1)
plt.title("accuracy test of MNIST")
plt.xlabel("trials")
plt.ylabel("accuracy")
mnist_line, = plt.plot(Xaxis, mnist_acc,color="blue",marker="x",linestyle="-")
mnist_avg_line, = plt.plot(Xaxis, np.ones(trials) * np.mean(mnist_acc), color="red",linestyle="--")
plt.legend((mnist_line, mnist_avg_line),("MNIST Accuracy", "MNIST Average Accuracy"))
plt.subplot(2,1,2)
plt.title("accuracy test of own Imgs")
plt.xlabel("trials")
plt.ylabel("accuracy")
imgs_line, = plt.plot(Xaxis, img_acc,color="blue",marker="x",linestyle="-")
imgs_avg_line, = plt.plot(Xaxis, np.ones(trials) * np.mean(img_acc), color="red",linestyle="--")
plt.legend((imgs_line, imgs_avg_line), ("IMGS Accuracy", "IMGS Average Accuracy"))
plt.show()