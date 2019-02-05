import keras.layers as nn_layers
from keras.models import Sequential
from keras.datasets import mnist
import tensorflow as tf
import os

import utils
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(".","images","60 Images")

imgs,labels = utils.load_images(path)

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)