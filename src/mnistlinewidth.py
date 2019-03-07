import utils
import numpy as np
from matplotlib import pyplot as plt
import os
import ANN as ann
import cv2
from multiprocessing import Pool

network_model1 = """
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
            "type" : "BatchNormalization",
            "axis" : -1
        },
        {
            "type" : "Conv2D",
            "units" : 32,
            "kernel_size" : [3,3],
            "activation" : "relu"
        },
        {
            "type" : "BatchNormalization",
            "axis" : -1
        },
        {
            "type" : "MaxPooling2D",
            "pool_size" : [2,2],
            "strides" : [2,2]
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

def resize_helper(img):
    sides = 400
    return cv2.resize(img, (sides,sides))

xtrain, ytrain, xtest, ytest = utils.load_mnist(normalize=False)
xtrain = xtrain.reshape(60000, 28, 28,1)
xtest = xtest.reshape(10000, 28, 28,1)
xtrain, ytrain, xval, yval = utils.create_validation(xtrain,ytrain,1/6)

mnist_total = np.concatenate((xtrain,xtest))
chunksize = 10000
partitions = int(mnist_total.shape[0]/chunksize)
taus = []

for i in range(partitions):
    print("Starting Loop")
    with Pool(4) as p:
        mnist_batch = mnist_total[i*chunksize:(i+1)*chunksize]
        mnist_batch_big = np.array(p.map(resize_helper, mnist_batch))

    taus.append(utils.calc_linewidth(mnist_batch_big))
    print("Calculated Line width of %s Digits" % ((i+1) * chunksize))

print(np.mean(taus))
