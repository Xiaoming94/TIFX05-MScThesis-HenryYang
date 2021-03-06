import utils
import numpy as np
from matplotlib import pyplot as plt
import os
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
    sides = 280
    return cv2.resize(img, (sides,sides))

xtrain, ytrain, xtest, ytest = utils.load_mnist()
chunksize = 1000
mnist_total_size = xtest.shape[0]
partitions = int(mnist_total_size/chunksize)
taus = []

for i in range(partitions):
    print("Starting Loop")
    with Pool(6) as p:
        mnist_batch = xtest[i*chunksize:(i+1)*chunksize]
        mnist_batch_big = np.array(p.map(resize_helper, mnist_batch))

    taus.append(utils.calc_linewidth(mnist_batch_big))
    print("Calculated Line width of %s Digits of %s" % ((i+1) * chunksize, mnist_total_size))

print(np.mean(taus))
