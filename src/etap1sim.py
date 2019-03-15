import tensorflow as tf
import os
from time import time

import utils
import numpy as np
import matplotlib.pyplot as plt
import ANN as ann


trials = 20
mnist_acc = np.zeros(trials)
imgXM_acc = np.zeros(trials)
imgOB_acc = np.zeros(trials)
combined_acc = np.zeros(trials)
num_epochs = 4

network_model = """
{
    "input_shape" : [28,28,1],
    "layers" : [
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
            "type" : "MaxPooling2D",
            "pool_size" : [2,2],
            "strides" : [2,2]
        },
        {
            "type" : "Flatten"
        },
        {
            "type" : "Dense",
            "units" : 32,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax"
        }
    ]
}
"""

def test_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    correct = np.equal(np.argmax(predictions,1),np.argmax(test_labels,1))
    accuracy = np.mean(correct)
    return accuracy

utils.setup_gpu_session()

pathXM = os.path.join(".","images","XiaoMing_Digits")
pathOB = os.path.join(".","images","60 Images")

imgsXM,labelsXM = utils.load_image_data(pathXM)
imgsOB,labelsOB = utils.load_image_data(pathOB)

xtrain,ytrain,xtest,ytest = utils.load_mnist(normalize=True)
xtrain,xtest = xtrain.reshape(60000,28,28,1),xtest.reshape(10000,28,28,1)

img_size = imgsXM.shape[0]
ximgXM = imgsXM.reshape(img_size,28,28,1)
ximgXM = utils.normalize_data(ximgXM)
yimgXM = utils.create_one_hot(labelsXM)
img_size = imgsOB.shape[0]
ximgOB = imgsOB.reshape(img_size,28,28,1)
ximgOB = utils.normalize_data(ximgOB)
yimgOB = utils.create_one_hot(labelsOB)
imgs_combined = np.concatenate((ximgXM,ximgOB))
labels_combined = np.concatenate((yimgXM,yimgOB))

for i in range(trials):

    t_xtrain,t_ytrain,xval,yval = utils.create_validation(xtrain,ytrain)

    
    print("starting trial %s" % (i+1))
    ## Realizing and training the data
    ## uses TensorBoard to monitor the progress
    model = ann.parse_model_js(network_model)

    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(t_xtrain,t_ytrain,verbose=1, validation_data=(xval,yval),epochs=num_epochs)

    mnist_acc[i] = test_model(model, xtest,ytest)
    imgXM_acc[i] = test_model(model, ximgXM, yimgXM)
    imgOB_acc[i] = test_model(model, ximgOB, yimgOB)
    combined_acc[i] = test_model(model, imgs_combined, labels_combined)
    

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
plt.title("accuracy test of Combined Dataset")
plt.xlabel("trials")
plt.ylabel("accuracy")
imgs_line, = plt.plot(Xaxis, combined_acc,color="blue",marker="x",linestyle="-")
imgs_avg_line, = plt.plot(Xaxis, np.ones(trials) * np.mean(combined_acc), color="red",linestyle="--")
plt.legend((imgs_line, imgs_avg_line), ("IMGS Accuracy", "IMGS Average Accuracy"))

plt.figure()
plt.subplot(2,1,1)
plt.title("accuracy test of Olek's Digits")
plt.xlabel("trials")
plt.ylabel("accuracy")
mnist_line, = plt.plot(Xaxis, imgOB_acc,color="blue",marker="x",linestyle="-")
mnist_avg_line, = plt.plot(Xaxis, np.ones(trials) * np.mean(imgOB_acc), color="red",linestyle="--")
plt.legend((mnist_line, mnist_avg_line),("Oleksandr's Digits Accuracy", "Oleksandr's Average Accuracy"))
plt.subplot(2,1,2)
plt.title("accuracy test of Henry's Digits")
plt.xlabel("trials")
plt.ylabel("accuracy")
imgs_line, = plt.plot(Xaxis, imgXM_acc,color="blue",marker="x",linestyle="-")
imgs_avg_line, = plt.plot(Xaxis, np.ones(trials) * np.mean(imgXM_acc), color="red",linestyle="--")
plt.legend((imgs_line, imgs_avg_line), ("Henry's Digits Accuracy", "Henry's Average Accuracy"))

plt.show()