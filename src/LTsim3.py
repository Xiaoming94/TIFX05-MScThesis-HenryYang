import utils
from utils import digitutils as dutils
import ANN as ann
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cv2

network_model = """
{
    "input_shape" : [28,28,1],
    "layers" : [
        {
            "type" : "Conv2D",
            "units" : 16,
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

network_model2 = '''
{
    "input_shape" : [784],
    "layers" : [
        {
            "type" : "Dense",
            "units" : 256,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 128,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 256,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax"
        }
    ]
}
'''

def scale(img,side = 400):
    return cv2.resize(img, (side,side))

def scale_down(data):
    #with Pool(4) as p:
    #    unpadded = p.map(dutils.unpad_img, data)
    #
    #unpadded = list(map(
    #    lambda img: dutils.unpad_img(img), data
    #))
    #downscaled = list(map(
    #    lambda img: dutils.resize_image(img,20), unpadded
    #))
#
    #downscaled = list(map(
    #    lambda img: dutils.center_box_image(img, 20, 4), downscaled
    #))
    downscaled = list(map(
        lambda img: cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC), data
    ))

    return np.array(downscaled)

def test_model(model, data, labels):
    data = utils.normalize_data(data)
    data_size = data.shape[0]
    data = data.reshape(data_size, 784)
    return ann.test_model(model, data, labels, metric = "accuracy")

def change_thickness(data, factor):
    #print("===== CHANGING LINETHICKNESS =====")
    new_data = list(map(
        lambda img: dutils.change_linewidth(img, factor), data   
    ))
    return np.array(new_data)

def thickness_sim(model, data):
    labels = utils.create_one_hot(data["labels"])
    thicknesses = list(data.keys())[:-1]
    numdata = len(thicknesses)
    acc = np.zeros(numdata)
    for i,t in zip(range(numdata),thicknesses):
        acc[i] = test_model(model,data[t],labels)
    return acc, thicknesses




#mnist_linethickness = 66.97000583000295 ## Obtained from running mnistlinewidth.py file
mnist_linethickness = 67.14082725553595
# 93.62709087870702

epochs = 5

xm_digits_path = os.path.join(".","images","XiaoMing_Digits")
ob_digits_path = os.path.join(".","images","60 Images")

xtrain,ytrain,xtest,ytest = utils.load_mnist()
xtrain,xtest = xtrain.reshape(60000,784),xtest.reshape(10000,784)
xtrain,ytrain,xval,yval = utils.create_validation(xtrain,ytrain,1/6)
utils.setup_gpu_session()
print("====== START TRAINING NEURAL NETWORK MODEL ======")
model = ann.parse_model_js(network_model2)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(xtrain,ytrain, verbose=1, validation_data=(xval,yval),epochs=epochs)
print("====== CALCULATING MNIST ACCURACY ======")
mnist_accuracy = ann.test_model(model, xtest, ytest, "accuracy")

print("====== LOAD CUSTOM DIGITS FROM OLEKSANDR AND HENRY ======")
xm_data = utils.load_processed_data("xiaoming_digits")
ob_data = utils.load_processed_data("Oleks_digits")
combined_data = utils.load_processed_data("combined_digits")
combined_data["labels"] = np.concatenate([xm_data["labels"],ob_data["labels"]])

print("Analysing Henry's Digits")
xm_acc, xm_tau = thickness_sim(model,xm_data)
print("== FINISH ==\n")
print("Analysing Oleksandr's Digits")
ob_acc, ob_tau = thickness_sim(model,ob_data)
print("== FINISH ==\n")
print("Analysing COmbined digit set")
comb_acc, comb_tau = thickness_sim(model,combined_data)
print("Finish \n")
print("plotting:")
plt.figure()
plt.plot([mnist_linethickness,mnist_linethickness],[0,1.2],linestyle="--")
plt.plot([0,68],[mnist_accuracy,mnist_accuracy],linestyle="--")
plt.grid()
xm_line, = plt.plot(xm_tau,xm_acc)
ob_line, = plt.plot(ob_tau,ob_acc)
comb_line, = plt.plot(comb_tau,comb_acc)
plt.xlabel("Line Thickness")
plt.ylabel("Accuracy")
plt.title("Accuracy change over Line thickness")
plt.legend((xm_line,ob_line,comb_line),("Henry's Digits","Oleksandr's Digits","Combined Digits"))
plt.show()
