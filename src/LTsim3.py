import utils
from utils import digitutils as dutils
import ANN as ann
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cv2

network_model1 = """
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
            "units" : 200,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 200,
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

def test_model(model, data, labels):
    data = utils.normalize_data(data)
    data_size = data.shape[0]
    data = data.reshape(data_size, 784)
    return ann.test_model(model, data, labels, metric = "entropy")

def thickness_sim(model, data):
    labels = utils.create_one_hot(data['labels'].astype('uint'))
    thicknesses = list(data.keys())[:-1]
    numdata = len(thicknesses)
    acc = np.zeros(numdata)
    for i,t in zip(range(numdata),thicknesses):
        acc[i] = test_model(model,data[t],labels)
    return acc

#mnist_linethickness = 66.97000583000295 ## Obtained from running mnistlinewidth.py file
mnist_linethickness = 14.095163376059986
# 93.62709087870702

epochs = 3

trials = 5

utils.setup_gpu_session()
acc_results_custom = []
acc_results_mnist = []
test_digits = utils.load_processed_data("combined_testing_data")

for i in range(trials):
    print("====== START TRAINING NEURAL NETWORK MODEL ======")
    xtrain,ytrain,xtest,ytest = utils.load_mnist()
    xtrain,xtest = xtrain.reshape(60000, 784),xtest.reshape(10000, 784)
    xtrain,ytrain,xval,yval = utils.create_validation(xtrain,ytrain,1/6)
    model = ann.parse_model_js(network_model2)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    model.fit(xtrain,ytrain, verbose=1, validation_data=(xval,yval),epochs=epochs)
    print("====== CALCULATING MNIST ACCURACY ======")
    mnist_accuracy = ann.test_model(model, xtest, ytest, "entropy")
    comb_acc = thickness_sim(model,test_digits)
    acc_results_mnist.append(mnist_accuracy)
    acc_results_custom.append(comb_acc)
    print("Finished trial %s" % (i + 1))

#print(acc_results_custom)
mnist_accuracy = np.mean(np.array(acc_results_mnist))
comb_tau = list(test_digits.keys())[:-1]
acc_results_custom = np.stack(acc_results_custom)
comb_acc = np.mean(acc_results_custom,axis=0)
print(comb_acc)

print("plotting:")
plt.figure()
p_mnist_tau, = plt.plot([mnist_linethickness,mnist_linethickness],[-0.1,1.0],linestyle="--",color="black")
p_mnist_acc, = plt.plot([4.8,25],[mnist_accuracy,mnist_accuracy],linestyle="--",color="darkblue")
custom_trial = plt.plot(comb_tau,acc_results_custom.transpose(),color="lightgrey")
plt.grid()
comb_line, = plt.plot(comb_tau,comb_acc,color="red")
plt.xlabel("Line Thickness")
plt.ylabel("Entropy")
plt.title("Entropy change over Line thickness")
plt.legend((custom_trial[0], p_mnist_tau, p_mnist_acc, comb_line),("combined digits trials","mnist_linethickness","mnist_entropy","Combined Digits Mean"))
plt.show()
