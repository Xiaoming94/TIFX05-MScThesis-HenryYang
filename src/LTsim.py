import utils
from utils import digitutils as dutils
import ANN as ann
import os
import numpy as np
import matplotlib.pyplot as plt

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
            "type" : "MaxPooling2D",
            "pool_size" : [2,2],
            "strides" : [2,2]
        },
        {
            "type" : "Flatten"
        },
        {
            "type" : "Dense",
            "units" : 16,
            "activation" : "relu"
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
def test_out_digits(model, data, labels):
    #print("===== TESTING THE CURRENT DIGITS =====")
    rescaled = list(map(dutils.unpad_img, data))
    rescaled = list(
        map(
            lambda img: dutils.center_box_image(dutils.resize_image(img, 20), 20, 4),rescaled
        )
    )
    testing_data = np.array(rescaled)
    testing_data = utils.normalize_data(testing_data)
    testing_data_size = testing_data.shape[0]
    return ann.test_model(model,testing_data.reshape(testing_data_size,28,28,1),labels)

def change_thickness(data, factor):
    #print("===== CHANGING LINETHICKNESS =====")
    new_data = list(map(
        lambda img: dutils.change_linewidth(img, factor), data   
    ))
    return np.array(new_data)

def thickness_sim(model, data, labels, ref_thickness):
    opt_r = 0
    epsilon = 2
    current_acc = test_out_digits(model, data, labels)
    current_tau = utils.calc_linewidth(data)
    metrics = [current_acc]
    tau = [current_tau]
    curr_data = data
    r = int(np.sign(ref_thickness - current_tau))
    new_r = r
    while abs(ref_thickness - current_tau) > epsilon:
        opt_r += r
        pre_acc = current_acc
        new_data = change_thickness(curr_data, r)
        current_acc = test_out_digits(model, new_data, labels)
        current_tau = utils.calc_linewidth(new_data)
        metrics.append(current_acc)
        tau.append(current_tau)
        curr_data = new_data
        new_r = int(np.sign(ref_thickness - current_tau))
        print(current_acc)
    return opt_r, metrics, tau, curr_data




mnist_linethickness = 66.97000583000295 ## Obtained from running mnistlinewidth.py file
epochs = 3

xm_digits_path = os.path.join(".","images","XiaoMing_Digits")
ob_digits_path = os.path.join(".","images","60 Images")

xtrain,ytrain,xtest,ytest = utils.load_mnist()
xtrain,xtest = xtrain.reshape(60000,28,28,1),xtest.reshape(10000,28,28,1)
xtrain,ytrain,xval,yval = utils.create_validation(xtrain,ytrain,1/6)
utils.setup_gpu_session()
print("====== START TRAINING NEURAL NETWORK MODEL ======")
model = ann.parse_model_js(network_model)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(xtrain,ytrain, verbose=1, validation_data=(xval,yval),epochs=epochs)
print("====== CALCULATING MNIST ACCURACY ======")
mnist_accuracy = ann.test_model(model, xtest, ytest, "accuracy")

print("====== LOAD CUSTOM DIGITS FROM OLEKSANDR AND HENRY ======")
xm_digits, xm_labels = utils.load_image_data(xm_digits_path, side=286, padding=40,unpad=False)
ob_digits, ob_labels = utils.load_image_data(ob_digits_path, side=286, padding=40,unpad=False)

xm_labels = utils.create_one_hot(xm_labels)
ob_labels = utils.create_one_hot(ob_labels)

combined_data = np.concatenate((xm_digits,ob_digits))
combined_labels = np.concatenate((xm_labels,ob_labels))

print("Analysing Henry's Digits")
xm_r,xm_acc,xm_tau,xm_new_dig = thickness_sim(model,xm_digits,xm_labels,mnist_linethickness)
print("== FINISH ==\n")
print("Analysing Oleksandr's Digits")
ob_r,ob_acc,ob_tau,ob_new_dig = thickness_sim(model,ob_digits,ob_labels,mnist_linethickness)
print("== FINISH ==\n")
print("Analysing COmbined digit set")
comb_r,comb_acc,comb_tau,comb_new_data = thickness_sim(model,combined_data,combined_labels,mnist_linethickness)
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
