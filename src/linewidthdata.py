import utils
from utils import digitutils as dutils
import ANN as ann
import numpy as np
import cv2

network_model1 = '''
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
'''

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


epochs = 3
trials = 100

xtrain,ytrain,xtest,ytest = utils.load_mnist()
digits_data = utils.load_processed_data('digits_og_and_optimal')
digits_big = digits_data['lecunn_big']
digits_labels = utils.create_one_hot(digits_data['labels'].astype('uint'))
digits = digits_data['lecunn']

def resize_helper(img):
    sides = 280
    return cv2.resize(img, (sides,sides), interpolation = cv2.INTER_CUBIC)

def calc_linewidths(data):
    taus = np.zeros(data.shape[0])
    for i,d in zip(range(data.shape[0]),data):
            taus[i] = dutils.intern_calc_linewidth(d)

    return np.mean(taus), np.var(taus)

def calc_acc():
    reshape_funs = {
        "conv" : lambda d : d.reshape(-1,28,28,1),
        "mlp" : lambda d : d.reshape(-1,784)
    }
    
    acc_mnist = np.zeros(trials)
    bits_mnist = np.zeros(trials)
    acc_digits = np.zeros(trials)
    bits_digits = np.zeros(trials)

    cnn_acc_mnist = np.zeros(trials)
    cnn_bits_mnist = np.zeros(trials)
    cnn_acc_digits = np.zeros(trials)
    cnn_bits_digits = np.zeros(trials)

    for i in range(trials):
        t_xtrain, t_ytrain, xval, yval = utils.create_validation(xtrain,ytrain)

        reshape_fun = reshape_funs['mlp']

        t_xtrain,c_xtest = reshape_fun(t_xtrain), reshape_fun(xtest)
        xval = reshape_fun(xval)
        c_digits = reshape_fun(digits)

        model = ann.parse_model_js(network_model2)
        model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['acc'])
        model.fit(t_xtrain,t_ytrain, epochs = 5, validation_data = (xval,yval))
    
        acc_mnist[i] = ann.test_model(model, c_xtest, ytest, metric = 'accuracy')
        bits_mnist[i] = ann.test_model(model, c_xtest, ytest, metric = 'entropy')
        acc_digits[i] = ann.test_model(model, c_digits, digits_labels, metric = 'accuracy')
        bits_digits[i] = ann.test_model(model, c_digits, digits_labels, metric = 'entropy')

        reshape_fun = reshape_funs['conv']
        t_xtrain,c_xtest = reshape_fun(t_xtrain), reshape_fun(xtest)
        xval = reshape_fun(xval)
        c_digits = reshape_fun(digits)

        model = ann.parse_model_js(network_model1)
        model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['acc'])
        model.fit(t_xtrain,t_ytrain, epochs = 3, validation_data = (xval,yval))
    
        cnn_acc_mnist[i] = ann.test_model(model, c_xtest, ytest, metric = 'accuracy')
        cnn_bits_mnist[i] = ann.test_model(model, c_xtest, ytest, metric = 'entropy')
        cnn_acc_digits[i] = ann.test_model(model, c_digits, digits_labels, metric = 'accuracy')
        cnn_bits_digits[i] = ann.test_model(model, c_digits, digits_labels, metric = 'entropy')

    results_mlp = {
        'mnist_acc' : acc_mnist,
        'mnist_bits' : bits_mnist,
        'digits_acc' : acc_digits,
        'digits_bits' : bits_digits
    }

    results_cnn = {
        'mnist_acc' :  cnn_acc_mnist,
        'mnist_bits' : cnn_bits_mnist,
        'digits_acc' : cnn_acc_digits,
        'digits_bits' : cnn_bits_digits
    }

    utils.save_processed_data(results_mlp,'100-trials-mlp')
    utils.save_processed_data(results_cnn, '100-trials-cnn')

#mnist_mean, mnist_var = calc_linewidths(xtest)
#digits_mean, digits_var = calc_linewidths(digits)
#print('mnist: (mean %s, var %s)' % (mnist_mean, mnist_var))
#print('digits: (mean %s, var %s)' % (digits_mean, digits_var))
utils.setup_gpu_session()
calc_acc()
