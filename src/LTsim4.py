import utils
from utils import digitutils as dutils
import ANN as ann
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cv2
from scipy.stats import entropy

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

def calc_cerror(preds,labels):
    classes = ann.classify(preds)
    num_data = preds.shape[0]
    diff = classes - labels
    cerr = (1/(2 * num_data)) * np.sum(np.sum(np.abs(diff)))
    return cerr

def thickness_sim(model_list, data, labels ,thicknesses):

    m_preds = {}
    m_bits = {}
    m_cerr = {}

    for d,t in zip(digits,thicknesses):

        m_preds[t] = list(map(lambda m: m.predict(d), model_list))
        m_cerr[t] = list(map(lambda m: ann.test_model(m,d,labels,"c_error"), model_list))
        m_bits[t] = list(map(lambda m: ann.test_model(m,d,labels,"entropy"), model_list))

    return m_preds, m_cerr, m_bits

#mnist_linethickness = 66.97000583000295 ## Obtained from running mnistlinewidth.py file
mnist_linethickness = 14.095163376059986
# 93.62709087870702

epochs = 5

ensemblesize = 100

chunksize = 25

nchunks = ensemblesize // chunksize



xtrain,ytrain,xtest,ytest = utils.load_mnist()
reshape_funs = {
    "conv" : lambda d : d.reshape(-1,28,28,1),
    "mlp" : lambda d : d.reshape(-1,784)
}
reshape_fun = reshape_funs['mlp']
xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)

utils.setup_gpu_session()
digits_data = utils.load_processed_data("combined_testing_data")
taus = list(digits_data.keys())[:-1]
digits = list(map(reshape_fun, [digits_data[t] for t in taus]))
digits = list(map(utils.normalize_data, digits))
labels = utils.create_one_hot(digits_data['labels'].astype('uint'))    


mnist_mpreds = []
digits_mpreds = {}
mnist_mcerr = []
digits_mcerr = {}
mnist_mbits = []
digits_mbits = {}

for t in taus:
    digits_mpreds[t] = []
    digits_mcerr[t] = []
    digits_mbits[t] = []

for _ in range(nchunks):
    print("====== START TRAINING NEURAL NETWORK MODEL ======")

    l_xtrain = []
    l_xval = []
    l_ytrain = []
    l_yval = []
    for _ in range(chunksize):
        t_xtrain,t_ytrain,t_xval,t_yval = utils.create_validation(xtrain,ytrain,(1/6))
        l_xtrain.append(t_xtrain)
        l_xval.append(t_xval)
        l_ytrain.append(t_ytrain)
        l_yval.append(t_yval)

    inputs, outputs, train_model, model_list, _ = ann.build_ensemble([network_model2],chunksize,None)
    train_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    train_model.fit(l_xtrain,l_ytrain, verbose=1, validation_data=(l_xval,l_yval),epochs=epochs)
    m_mpreds = list(map(lambda m: m.predict(xtest), model_list))
    m_mcerr = list(map(lambda m: ann.test_model(m,xtest,ytest,"c_error"), model_list))
    m_mbits = list(map(lambda m: ann.test_model(m,xtest,ytest,"entropy"), model_list))
    d_mpreds, d_mcerr, d_mbits = thickness_sim(model_list, digits, labels, taus)
    mnist_mcerr.extend(m_mcerr)
    mnist_mpreds.extend(m_mpreds)
    mnist_mbits.extend(m_mbits)

    for t in taus:
        digits_mpreds[t].extend(d_mpreds[t])
        digits_mcerr[t].extend(d_mcerr[t])
        digits_mbits[t].extend(d_mbits[t])

mnist_preds = np.mean(np.array(mnist_mpreds),axis=0)
mnist_cerr = calc_cerror(mnist_preds, ytest)
mnist_bits = np.mean(list(map(entropy, mnist_preds)))

digits_cerr = {}
digits_bits = {}

for t in taus:
    preds = np.mean(np.array(digits_mpreds[t]), axis=0)
    digits_cerr[t] = calc_cerror(preds, labels)
    digits_bits[t] = np.mean(list(map(entropy, preds)))

results = {
    "ensembles": {
        "mnist_bits" : mnist_bits,
        "mnist_cerr" : mnist_cerr,
        "digits_cerr" : digits_cerr,
        "digits_bits" : digits_bits
    },
    "individuals": {
        "mnist_bits" : mnist_mbits,
        "mnist_cerr" : mnist_mcerr,
        "digits_cerr" : digits_mcerr,
        "digits_bits" : digits_mbits
    }
}


utils.save_processed_data(results, "results_ltsim_100")

