import utils
import ANN as ann
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import utils.digitutils as dutils
import cv2
import keras.callbacks as clb

network_model1 = '''
{
    "input_shape" : [784],
    "layers" : [
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu",
            "activity_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            }
        },
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu",
            "activity_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            }
        },
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu",
            "activity_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            }
        },
        {
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax",
            "activity_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            }
        }     
    ]
}
'''

network_model2 = '''
{
    "input_shape" : [28,28,1],
    "layers" : [
        {
            "type" : "Conv2D",
            "units" : 48,
            "kernel_size" : [3,3],
            "activation" : "relu",
            "kernel_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            }
        },
        {
            "type" : "BatchNormalization",
            "axis" : -1
        },
        {
            "type" : "Conv2D",
            "units" : 96,
            "kernel_size" : [3,3],
            "activation" : "relu",
            "kernel_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            },
        },
        {
            "type" : "BatchNormalization",
            "axis" : -1
        },
        {
            "type" : "Conv2D",
            "units" : 64,
            "kernel_size" : [3,3],
            "activation" : "relu",
            "kernel_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            },
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
            "units" : 100,
            "activation" : "relu",
            "kernel_regularizer" : {
                "type" : "l2",
                "lambda" : 0.001
            }
        },
        {
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax"
            "kernel_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            },
            "activity_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            }
        }
    ]
}
'''


def calc_pred_vars(mempred):
    M,K = mempred.shape
    cumsum = 0
    for k in mempred:

        cumsum += (np.sum(k*k)/K - ((np.sum(k)/K)**2))

    return cumsum/M



def experiment(network_model, reshape_mode = 'mlp'):
    reshape_funs = {
        "conv" : lambda d : d.reshape(-1,28,28,1),
        "mlp" : lambda d : d.reshape(-1,784)
    }
    xtrain,ytrain,xtest,ytest = utils.load_mnist()
    reshape_fun = reshape_funs[reshape_mode]
    xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)

    test_data = utils.load_processed_data('notmnist')
    letters = list(test_data.keys())

    ensemble_size = 20
    epochs = 50
    trials = 1

    results = {
        'A': [],
        'B': [],
        'C': [],
        'D': [],
        'E': [],
        'F': [],
        'G': [],
        'H': [],
        'I': [],
        'J': []
    }

    for t in range(trials):

        l_xtrain = []
        l_xval = []
        l_ytrain = []
        l_yval = []
        for _ in range(ensemble_size):
            t_xtrain,t_ytrain,t_xval,t_yval = utils.create_validation(xtrain,ytrain,(1/6))
            l_xtrain.append(t_xtrain)
            l_xval.append(t_xval)
            l_ytrain.append(t_ytrain)
            l_yval.append(t_yval)
        es = clb.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights = True)
        inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([network_model], pop_per_type=ensemble_size, merge_type="Average")
        #print(np.array(train_model.predict([xtest]*ensemble_size)).transpose(1,0,2).shape)
        train_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
        train_model.fit(l_xtrain,l_ytrain,epochs = epochs, batch_size=100 ,validation_data = (l_xval,l_yval),callbacks=[es])

        for letter in letters:
            inputs = test_data[letter]
            inputs = utils.normalize_data(inputs)
            inputs = reshape_fun(inputs)
            preds = merge_model.predict([inputs]*ensemble_size)
            mem_preds = np.array(train_model.predict([inputs]*ensemble_size)).transpose(1,2,0)
            bits = list(map(stats.entropy,preds))
            s_q = list(map(calc_pred_vars,mem_preds))
            results[letter].extend(list(zip(bits,s_q)))

    return results
utils.setup_gpu_session()

ensemble = experiment(network_model1, 'mlp')
utils.save_processed_data(ensemble , "distribution_not_mnist")

#plt.figure()
#plt.subplot(221)
#plt.hist(ensemble['mnist_correct'],color = 'blue')
#plt.xlabel('entropy')
#plt.ylabel('ncorrect')
#plt.subplot(222)
#plt.hist(ensemble['mnist_wrong'],color = 'red')
#plt.xlabel('entropy')
#plt.ylabel('nwrong')
#plt.subplot(223)
#plt.hist(ensemble['digits_correct'],color = 'blue')
#plt.xlabel('entropy')
#plt.ylabel('ncorrect')
#plt.subplot(224)
#plt.hist(ensemble['digits_wrong'],color = 'red')
#plt.xlabel('entropy')
#plt.ylabel('nwrong')

#plt.show()
