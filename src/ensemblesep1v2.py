import utils
import ANN as ann

import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
from keras import callbacks as clb
from keras.models import Sequential, Model
import keras.layers as layers
import keras.initializers as inits
import keras.optimizers as opt

network_model1 = '''
{
    "input_shape" : [784],
    "layers" : [
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu",
            "kernel_regularizer" : {
                "type" : "l2",
                "lambda" : 0.001
            },
            "activity_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            }
        },
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu",
            "kernel_regularizer" : {
                "type" : "l2",
                "lambda" : 0.001
            },
            "activity_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            }
        },
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu",
            "kernel_regularizer" : {
                "type" : "l2",
                "lambda" : 0.001
            },
            "activity_regularizer" : {
                "type" : "l2",
                "lambda" : 0.0001
            }
        },
        {
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax",
            "kernel_regularizer" : {
                "type" : "l2",
                "lambda" : 0.001
            },
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
                "type" : "l1",
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



def bin_entropies(preds, mempreds ,labels):
    bits = list(map(stats.entropy, preds))
    s_square = list(map(calc_pred_vars,mempreds))
    classes = ann.classify(preds)
    diff = classes - labels

    wrong = []
    correct = []

    for h,s2,d in zip(bits,s_square,diff):
        if np.linalg.norm(d) > 0:
            wrong.append([h,s2])
        else:
            correct.append([h,s2])
    
    return correct, wrong


def experiment(network_model, reshape_mode = 'mlp'):
    reshape_funs = {
        "conv" : lambda d : d.reshape(-1,28,28,1),
        "mlp" : lambda d : d.reshape(-1,784)
    }
    xtrain,ytrain,xtest,ytest = utils.load_mnist()
    reshape_fun = reshape_funs[reshape_mode]
    xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)

    digits_data = utils.load_processed_data('combined_testing_data')
    digits_data2 = utils.load_processed_data('digits_og_and_optimal')
    taus = [13,14,15]

    digits = list(map(reshape_fun, [digits_data[t] for t in taus]))
    digits = list(map(utils.normalize_data, digits))
    digits_og = reshape_fun(digits_data2['lecunn'])
    digits_og = utils.normalize_data(digits_og)

    d_labels = utils.create_one_hot(digits_data['labels'].astype('uint'))
    d2_labels = utils.create_one_hot(digits_data2['labels'].astype('uint'))

    ensemble_size = 20
    epochs = 50
    trials = 10

    mnist_correct = []
    mnist_wrong = []
    digits_wrong = []
    digits_correct = []
    d2_wrong = []
    d2_correct = []


    for t in range(trials):

        inputs = []
        outputs = []
        model_list = []

        for e in range(ensemble_size):
            model = Sequential()
            model.add(layers.Dense(200,input_dim=784, kernel_initializer = inits.RandomUniform(maxval=0.5,minval=-0.5)))
            model.add(layers.Activation("relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(200, kernel_initializer = inits.RandomUniform(maxval=0.5,minval=-0.5)))
            model.add(layers.Activation("relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(200, kernel_initializer = inits.RandomUniform(maxval=0.5,minval=-0.5)))
            model.add(layers.Activation("relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(10, kernel_initializer = inits.RandomUniform(maxval=0.5,minval=-0.5)))
            model.add(layers.Activation("softmax"))

            es = clb.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
            model.compile(optimizer=opt.Adam(),loss="categorical_crossentropy",metrics=['acc'])
            model.fit(xtrain,ytrain,epochs=epochs,batch_size=100,validation_split=(1/6),callbacks=[es])
            model_list.append(model)

            inputs.extend(model.inputs)
            outputs.extend(model.outputs)

        merge_model = Model(inputs = inputs, outputs = layers.Average()(outputs))

        mnist_preds = merge_model.predict([xtest]*ensemble_size)
        mnist_mem_preds = np.array(list(map(lambda m : m.predict(xtest), model_list))).transpose(1,2,0)
        correct, wrong = bin_entropies(mnist_preds, mnist_mem_preds ,ytest)
        mnist_correct.extend(correct)
        mnist_wrong.extend(wrong)

        d2_preds = merge_model.predict([digits_og]*ensemble_size)
        d2_mempreds = np.array(list(map(lambda m : m.predict(digits_og), model_list))).transpose(1,2,0)
        correct, wrong = bin_entropies(d2_preds, d2_mempreds, d2_labels)
        d2_correct.extend(correct)
        d2_wrong.extend(wrong)

        for d in digits:
            digits_preds = merge_model.predict([d]*ensemble_size)
            mempreds = np.array(list(map(lambda m : m.predict(d),model_list))).transpose(1,2,0)
            correct, wrong = bin_entropies(digits_preds,mempreds,d_labels)
            digits_wrong.extend(wrong)
            digits_correct.extend(correct)

        ensemble = {
            'mnist_correct' : mnist_correct,
            'mnist_wrong' : mnist_wrong,
            'digits_correct' : digits_correct,
            'digits_wrong' : digits_wrong,
            'lecunn_correct' : d2_correct,
            'lecunn_wrong' : d2_wrong
        }

    return ensemble

ensemble = experiment(network_model1, 'mlp')
utils.save_processed_data(ensemble , "entropy5sep-bins")

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
