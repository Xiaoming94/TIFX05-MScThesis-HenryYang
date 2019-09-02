import utils
import ANN as ann

import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats

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
                "type" : "l1",
                "lambda" : 0.0001
            }
        }
    ]
}
'''
def scale_down(img):
    downscaled =  cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC )

    return downscaled

def salt_and_pepper(digits,num):
    def reduce_black_pixel(digits, num):
        dless = []
        for d in digits:
            nonzeros = np.array(np.where(d != 0)).transpose()
            # reducing black pixels
            if nonzeros.shape[0] < num:
                dless.append(np.zeros((28,28)))
            else:
                img_dl = d.copy()
                randidx = np.random.permutation(nonzeros.shape[0])
                for i in randidx[:num]:
                    [x,y] = nonzeros[i]
                    img_dl[x,y] = 0
                dless.append(scale_down(img_dl))
        return np.array(dless)

    def increase_black_pixels(digits, num):
        dmore = []
        for d in digits:
            zeros = np.array(np.where(d == 0)).transpose()
            if zeros.shape[0] < num:
                dmore.append(np.ones((28,28)))
            else:
                img_more = d.copy()
                randidx = np.random.permutation(zeros.shape[0])
                for i in randidx[:num]:
                    [x,y] = zeros[i]
                    img_more[x,y] = 255
                dmore.append(scale_down(img_more))
        return np.array(dmore)

    if num < 0 :
        return reduce_black_pixel(digits,abs(num))
    else:
        return increase_black_pixels(digits,abs(num))

def calc_pred_vars(mempred):
    M,K = mempred.shape
    cumsum = 0
    for k in mempred:
        cumsum += (np.sum(k*k)/K - ((np.sum(k)/K)**2))
    
    return cumsum/M


def apply_salt_pepper(digits):
    dnum = 800
    steps = [25,27,30]
    sp_digits = []
    for s in steps:
       sp_digit = salt_and_pepper(digits,dnum*(s-15))
       sp_digits.append(sp_digit)

    return sp_digits

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

    digits_data = utils.load_processed_data('combined_testing_data_more')
    digits_data2 = utils.load_processed_data('digits_og_and_optimal')
    taus = [25,27,30]

    digits = list(map(reshape_fun, [digits_data[t] for t in taus]))
    digits = list(map(utils.normalize_data, digits))
    digits_og = digits_data2['optimal_lw']

    d_labels = utils.create_one_hot(digits_data['labels'].astype('uint'))
    d2_labels = utils.create_one_hot(digits_data2['labels'].astype('uint'))

    ensemble_size = 10
    epochs = 5
    trials = 5

    mnist_correct = []
    mnist_wrong = []
    digits_wrong = []
    digits_correct = []
    d2_wrong = []
    d2_correct = []


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

        es = clb.EarlyStopping(monitor='val_loss', patience = 2, restore_best_weights = True)
        inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([network_model], pop_per_type=ensemble_size, merge_type="Average")
        #print(np.array(train_model.predict([xtest]*ensemble_size)).transpose(1,0,2).shape)
        train_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
        train_model.fit(l_xtrain,l_ytrain,epochs = epochs, validation_data = (l_xval,l_yval),callbacks=[es])

        #mnist_preds = merge_model.predict([xtest]*ensemble_size)
        #mnist_mem_preds = np.array(train_model.predict([xtest]*ensemble_size)).transpose(1,2,0)
        #correct, wrong = bin_entropies(mnist_preds, mnist_mem_preds ,ytest)
        #mnist_correct.extend(correct)
        #mnist_wrong.extend(wrong)
        sp_digits = apply_salt_pepper(digits_og)
        
        for s_d in sp_digits:
            s_d = utils.normalize_data(reshape_fun(s_d))
            d2_preds = merge_model.predict([s_d]*ensemble_size)
            d2_mempreds = np.array(train_model.predict([s_d]*ensemble_size)).transpose(1,2,0)
            correct, wrong = bin_entropies(d2_preds, d2_mempreds, d2_labels)
            d2_correct.extend(correct)
            d2_wrong.extend(wrong)

        for d in digits:
            digits_preds = merge_model.predict([d]*ensemble_size)
            mempreds = np.array(train_model.predict([d]*ensemble_size)).transpose(1,2,0)
            correct, wrong = bin_entropies(digits_preds,mempreds,d_labels)
            digits_wrong.extend(wrong)
            digits_correct.extend(correct)

        ensemble = {
            #'mnist_correct' : mnist_correct,
            #'mnist_wrong' : mnist_wrong,
            'digits_correct' : digits_correct,
            'digits_wrong' : digits_wrong,
            'lecunn_correct' : d2_correct,
            'lecunn_wrong' : d2_wrong
        }

    return ensemble

ensemble = experiment(network_model1, 'mlp')
utils.save_processed_data(ensemble , "error_entropy5sep-bins")

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
