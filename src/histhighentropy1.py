import utils
import ANN as ann

import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
import cv2

network_model1 = '''
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

network_model2 = '''
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
def scale_down(img):
    downscaled =  cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC )

    return downscaled


def bin_entropies(preds, labels):
    bits = list(map(stats.entropy, preds))
    classes = ann.classify(preds)
    diff = classes - labels

    wrong = []
    correct = []

    for h,d in zip(bits,diff):
        if np.linalg.norm(d) > 0:
            wrong.append(h)
        else:
            correct.append(h)
    
    return correct, wrong

def salt_and_pepper(digits,num):
    dnoice = []
    _,xl,yl = digits.shape
    for d in digits:
        xaxis = np.arange(xl)
        yaxis = np.arange(yl)
        indices = np.array(np.meshgrid(yaxis,xaxis)).transpose().reshape(-1,2)
        randints = np.random.permutation(xl * yl)
        noice_img = d.copy()
        for i in randints[:num]:
            [x,y] = indices[i]
            noice_img[x,y] = 0 if (noice_img[x,y] != 0) else 255
        dnoice.append(scale_down(noice_img))

    return np.array(dnoice) 



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
    taus = [26,28,30]
    taus2 = [2,4,5]
    digits1 = list(map(reshape_fun, [digits_data[t] for t in taus]))
    digits1 = list(map(utils.normalize_data, digits1))
    digits2 = list(map(reshape_fun, [digits_data[t] for t in taus2]))
    digits2 = list(map(utils.normalize_data, digits2))
    digits_og = digits_data2['lecunn_big']
    digits_og = salt_and_pepper(digits_og,72000)
    digits_og = reshape_fun(utils.normalize_data(digits_og))
    d_labels = utils.create_one_hot(digits_data['labels'].astype('uint'))
    d2_labels = utils.create_one_hot(digits_data2['labels'].astype('uint'))

    ensemble_size = 20
    epochs = 3

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

    inputs, outputs, train_model, model_list, _ = ann.build_ensemble([network_model], pop_per_type=ensemble_size, merge_type=None)
    train_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
    train_model.fit(l_xtrain,l_ytrain,epochs = epochs, validation_data = (l_xval,l_yval))
    
    d3_preds = np.array(list(map(lambda m: m.predict(digits_og), model_list)))
    d3_mnwrong = []
    d3_mncorrect = []

    for mp in d3_preds:
        correct, wrong = bin_entropies(mp,d2_labels)
        d3_mnwrong.extend(wrong)
        d3_mncorrect.extend(correct)
    

    d1_mnwrong = []
    d1_mncorrect = []
    for d in digits1:
        for m in model_list:
            preds = m.predict(d)
            correct,wrong = bin_entropies(preds, d_labels)
            d1_mnwrong.extend(wrong)
            d1_mncorrect.extend(correct)

    d2_mnwrong = []
    d2_mncorrect = []
    for d in digits2:
        for m in model_list:
            preds = m.predict(d)
            correct,wrong = bin_entropies(preds, d_labels)
            d2_mnwrong.extend(wrong)
            d2_mncorrect.extend(correct)
    
    individual = {
        'd1_correct' : d1_mncorrect,
        'd1_wrong' : d1_mnwrong,
        'd2_correct' : d2_mncorrect,
        'd2_wrong' : d2_mnwrong,
        'lecunn_correct' : d3_mncorrect,
        'lecunn_wrong' : d3_mnwrong
    }

    return individual

utils.setup_gpu_session()
individual = experiment(network_model1, 'mlp')
utils.save_processed_data(individual,'individual_bin')

#plt.figure()
#plt.subplot(231)
#plt.hist(individual['d1_correct'],color = 'blue')
#plt.xlabel('entropy')
#plt.ylabel('n_correct')
#plt.subplot(232)
#plt.hist(individual['d2_correct'],color = 'blue')
#plt.xlabel('entropy')
#plt.ylabel('n_correct')
#plt.subplot(233)
#plt.hist(individual['lecunn_correct'],color = 'blue')
#plt.xlabel('entropy')
#plt.ylabel('n_correct')
#plt.subplot(234)
#plt.hist(individual['d1_wrong'],color = 'red')
#plt.xlabel('entropy')
#plt.ylabel('n_wrong')
#plt.subplot(235)
#plt.hist(individual['d2_wrong'],color = 'red')
#plt.xlabel('entropy')
#plt.ylabel('n_wrong')
#plt.subplot(236)
#plt.hist(individual['lecunn_wrong'],color = 'red')
#plt.xlabel('entropy')
#plt.ylabel('n_wrong')
#plt.show()