import utils
import ANN as ann

import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats

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
    trials = 5

    d1_correct = []
    d1_wrong = []
    d2_wrong = []
    d2_correct = []
    dog_wrong = []
    dog_correct = []


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

        inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([network_model], pop_per_type=ensemble_size, merge_type="Average")
        train_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
        train_model.fit(l_xtrain,l_ytrain,epochs = epochs, validation_data = (l_xval,l_yval))

        dog_preds = merge_model.predict([digits_og]*ensemble_size)
        correct, wrong = bin_entropies(d2_preds, d2_labels)
        dog_correct.extend(correct)
        dog_wrong.extend(wrong)

        for d in digits1:
            digits_preds = merge_model.predict([d]*ensemble_size)
            correct, wrong = bin_entropies(digits_preds,d_labels)
            d1.extend(wrong)
            d1.extend(correct)

        for d in digits2:
            digits_preds = merge_model.predict([d]*ensemble_size)
            correct, wrong = bin_entropies(digits_preds,d_labels)
            d2.extend(wrong)
            d2.extend(correct)

        ensemble = {
            'd1_correct' : d1_correct,
            'd1_wrong' : d1_wrong,
            'd2_correct' : d2_correct,
            'd2_wrong' : d2_wrong,
            'lecunn_correct' : dog_correct,
            'lecunn_wrong' : dog_wrong
        }

    return ensemble

ensemble = experiment(network_model2, 'conv')
utils.save_processed_data(ensemble , "cnn_entropy5trial-bins")

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
