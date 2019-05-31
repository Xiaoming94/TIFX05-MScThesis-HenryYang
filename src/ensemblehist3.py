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

def vote_predict(model_list, ensemble_size, data):
    votes = 0
    for m in model_list:
        preds = m.predict(data)
        votes += ann.classify(preds)
        
    return votes/ensemble_size


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
    taus = [13,14,15]

    digits = list(map(reshape_fun, [digits_data[t] for t in taus]))
    digits = list(map(utils.normalize_data, digits))
    digits_og = reshape_fun(digits_data2['lecunn'])
    digits_og = utils.normalize_data(digits_og)

    d_labels = utils.create_one_hot(digits_data['labels'].astype('uint'))
    d2_labels = utils.create_one_hot(digits_data2['labels'].astype('uint'))

    ensemble_size = 20
    epochs = 3
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

        inputs, outputs, train_model, model_list,_ = ann.build_ensemble([network_model], pop_per_type=ensemble_size, merge_type=None)
        train_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
        train_model.fit(l_xtrain,l_ytrain,epochs = epochs, validation_data = (l_xval,l_yval))

        mnist_preds = vote_predict(model_list, ensemble_size, xtest)
        correct, wrong = bin_entropies(mnist_preds, ytest)
        mnist_correct.extend(correct)
        mnist_wrong.extend(wrong)

        d2_preds = vote_predict(model_list, ensemble_size, digits_og)
        correct, wrong = bin_entropies(d2_preds, d2_labels)
        d2_correct.extend(correct)
        d2_wrong.extend(wrong)

        for d in digits:
            digits_preds = vote_predict(model_list, ensemble_size, d)
            correct, wrong = bin_entropies(digits_preds,d_labels)
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
utils.save_processed_data(ensemble , "vote_entropy5trial-bins")

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
