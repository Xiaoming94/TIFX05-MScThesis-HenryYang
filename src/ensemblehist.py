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
    taus = [13,14,15]

    digits = list(map(reshape_fun, [digits_data[t] for t in taus]))
    digits = list(map(utils.normalize_data, digits))

    d_labels = utils.create_one_hot(digits_data['labels'].astype('uint'))

    ensemble_size = 20
    epochs = 5

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

    mnist_mpreds = np.array(list(map(lambda m: m.predict(xtest), model_list)))
    
    mnist_mnwrong = []
    mnist_mncorrect = []

    for mp in mnist_mpreds:
        correct, wrong = bin_entropies(mp,ytest)
        mnist_mnwrong.extend(wrong)
        mnist_mncorrect.extend(correct)

    mnist_preds = mnist_mpreds.mean(axis=0)
    
    digits_mnwrong = []
    digits_mncorrect = []
    for d in digits:
        for m in model_list:
            preds = m.predict(d)
            correct,wrong = bin_entropies(preds, d_labels)
            digits_mnwrong.extend(wrong)
            digits_mncorrect.extend(correct)
    
    individual = {
        'mnist_correct' : mnist_mncorrect,
        'mnist_wrong' : mnist_mnwrong,
        'digits_correct' : digits_mncorrect,
        'digits_wrong' : digits_mnwrong
    }

    return individual

individual = experiment(network_model1, 'mlp')

plt.figure()
plt.subplot(221)
plt.hist(individual['mnist_correct'],color = 'blue')
plt.xlabel('entropy')
plt.ylabel('ncorrect')
plt.subplot(222)
plt.hist(individual['mnist_wrong'],color = 'red')
plt.xlabel('entropy')
plt.ylabel('nwrong')
plt.subplot(223)
plt.hist(individual['digits_correct'],color = 'blue')
plt.xlabel('entropy')
plt.ylabel('ncorrect')
plt.subplot(224)
plt.hist(individual['digits_wrong'],color = 'red')
plt.xlabel('entropy')
plt.ylabel('nwrong')

plt.show()
