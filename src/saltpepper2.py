import utils
import ANN as ann
import numpy as np
import cv2

import matplotlib.pyplot as plt


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

def test_digits(model, digits, labels, ensemble_size, reshape_fun):
    steps_results = {
        'c_error' : {},
        'entropy' : {}
    }

    dnum = 100

    for i in range(1,21):
        dnoice = salt_and_pepper(digits,i * dnum)
        
        d = utils.normalize_data(reshape_fun(dnoice))
        entropy = ann.test_model(model, [d]*ensemble_size, labels, metric = 'entropy')
        c_error = ann.test_model(model, [d]*ensemble_size, labels, metric = 'c_error')
        steps_results['entropy'][i] = entropy
        steps_results['c_error'][i] = c_error

    return steps_results

def experiment(network_model, reshape_mode = 'mlp'):
    reshape_funs = {
        "conv" : lambda d : d.reshape(-1,28,28,1),
        "mlp" : lambda d : d.reshape(-1,784)
    }
    xtrain,ytrain,xtest,ytest = utils.load_mnist()
    reshape_fun = reshape_funs[reshape_mode]
    xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)
    digits_data = utils.load_processed_data('digits_og_and_optimal')
    digits = digits_data['optimal_lw']
    digits_og = digits_data['lecunn_big']
    digits_og_small = reshape_fun(digits_data['lecunn'])
    labels = utils.create_one_hot(digits_data['labels'].astype('uint'))

    ensemble_size = 20
    epochs = 5
    small_digits = reshape_fun(np.array(list(map(scale_down, digits))))
    
    trials = 10
    for t in range(1,trials+1):
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
        train_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics = ['acc'])
        train_model.fit(x=l_xtrain,y=l_ytrain, verbose=1,batch_size=100, epochs = epochs,validation_data=(l_xval,l_yval))
        merge_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])

        results_linewidth = test_digits(merge_model, digits, labels, ensemble_size, reshape_fun)

        entropy = ann.test_model(merge_model, [small_digits]*ensemble_size, labels, metric = 'entropy')
        c_error = ann.test_model(merge_model, [small_digits]*ensemble_size, labels, metric = 'c_error')

        results_linewidth['c_error'][0] = c_error
        results_linewidth['entropy'][0] = entropy

        results_lecunn = test_digits(merge_model, digits_og, labels, ensemble_size, reshape_fun)

        entropy = ann.test_model(merge_model, [digits_og_small]*ensemble_size, labels, metric = 'entropy')
        c_error = ann.test_model(merge_model, [digits_og_small]*ensemble_size, labels, metric = 'c_error')

        results_lecunn['c_error'][0] = c_error
        results_lecunn['entropy'][0] = entropy

        total_results = {
            'optimal_lw' : results_linewidth,
            'lecunn' : results_lecunn
        }


        filename = "saltpepper_random_trial-%s" % t
        utils.save_processed_data(total_results, filename)




utils.setup_gpu_session()
experiment(network_model1, 'mlp')

#results_linewidth = results['optimal_lw']
#results_lecunn = results['lecunn']
#
#plt.figure()
#plt.subplot(121)
#plt.plot(list(results_linewidth['c_error'].keys()),list(results_linewidth['c_error'].values()),'*')
#plt.plot(list(results_lecunn['c_error'].keys()), list(results_lecunn['c_error'].values()),'x')
#
#plt.subplot(122)
#plt.plot(list(results_linewidth['entropy'].keys()),list(results_linewidth['entropy'].values()),'*')
#plt.plot(list(results_lecunn['entropy'].keys()), list(results_lecunn['entropy'].values()),'x')
#plt.show()

