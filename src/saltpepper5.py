import utils
import ANN as ann
import numpy as np
import cv2
import gc

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
        return dless

    def increase_black_pixels(digits, num):
        dmore = []
        for d in digits:
            if zeros.shape[0] < num:
                dmore.append(np.ones((28,28)))
            else:
                img_more = d.copy()
                randidx = np.random.permutation(zeros.shape[0])
                for i in randidx[:num]:
                    [x,y] = zeros[i]
                    img_more[x,y] = 255
                dmore.append(scale_down(img_more))
        return dmore
    
    if num < 0 :
        return reduce_black_pixel(digits,abs(num))
    else:
        return increase_black_pixels(digits,abs(num))

def test_digits(model, digits, labels, ensemble_size, reshape_fun):
    steps_results = {
        'c_error' : {},
        'entropy' : {}
    }

    dnum = 800

    for i in range(2,31):
        dchange = salt_and_pepper(digits,i * dnum - 15)

        d = utils.normalize_data(reshape_fun(dmore))
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
    labels = utils.create_one_hot(digits_data['labels'].astype('uint'))

    ensemble_size = 20
    epochs = 3
    small_digits = reshape_fun(np.array(list(map(scale_down, digits))))
    small_digits = utils.normalize_data(small_digits)
    trials = 10

    for t in range(1,trials+1):
        gc.collect()

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

        results = test_digits(merge_model, digits, labels, ensemble_size, reshape_fun)

        entropy = ann.test_model(merge_model, [small_digits]*ensemble_size, labels, metric = 'entropy')
        c_error = ann.test_model(merge_model, [small_digits]*ensemble_size, labels, metric = 'c_error')

        results['c_error'][0] = c_error
        results['entropy'][0] = entropy

        filename = "conv_saltpepper_leftright_trial-%s" % t
        utils.save_processed_data(results, filename)



utils.setup_gpu_session()
experiment(network_model1, 'mlp')

print("Script Done")
