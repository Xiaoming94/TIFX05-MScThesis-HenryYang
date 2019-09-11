import utils
import ANN as ann
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import utils.digitutils as dutils
import cv2
import keras.callbacks as clb
import keras.optimizers as opt
from keras.models import Model, Sequential
import keras.layers as layers
import scipy.io as sio
import keras.initializers as inits

def load_notMNIST():
    data = sio.loadmat('notMNIST_small')
    x = data['images'].transpose(2,0,1)
    return utils.normalize_data(x)


def calc_pred_vars(mempred):
    M,K = mempred.shape
    cumsum = 0
    for k in mempred:

        cumsum += (np.sum(k*k)/K - ((np.sum(k)/K)**2))

    return cumsum/M



def experiment(reshape_mode = 'mlp'):
    reshape_funs = {
        "conv" : lambda d : d.reshape(-1,28,28,1),
        "mlp" : lambda d : d.reshape(-1,784)
    }
    xtrain,ytrain,xtest,ytest = utils.load_mnist()
    reshape_fun = reshape_funs[reshape_mode]
    xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)

    notmnist = load_notMNIST()
    notmnist = reshape_fun(notmnist)
    print(notmnist.shape)

    ensemble_size = 20
    epochs = 50
    trials = 10

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

    results = []
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

            es = clb.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)
            model.compile(optimizer=opt.Adam(),loss="categorical_crossentropy",metrics=['acc'])
            model.fit(xtrain,ytrain,epochs=epochs,batch_size=100,validation_split=(1/6),callbacks=[es])
            model_list.append(model)

            inputs.extend(model.inputs)
            outputs.extend(model.outputs)
        
        merge_model = Model(inputs = inputs, outputs = layers.Average()(outputs))

        preds = merge_model.predict([notmnist]*ensemble_size)
        mem_preds = np.array(list(map(lambda m : m.predict(notmnist), model_list))).transpose(1,2,0)
        print(mem_preds.shape)
        bits = list(map(stats.entropy,preds))
        s_q = list(map(calc_pred_vars,mem_preds))
        results.extend(list(zip(bits,s_q)))
    return results
utils.setup_gpu_session()

ensemble = experiment('mlp')
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
