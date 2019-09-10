import utils
import ANN as ann
import keras.layers as layers
from keras.models import Model, Sequential
import numpy as np
import keras.regularizers as reg
import keras.optimizers as opt
import keras.callbacks as clb
import keras.initializers as inits
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.io as sio

ensemble_size = 5
trials = 1
epochs = 50

reshape_fun = lambda d : d.reshape(-1,784)

def load_notMNIST():
    data = sio.loadmat('notMNIST_small')
    x = data['images'].transpose(2,0,1)
    return utils.normalize_data(x)

xtrain,ytrain,xtest,ytest = utils.load_mnist()
xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)
notmnist = load_notMNIST()
notmnist = reshape_fun(notmnist)
print(notmnist.shape)
print(notmnist.max())

ensemble_sizes = [1,5,10,20]

model_list = []
inputs = []
outputs = []

for t in range(trials):
    print("On trial %s" % (t+1))
    
    results = {}
    for e in ensemble_sizes:
        results[e] = []
    
    for ensemble_size in ensemble_sizes:
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

            es = clb.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
            model.compile(optimizer=opt.Adam(),loss="categorical_crossentropy",metrics=['acc'])
            model.fit(xtrain,ytrain,epochs=epochs,batch_size=100,validation_split=(1/6),callbacks=[es])
            model_list.append(model)

            inputs.extend(model.inputs)
            outputs.extend(model.outputs)

        merge_layer = layers.Average()(outputs) if ensemble_size > 1 else outputs
        ensemble = Model(inputs=inputs,outputs=merge_layer)
        pred = ensemble.predict([notmnist]*ensemble_size)
        h = list(map(stats.entropy, pred))
        results[ensemble_size].extend(h)
    
    utils.save_processed_data(results,'notmnist_sim-trial-%s' % (t+1))

