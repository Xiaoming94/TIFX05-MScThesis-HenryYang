import utils
import ANN as ann
import numpy as np
import keras.callbacks as clb
import keras.optimizers as opt
import matplotlib.pyplot as plt
import keras.initializers as inits
from keras.models import Sequential, Model
import keras.layers as layers

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
            "type" : "BatchNormalization",
            "axis" : -1
        },
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu"
        },
        {
            "type" : "BatchNormalization",
            "axis" : -1
        },
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu"
        },
        {
            "type" : "BatchNormalization",
            "axis" : -1
        },
        {
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax"
        }    
    ]
}
'''

trials = 10
epochs = 50
ensemble_size_top = 30

reshape_fun = lambda d : d.reshape(-1,784)

xtrain, ytrain, xtest, ytest = utils.load_mnist()
xtrain, xtest = reshape_fun(xtrain),reshape_fun(xtest)

for t in range(trials):

    print("==== on trial %s ====" % (t+1))
    t_accuracies = []
    members = []
    for i in range(int((ensemble_size_top * (ensemble_size_top + 1))/2)):
        model = Sequential()
        model.add(layers.Dense(200, input_dim = 784, kernel_initializer = inits.RandomUniform(maxval=(0.5),minval=(-0.5))))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(200, kernel_initializer = inits.RandomUniform(maxval=(0.5),minval=(-0.5))))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(200, kernel_initializer = inits.RandomUniform(maxval=(0.5),minval=(-0.5))))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(10, kernel_initializer = inits.RandomUniform(maxval=(0.5),minval=(-0.5))))
        model.add(layers.Activation('softmax'))
        
        es = clb.EarlyStopping(monitor='val_loss', patience = 10, restore_best_weights = True)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
        model.fit(xtrain,ytrain,epochs=epochs,batch_size=100,callbacks=[es],validation_split = (1/6))
        members.append(model)

    i = 0
    for ensemble_size in range(2,ensemble_size_top+1):
        inputs = []
        outputs = []
        members_to_use = members[i:i+ensemble_size]
        for m in members_to_use:
            inputs.extend(m.inputs)
            outputs.extend(m.outputs)
        
        print((outputs))
        ensemble = Model(inputs = inputs, outputs = layers.Average()(outputs))
        ensemble.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
        accuracy = ann.test_model(ensemble, [xtest]*ensemble_size,ytest, 'accuracy')
        i += ensemble_size
        t_accuracies.append(accuracy)
    
    t_accuracies = np.array(t_accuracies)
    utils.save_processed_data(t_accuracies,'ensemble_sizesim-trial-%s' % (t+1))
