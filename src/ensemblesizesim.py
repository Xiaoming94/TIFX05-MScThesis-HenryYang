import utils
import ANN as ann
import numpy as np
import keras.callbacks as clb
import keras.optimizers as opt
import matplotlib.pyplot as plt

network_model1 = '''
{
    "input_shape" : [784],
    "layers" : [
        {
            "type" : "Dense",
            "units" : 200,
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
            "type" : "Dense",
            "units" : 200,
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
            "type" : "Dense",
            "units" : 200,
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
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax"
        }    
    ]
}
'''

trials = 10
epochs = 50

reshape_funs = {
    "mlp" : lambda d : d.reshape(-1,784),
    "conv" : lambda d : d.reshape(-1,28,28,1)
}

xtrain,ytrain,xtest,ytest = utils.load_mnist()
reshape_fun = reshape_funs["mlp"]
xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)

accuracies = []
utils.setup_gpu_session()

for t in range(trials):

    t_accuracies = []
    for e in range(25):
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

        es = clb.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights = True)
        ensemble_size = e + 1
        inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([network_model1], pop_per_type=ensemble_size, merge_type="Average")
        #print(np.array(train_model.predict([xtest]*ensemble_size)).transpose(1,0,2).shape)
        train_model.compile(optimizer = opt.Adam(0.1), loss = 'categorical_crossentropy', metrics = ['acc'])
        train_model.fit(l_xtrain,l_ytrain,epochs = epochs, batch_size=100 ,validation_data = (l_xval,l_yval), callbacks=[es])

        merge_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
        c_error = ann.test_model(merge_model, [xtest]*ensemble_size, ytest, metric = 'accuracy' )
        t_accuracies.append(c_error)
    
    accuracies.append(t_accuracies)

accuracies = np.array(accuracies)
utils.save_processed_data(accuracies,"results_ensizesim")
