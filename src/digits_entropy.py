import utils
import ANN as ann
import numpy as np
from keras import losses as klosses
from functools import reduce

ensemble_sizes = list(range(3,6))
num_epochs = 3

mlp_structure = '''
{
    "input_shape" : [784],
    "layers" : [
        {
            "type" : "Dense",
            "units" : 64,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 128,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 32,
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

convnet_structure = '''
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
def calc_shannon_entropy(pred):
    pred[np.where(pred == 0)]=1
    bitsmat = pred * np.log2(pred)
    bits = -1 * np.sum(bitsmat, axis=1)
    return bits

def calc_class_entropy(model_list, data):
    ensemble_size = len(model_list)
    pred_list = list(map(lambda m : m.predict(data), model_list))
    class_list = list(map(ann.classify,pred_list))
    ensemble_sum = reduce(np.add,class_list)
    #print(ensemble_sum * (1/ensemble_size))
    return calc_shannon_entropy(ensemble_sum * (1/ensemble_size))

def experiment(network_conf_json, reshape_mode = "conv"):
    reshape_funs = {
        "conv" : lambda d : d.reshape(-1,28,28,1),
        "mlp" : lambda d : d.reshape(-1,784)
    }

    xtrain,ytrain,xtest,ytest = utils.load_mnist()
    reshape_fun = reshape_funs[reshape_mode]
    xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)

    xtrain,ytrain,xval,yval = utils.create_validation(xtrain,ytrain)

    

    mnist_c_errors = []
    mnist_pred_bits = []
    mnist_class_bits = []

    xm_c_errors = []
    xm_pred_bits = []
    xm_class_bits = []

    ob_c_errors = []
    ob_pred_bits = []
    ob_class_bits = []

    xm_data = utils.load_processed_data("xiaoming_digits")
    ob_data = utils.load_processed_data("Oleks_digits")

    xm_digits = reshape_fun(utils.normalize_data(list(xm_data.values())[0]))
    xm_labels = utils.create_one_hot(xm_data["labels"])

    ob_digits = reshape_fun(utils.normalize_data(list(ob_data.values())[0]))
    ob_labels = utils.create_one_hot(ob_data["labels"])

    for ensemble_size in ensemble_sizes:
        # Building the ensemble models and training the networks
        print("===== Building the ensemble models and training the networks =====")
        
        inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([network_conf_json], pop_per_type=ensemble_size, merge_type="Average")
        train_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
        train_model.fit([xtrain]*ensemble_size,[ytrain]*ensemble_size,batch_size=100,verbose=1,validation_data=([xval]*ensemble_size,[yval]*ensemble_size),epochs=num_epochs)
        
        # Calculating classification errors 
        print("===== Calculating classification errors =====")

        mnist_c_errors.append(ann.test_model(merge_model,[xtest] * ensemble_size,ytest,metric="c_error"))
        xm_c_errors.append(ann.test_model(merge_model,[xm_digits] * ensemble_size,xm_labels,metric="c_error"))
        ob_c_errors.append(ann.test_model(merge_model,[ob_digits] * ensemble_size,ob_labels,metric="c_error"))

        # Calculating ensemble prediciton entropy
        print("===== Calculating ensemble prediciton entropy =====")

        mnist_pred_bits.append(np.mean(calc_shannon_entropy(merge_model.predict([xtest]*ensemble_size))))
        xm_pred_bits.append(np.mean(calc_shannon_entropy(merge_model.predict([xm_digits]*ensemble_size))))
        ob_pred_bits.append(np.mean(calc_shannon_entropy(merge_model.predict([ob_digits]*ensemble_size))))

        # Calculating ensemble members classification entropy

        print("===== Calculating ensemble members classification entropy =====")
        mnist_class_bits.append(np.mean(calc_class_entropy(model_list,xtest)))
        xm_class_bits.append(np.mean(calc_class_entropy(model_list,xm_digits)))
        ob_class_bits.append(np.mean(calc_class_entropy(model_list,ob_digits)))
    
    mnist_results = {
        "c_error" : mnist_c_errors,
        "pred_bits" : mnist_pred_bits,
        "class_bits" : mnist_class_bits
    }

    xm_results = {
        "c_error" : xm_c_errors,
        "pred_bits" : xm_pred_bits,
        "class_bits" : xm_class_bits
    }

    ob_results = {
        "c_error" : ob_c_errors,
        "pred_bits" : ob_pred_bits,
        "class_bits" : ob_class_bits
    }

    return mnist_results, xm_results, ob_results

mnist_results, xm_results, ob_results = experiment(mlp_structure,"mlp")

print(mnist_results["class_bits"])

