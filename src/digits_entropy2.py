import utils
import ANN as ann
import numpy as np
from keras import losses as klosses
from functools import reduce
import matplotlib.pyplot as plt

num_epochs = 5
ensemble_size = 15

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
def make_plots(mnist_results, ob_results, xm_results, fig_title):
    mnist_linethickness = 67.14082725553595 # obtained by running mnistlinewidth.py

    fig = plt.figure()
    fig.suptitle(fig_title)
    plt.subplot(311)
    plt.plot([0,mnist_linethickness+0.5],[mnist_results["c_error"],mnist_results["c_error"]],linestyle='--')
    plt.plot([mnist_linethickness,mnist_linethickness],[0,mnist_results["c_error"]+0.01],linestyle='--')
    line1, = plt.plot(xm_results['taus'],xm_results["c_error"])
    line2, = plt.plot(ob_results['taus'],ob_results["c_error"])
    plt.xlabel("Line Thickness")
    plt.ylabel("classification error")
    plt.legend((line1,line2),("Henry's Digits","Olek's Digits"))

    plt.subplot(312)
    plt.plot([0,mnist_linethickness+0.5],[mnist_results["pred_bits"],mnist_results["pred_bits"]],linestyle='--')
    plt.plot([mnist_linethickness,mnist_linethickness],[0,mnist_results["pred_bits"]+0.01],linestyle='--')
    line1, = plt.plot(xm_results['taus'],xm_results["pred_bits"])
    line2, = plt.plot(ob_results['taus'],ob_results["pred_bits"])
    plt.xlabel("Line Thickness")
    plt.ylabel("Prediciton entropy (bits)")
    plt.legend((line1,line2),("Henry's Digits","Olek's Digits"))

    plt.subplot(313)
    plt.plot([0,mnist_linethickness+0.5],[mnist_results["class_bits"],mnist_results["class_bits"]],linestyle='--')
    plt.plot([mnist_linethickness,mnist_linethickness],[0,mnist_results["class_bits"]+0.01],linestyle='--')
    line1, = plt.plot(xm_results['taus'],xm_results["class_bits"])
    line2, = plt.plot(ob_results['taus'],ob_results["class_bits"])
    plt.xlabel("Line Thickness")
    plt.ylabel("class disagreement entropy (bits)")
    plt.legend((line1,line2),("Henry's Digits","Olek's Digits"))

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

def test_digits(data, labels, merge_model, model_list):
    data_count = len(data)
    c_errors = np.zeros(data_count)
    pred_bits = np.zeros(data_count)
    class_bits = np.zeros(data_count)
    for i,d in zip(range(data_count),data):
        c_errors[i] = ann.test_model(merge_model,[d] * ensemble_size,labels,metric="c_error")
        pred_bits[i] = np.mean(calc_shannon_entropy(merge_model.predict([d] * ensemble_size)))
        class_bits[i] = np.mean(calc_class_entropy(model_list, d))
    return c_errors, pred_bits, class_bits
    

def experiment(network_conf_json, reshape_mode = "conv"):
    reshape_funs = {
        "conv" : lambda d : d.reshape(-1,28,28,1),
        "mlp" : lambda d : d.reshape(-1,784)
    }

    xtrain,ytrain,xtest,ytest = utils.load_mnist()
    reshape_fun = reshape_funs[reshape_mode]
    xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)

    xtrain,ytrain,xval,yval = utils.create_validation(xtrain,ytrain)

    xm_data = utils.load_processed_data("xiaoming_digits")
    ob_data = utils.load_processed_data("Oleks_digits")

    xm_taus = np.array(list(xm_data.keys())[:-1])
    ob_taus = np.array(list(ob_data.keys())[:-1])

    xm_digits = list(map(reshape_fun,[xm_data[t] for t in xm_taus]))
    xm_labels = utils.create_one_hot(xm_data["labels"])

    ob_digits = list(map(reshape_fun,[ob_data[t] for t in ob_taus]))
    ob_labels = utils.create_one_hot(ob_data["labels"])

    # Building the ensemble models and training the networks
    print("===== Building the ensemble of size %s =====" % ensemble_size)
    
    inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([network_conf_json], pop_per_type=ensemble_size, merge_type="Average")
    train_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    train_model.fit([xtrain]*ensemble_size,[ytrain]*ensemble_size,batch_size=100,verbose=1,validation_data=([xval]*ensemble_size,[yval]*ensemble_size),epochs=num_epochs)

    mnist_c_errors, mnist_pred_bits, mnist_class_bits = test_digits([xtest], ytest, merge_model, model_list)
    xm_c_errors, xm_pred_bits, xm_class_bits = test_digits(xm_digits, xm_labels,merge_model, model_list)
    ob_c_errors, ob_pred_bits, ob_class_bits = test_digits(ob_digits, ob_labels,merge_model, model_list)

    mnist_values = {
        "c_error" : mnist_c_errors[0],
        "pred_bits" : mnist_pred_bits[0],
        "class_bits" : mnist_class_bits[0]
    }

    xm_results = {
        "taus" : xm_taus,
        "c_error" : xm_c_errors,
        "pred_bits" : xm_pred_bits,
        "class_bits" : xm_class_bits
    }

    ob_results = {
        "taus" : ob_taus,
        "c_error" : ob_c_errors,
        "pred_bits" : ob_pred_bits,
        "class_bits" : ob_class_bits
    }

    return mnist_values, xm_results, ob_results

utils.setup_gpu_session()
#mnist_values, xm_results, ob_results = experiment(mlp_structure,"mlp")
#make_plots(mnist_values, ob_results,xm_results,"Ensemble of MLP")
mnist_results, xm_results, ob_results = experiment(convnet_structure,"conv")
make_plots(mnist_results,ob_results,xm_results,"Ensemble of ConvNets")

plt.show()
