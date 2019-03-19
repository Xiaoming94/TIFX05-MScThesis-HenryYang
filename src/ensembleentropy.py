import utils
import ANN as ann
import numpy as np

network_structure = '''
{
    "input_shape" : [2],
    "layers" : [
        {
            "type" : "Dense",
            "units" : 8,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 16,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 2,
            "activation" : "softmax"
        }     
    ]
}
'''
datapoints = 200000
ensemble_size = 10

def calc_shannon_entropy(pred):
    bitsmat = pred * np.log2(pred)
    bits = -1 * np.sum(bitsmat, axis=1)
    return bits

def create_onehot_labels(labels):
    one_hot_labels = np.zeros([labels.shape[0],2])
    one_hot_labels[np.where(labels == -1),1] = 1
    one_hot_labels[np.where(labels == 1),0] = 1
    return one_hot_labels

xtrain,ytrain,xtest,ytest = utils.henon_map_data(datapoints,int(datapoints/10))
ytrain,ytest = create_onehot_labels(ytrain),create_onehot_labels(ytest)

xtrain,ytrain,xval,yval = utils.create_validation(xtrain,ytrain)

inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([network_structure],pop_per_type=ensemble_size,merge_type="Average")
train_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
train_model.fit([xtrain]*ensemble_size,[ytrain]*ensemble_size, verbose=1, validation_data=([xval]*ensemble_size,[yval]*ensemble_size),epochs=3)

accuracy = ann.test_model(merge_model, [xtest]*ensemble_size, ytest)
entropy = calc_shannon_entropy(merge_model.predict([xtest]*ensemble_size))
print("====== Current results: ======")
print("Accuracy : %s" % (accuracy))
print("Total Entropy (sum) : %s" % np.sum(entropy))
print("Mean Entropy : %s" % np.mean(entropy))