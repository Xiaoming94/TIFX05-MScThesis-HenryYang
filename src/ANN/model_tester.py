from keras.models import Sequential, Model
import numpy as np
from scipy.stats import entropy

def test_model(model, test_data, test_labels, metric = "accuracy"):
    if metric == "accuracy":
        metric = test_accuracy
    if metric == "c_error":
        metric = test_classification_err
    if metric == "entropy":
        metric = shannon_entropy

    return metric(model,test_data, test_labels)

def classify(pred):
    class_index = np.argmax(pred,1)
    classes = np.zeros(pred.shape)
    for i,j in zip(range(class_index.size),class_index):
        classes[i,j] = 1
    return classes   
    
def test_accuracy(model,test_data, test_labels):
    if "acc" in model.metrics_names:
        [_,accuracy] = model.evaluate(x=test_data,y=test_labels,verbose=0)
    else:
        predictions = model.predict(test_data)
        correct = np.equal(np.argmax(predictions,1),np.argmax(test_y,1))
        accuracy = np.mean(correct)
    return accuracy

def test_classification_err(model, test_data, test_labels):
    predictions = model.predict(test_data)
    num_data = predictions.shape[0]
    classes = classify(predictions)
    diff = classes - test_labels
    c_err = (1/(2 * num_data)) * np.sum(np.sum(np.abs(diff)))
    return c_err

def shannon_entropy(model, test_data, test_labels):
    predictions = model.predict(test_data)
    bits = np.array(list(map(entropy, predictions)))
    #bits = entropy(predictions.transpose())
    return np.mean(bits)