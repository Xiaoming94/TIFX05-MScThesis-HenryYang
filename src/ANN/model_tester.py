from keras.models import Sequential, Model
import numpy as np

def test_model(model, test_data, test_labels, metric = "accuracy"):
    if metric == "accuracy":
        metric = test_accuracy
    if metric == "c_error":
        metric = test_classification_err
    
    predictions = model.predict(test_data)
    return metric(predictions, test_labels)
        
    
def test_accuracy(test_pred, test_labels):
    correct = np.equal(np.argmax(test_pred,1),np.argmax(test_labels,1))
    accuracy = np.mean(correct)
    return accuracy

def test_classification_err(test_pred, test_labels):
    num_data = test_pred.shape[0]
    diff = test_pred - test_labels
    c_err = (1/(2 * num_data)) * np.sum(np.sum(np.abs(diff)))
    return c_err