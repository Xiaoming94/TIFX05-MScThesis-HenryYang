import numpy as np
from keras.datasets import mnist

def normalize(data):
    """
    This function normalizes the input data between 0 and 1 and center the data around origin
    Assumes that the data points consist of integers between 0 and 255

    Parameters:
    data (np.array(np.int)):  The data points

    Returns:
    input data normalized between 0 and 1 centered around origin
    """

    normalized_data = data * (1/255)
    return normalized_data

def create_one_hot(labels):
    """
    This function creates one-hot encoded vectors of integer valued labels for number classification.

    Parameters:
    labels (np.array(int)):  Integer labels of the handwritten digits

    returns:
    A matrix of one-hot encoded vectors
    """

    num_labels = labels.size
    m_onehot = np.zeros([num_labels,10])
    for (i,t) in zip(list(range(num_labels)),labels):
        m_onehot[i,t] = 1
    
    return m_onehot

def load_mnist():
    (xtrain,ytrain),(xtest,ytest) = mnist.load_data()
    xtrain,xtest = normalize(xtrain),normalize(xtest)
    ytrain,ytest = create_one_hot(ytrain),create_one_hot(ytest)

    return xtrain,ytrain,xtest,ytest