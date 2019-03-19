import numpy as np
from keras.datasets import mnist
import pickle
import os

def normalize_data(data):
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

def load_mnist(normalize = True):
    (xtrain,ytrain),(xtest,ytest) = mnist.load_data()
    if normalize:
        xtrain,xtest = normalize_data(xtrain),normalize_data(xtest)
    ytrain,ytest = create_one_hot(ytrain),create_one_hot(ytest)

    return xtrain,ytrain,xtest,ytest

def create_validation(data,labels,validation_ratio=0.1):
    data_size = data.shape[0]
    n_valdata = int(data_size * validation_ratio)
    randperm = np.random.permutation(data_size)
    val_data, val_labels = data[randperm[0:n_valdata]], labels[randperm[0:n_valdata]]
    train_data, train_labels = data[randperm[n_valdata:]], labels[randperm[n_valdata:]]
    return train_data, train_labels, val_data, val_labels

def pickle_filepath_str(name):
    if not os.path.isdir(os.path.join(".","data")):
        os.mkdir(os.path.join(".","data"))
    return os.path.join(".","data",name + ".pickle")

def save_processed_data(dig_dict, name):
    full_filename = pickle_filepath_str(name)
    with open(full_filename, 'wb+') as fhandle:
        pickle.dump(dig_dict, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

def load_processed_data(name):
    with open(pickle_filepath_str(name), 'rb') as fhandle:
        return pickle.load(fhandle)

def calc_henon(x0,y0,steps,a=1.4, b=0.3):
    x_vec = np.zeros(3+steps)
    y_vec = np.zeros(3+steps)

    x = x0
    y = y0

    for t in range(3+steps):
        x_vec[t] = x
        y_vec[t] = y
        xn = 1 - a * x * x + y
        yn = b * x
        x = xn
        y = yn

    return x_vec[3:], y_vec[3:]


def henon_map_data(count=60000, testsize=10000 ,ratio=0.5):
    pos_size = int(count * ratio)
    neg_size = count - pos_size

    hx,hy = calc_henon(0,0,pos_size)
    pos_data = np.transpose(np.array([hx,hy]))
    neg_data = np.transpose(np.array([-hx,hy-0.2]))

    ran_indices = np.random.permutation(int(count*ratio))
    test_pos_data = pos_data[ran_indices[0:int(testsize * ratio)]]
    test_neg_data = neg_data[ran_indices[0:int(testsize * ratio)]]
    train_pos_data = pos_data[ran_indices[int(testsize * ratio):]]
    train_neg_data = neg_data[ran_indices[int(testsize * ratio):]]
    
    Xtrain = np.concatenate([train_pos_data,train_neg_data],axis=0)
    Xtest = np.concatenate([test_pos_data,test_neg_data],axis=0)
    Ytrain = np.zeros(count-testsize) - 1
    Ytrain[0:int((count-testsize)*ratio)] = 1
    Ytest = np.zeros(testsize) - 1
    Ytest[0:int(testsize * ratio)] = 1
    return Xtrain, Ytrain, Xtest, Ytest