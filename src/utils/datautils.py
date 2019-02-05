import numpy as np

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