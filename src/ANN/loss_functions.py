"""
loss_functions.py
Author: Henry Yang (940503-1056)
"""

from keras import backend as K

def adveserial_loss(loss_function, model, eps = 0.01):
    """
    Loss function used when adveserial training is to be used.
    This function is essentially a wrapper function around a loss function
    that also generates an adveserial example using the fast graident loss method

    Parameters:
    loss_function : The loss function that is used during training
    model : The Model used for training
    eps : (Default 0.01), Small constant used when generating the adveserial example

    Returns:
    A loss function value that is result from the wrapper
    """
    def loss_wrapper(y_true, y_pred):
        loss = loss_function(y_true, y_pred)
        grad = K.gradients(loss,model.inputs)
        xb = [x + eps * dx for (x,dx) in zip(model.inputs,grad)]
        advloss = loss_function(y_true, model(xb))
        return loss + advloss
    return loss_wrapper