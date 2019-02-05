import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

"""
kerasutils.py

Author: Henry Yang

Utility file with functions for setting up Keras with specific settings and other useful functions
"""

def setup_gpu_session(growth = True):
    """
    Function that setups Keras to work with the GPU growth
    Can be used to configure GPU growth
    """
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = growth
    
    sess = tf.Session(config = cfg)
    set_session(sess)