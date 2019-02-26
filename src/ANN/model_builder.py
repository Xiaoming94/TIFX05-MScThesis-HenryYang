'''
model_builder.py
Author: Henry Yang (940503-1056)
'''

import keras.layers as nn_layers
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np
import json

def parse_model_js(jsconfig):
    config = json.loads(jsconfig)
    return build_model(config)

def build_model(config):
    input_layer, output_layer = build_layers(config)
    return Model(inputs=input_layer, outputs=output_layer)

def build_layers(config):
    input_layer = nn_layers.Input(shape=tuple(config["input_shape"]))
    output_layer = input_layer
    for lc in config["layers"]:
        output_layer = build_layer(lc)(output_layer)
    
    return input_layer, output_layer

def build_layer(layer_conf):
    ltype = layer_conf["type"]
    neuron_units = layer_conf["units"]
    activation = layer_conf["activation"] if "activation" in layer_conf else None
    if ltype == "Dense":
        return nn_layers.Dense(neuron_units,activation=activation)
    
    if ltype == "Conv2D":
        return nn_layers.Conv2D(neuron_units,tuple(layer_conf["kernel_size"]),activation=activation)
    
    if ltype == "BatchNormalization":
        return nn_layers.BatchNormalization(axis=layer_conf["axis"])
    
    if ltype == "MaxPooling2D":
        return nn_layers.MaxPooling2D(pool_size=layer_conf["pool_size"],strides=layer_conf["strides"])

    if ltype == "Flatten":
        return nn_layers.Flatten()
    
    if ltype == "Activation":
        return nn_layers.Activation(layer_conf["function"])

    if ltype == "Dropout":
        return nn_layers.Dropout(layer_conf["rate"])
    else:
        return None