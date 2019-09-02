'''
model_builder.py
Author: Henry Yang (940503-1056)
'''

import keras.layers as nn_layers
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np
import json
from keras import regularizers as reg

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

def build_ensemble(net_configs, pop_per_type = 5, merge_type = None):
    
    merge_layers = {
        "Average" : nn_layers.Average()
    }
    
    inputs = []
    outputs = []
    for net_conf in net_configs:
        for i in range(pop_per_type):
            il, ol = build_layers(json.loads(net_conf))
            inputs.append(il)
            outputs.append(ol)
    
    train_model = Model(inputs=inputs, outputs=outputs)
    if len(outputs) > 1:
        merge_layer = merge_layers[merge_type] if merge_type != None else None
        merge_model = Model(inputs=inputs, outputs=merge_layer(outputs)) if merge_layer != None else None
    else: # There is only 1 member in the Ensemble
        merge_model = train_model

    model_list = [Model(inputs=il,outputs=ol) for il,ol in zip(inputs,outputs)]
    return inputs, outputs, train_model, model_list, merge_model

def build_regularizer(config):
    if config['type'] == 'l2':
        return reg.l2(config['lambda'])
    elif config['type'] == 'l1':
        return reg.l1(config['lambda'])

    else: 
        return None

def build_layer(layer_conf):
    ltype = layer_conf["type"]
    neuron_units = layer_conf["units"] if "units" in layer_conf else None
    activation = layer_conf["activation"] if "activation" in layer_conf else None
    ar = build_regularizer(layer_conf['activity_regularizer']) if 'activity_regularizer' in layer_conf else None
    kr = build_regularizer(layer_conf['kernel_regularizer']) if 'kernel_regularizer' in layer_conf else None
    br = build_regularizer(layer_conf['bias_regularizer']) if 'bias_regularizer' in layer_conf else None

    if ltype == "Dense":
        return nn_layers.Dense(neuron_units,activation=activation, activity_regularizer = ar,bias_regularizer= br, kernel_regularizer = kr)
    
    if ltype == "Conv2D":
        return nn_layers.Conv2D(neuron_units,tuple(layer_conf["kernel_size"]),activation=activation, activity_regularizer = ar,bias_regularizer= br, kernel_regularizer = kr)
    
    if ltype == "BatchNormalization":
        return nn_layers.BatchNormalization(axis=layer_conf["axis"])
    
    if ltype == "MaxPooling2D":
        strides = tuple(layer_conf["strides"]) if type(layer_conf["strides"]) is list else layer_conf["strides"]
        return nn_layers.MaxPooling2D(pool_size=tuple(layer_conf["pool_size"]),strides=strides)

    if ltype == "Flatten":
        return nn_layers.Flatten()
    
    if ltype == "Activation":
        return nn_layers.Activation(layer_conf["function"])

    if ltype == "Dropout":
        return nn_layers.Dropout(layer_conf["rate"])
    else:
        return None