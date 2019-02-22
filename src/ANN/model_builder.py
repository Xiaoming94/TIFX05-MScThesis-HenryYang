'''
model_builder.py
Author: Henry Yang (940503-1056)
'''

import keras.layers as nn_layers
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import json


def build_model(config_json):
    def build_layers(config):
 
        layers_configs = config["layers"]
        layers = [nn_layers.InputLayer(input_shape=tuple(config["input_shape"]))]
        for lc in layers_configs:
            layers += parse_layer_conf(lc)
        
        return layers
        
    config = json.loads(config_json)
    layers = build_layers(config)
    print(layers)
    model = Sequential(layers)
    model.compile(optimizer='adam', loss=loss_function(config['loss']), metrics=['accuracy'])
    return model

def parse_layer_conf(layer_conf):
    layer_type = layer_conf["type"]
    if layer_type == "Dense":
        return [nn_layers.Dense(
            layer_conf["units"],
            activation = layer_conf["activation"]         
        )]
    if layer_type == "Conv2D":
        return [
            nn_layers.Conv2D(
                layer_conf["units"],
                layer_conf["kernel_size"]
            ),
            nn_layers.BatchNormalization(axis=-1),
            nn_layers.Activation(layer_conf["activation"])
        ]
    else:
        return None
