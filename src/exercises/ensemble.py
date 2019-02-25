import keras.layers as nn_layers
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.backend.tensorflow_backend import set_session

ensemble_size = 3

def setup_keras():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

def process_data(data_x, data_y):
    # Normalize the data
    ndata_x = data_x * (1/255)

    # onehot encode labels
    length_y = data_y.size
    labels_y = np.zeros([length_y,10])
    for i,j in zip(list(range(length_y)),data_y.astype(np.int32)):
        labels_y[i,j] = 1
    
    return ndata_x,labels_y

def create_model():
    net_input = nn_layers.Input(shape=(28,28,1))
    net_layers = nn_layers.Conv2D(64, (3,3),activation="relu")(net_input)
    net_layers = nn_layers.BatchNormalization(axis=-1)(net_layers)
    net_layers = nn_layers.Conv2D(64, (3,3),activation="relu")(net_layers)
    net_layers = nn_layers.BatchNormalization(axis=-1)(net_layers)
    net_layers = nn_layers.MaxPooling2D(pool_size=(2,2),strides=2,padding="valid")(net_layers)
    net_layers = nn_layers.Flatten()(net_layers)
    net_layers = nn_layers.Dense(10,activation="softmax")(net_layers)
    return net_input, net_layers

## LOADING THE MNIST DATA
(x_train,y_train),(x_test,y_test) = mnist.load_data()

## RESHAPING THE DATA to suitable forms for the CNN
x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)

## Processing the data, and isolates a validation set
ntrain_x, oh_trainy = process_data(x_train,y_train)

randidx = np.random.permutation(y_train.size)
val_x,val_y = ntrain_x[randidx[0:10000]],oh_trainy[randidx[0:10000]]
train_x,train_y = ntrain_x[randidx[10000:]],oh_trainy[randidx[10000:]]

models = []
inputs = []
outputs = []

setup_keras()

for i in range(ensemble_size):
    net_input, net_output = create_model()
    inputs.append(net_input)
    outputs.append(net_output)
    model = Model(inputs=net_input, outputs=net_output)

    models.append(model)

ensemble_model = nn_layers.Average()(outputs)
ensemble_model = Model(inputs=inputs, outputs=ensemble_model)
multi_model = Model(inputs=inputs, outputs=outputs)
multi_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
multi_model.fit([train_x,train_x,train_x],[train_y,train_y,train_y],verbose=1, validation_data=([val_x,val_x,val_x],[val_y,val_y,val_y]),epochs=5)

ntest_x,test_y = process_data(x_test,y_test)
predictions = ensemble_model.predict([ntest_x,ntest_x,ntest_x])

correct = np.equal(np.argmax(predictions,1),np.argmax(test_y,1))
accuracy = np.mean(correct)

print("The Ensemble Accuracy on the Test set is: %s" % accuracy)

model_index = 0

for m in models:
    model_index += 1
    predictions = m.predict(ntest_x)

    correct = np.equal(np.argmax(predictions,1),np.argmax(test_y,1))
    accuracy = np.mean(correct)

    print("model %s : Accuracy: %s" % (model_index,accuracy))


