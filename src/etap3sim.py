import ANN as ann
import utils
import json
import keras.losses as klosses

network_model1 = """
{
    "input_shape" : [28,28,1],
    "layers" : [
        {
            "type" : "Conv2D",
            "units" : 64,
            "kernel_size" : [3,3],
            "activation" : "relu"
        },
        {
            "type" : "BatchNormalization",
            "axis" : -1
        },
        {
            "type" : "Conv2D",
            "units" : 32,
            "kernel_size" : [3,3],
            "activation" : "relu"
        },
        {
            "type" : "BatchNormalization",
            "axis" : -1
        },
        {
            "type" : "MaxPooling2D",
            "pool_size" : [2,2],
            "strides" : [2,2]
        },
        {
            "type" : "Flatten"
        },
        {
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax"
        }
    ]
}
"""
utils.setup_gpu_session(True)
xtrain,ytrain,xtest,ytest = utils.load_mnist()
xtrain = xtrain.reshape(60000,28,28,1)
xtest = xtest.reshape(10000,28,28,1)

#model = ann.parse_model_js(network_model1)

#model.compile(optimizer="adam", loss=ann.adveserial_loss(klosses.categorical_crossentropy,model,model.inputs), metrics=["accuracy"])
#model.fit(xtrain,ytrain,verbose=1,validation_data=(xtest,ytest),epochs=3)
#print("SCRIPT RUN SUCCÈSSFUL")

model_conf = json.loads(network_model1)

inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([model_conf])

ensemble_size = len(model_list)


lossfunctions = [ann.adveserial_loss(klosses.categorical_crossentropy, m) for m in model_list]
train_model.compile(optimizer="adam",loss=lossfunctions ,metrics=["accuracy"])
train_model.fit([xtrain]*ensemble_size,[ytrain]*ensemble_size,verbose=1,validation_data=([xtest]*ensemble_size,[ytest]*ensemble_size),epochs=3)



