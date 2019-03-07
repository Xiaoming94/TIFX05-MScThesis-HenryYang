import utils
import ANN as ann

network_model = """
{
    "input_shape" : [28,28,1],
    "layers" : [
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
            "type" : "MaxPooling2D",
            "pool_size" : [2,2],
            "strides" : [2,2]
        },
        {
            "type" : "Flatten"
        },
        {
            "type" : "Dense",
            "units" : 32,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax"
        }
    ]
}
"""

mnist_linethickness = 66.97000583000295
epochs = 5

xtrain,ytrain,xtest,ytest = utils.load_mnist()
xtrain,xtest = xtrain.reshape(60000,28,28,1),xtest.reshape(10000,28,28,1)
xtrain,ytrain,xval,yval = utils.create_validation(xtrain,ytrain,1/6)
utils.setup_gpu_session()



model = ann.parse_model_js(network_model)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(xtrain,ytrain, verbose=1, validation_data=(xval,yval))

print(ann.test_model(model, xtest, ytest, "accuracy"))
print(ann.test_model(model, xtest, ytest, "c_error"))
