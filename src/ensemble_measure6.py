import utils
import ANN as ann
import keras.losses as klosses
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

ensemble_size = 20

trials = 5

network_model2 = '''
{
    "input_shape" : [28,28,1],
    "layers" : [
        {
            "type" : "Conv2D",
            "units" : 16,
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
'''

network_model1 = '''
{
    "input_shape" : [784],
    "layers" : [
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 200,
            "activation" : "relu"
        },
        {
            "type" : "Dense",
            "units" : 10,
            "activation" : "softmax"
        }
    ]
}
'''
def calc_vote_entropy(model_list, ensemble_size, digits):
    vote_mat = np.zeros([digits.shape[0],10])
    for m in model_list:
        pred = m.predict(digits)
        votes = ann.classify(pred)
        vote_mat += votes
    
    vote_rate = vote_mat * 1/ensemble_size
    return np.mean(entropy(vote_rate.transpose()))
    


def test_digits(model, model_list, ensemble_size ,digits, labels):
    l_c_errors = []
    l_pred_entropy = []
    for d in digits:
        c_error = ann.test_model(model, [d]*ensemble_size, labels, metric = 'c_error')
        entropy = ann.test_model(model, [d]*ensemble_size, labels, metric = 'entropy')
        l_c_errors.append(c_error)
        l_pred_entropy.append(entropy)
    
    return l_c_errors, l_pred_entropy


def experiment(network_model, reshape_mode = 'mlp'):
    reshape_funs = {
        "conv" : lambda d : d.reshape(-1,28,28,1),
        "mlp" : lambda d : d.reshape(-1,784)
    }
    xtrain,ytrain,xtest,ytest = utils.load_mnist()
    reshape_fun = reshape_funs[reshape_mode]
    xtrain,xtest = reshape_fun(xtrain),reshape_fun(xtest)

    custom_digits_dict = utils.load_processed_data("combined_testing_data")
    digits_labels = custom_digits_dict['labels']
    digits_taus = list(custom_digits_dict.keys())[:-1]
    digits_data = list(map(reshape_fun, [custom_digits_dict[t] for t in digits_taus]))
    digits_data = list(map(utils.normalize_data, digits_data))
    digits_labels = utils.create_one_hot(digits_labels.astype('uint'))

    for t in range(trials):
        print("===== START TRIAL %s =====" % (t + 1))
        # Preparing Results
        # Classification Error
        ensemble_cerror = []
        #t_ensemble_adv_cerror = []
        digits_cerror = []
        #t_digits_adv_cerror = []

        # Prediction Entropy
        entropy_ensemble = []
        #t_entropy_adv_ensemble = []
        digits_entropy = []
        #t_digits_adv_entropy = []

        # Voting entropy
        #entropy_vote = []
        #entropy_adv_vote = []
        #digits_vote = []
        #digits_adv_entropy = []

        epochs = 3

        
        print('Working now with ensemble of size m = %s' % m)
        l_xtrain = []
        l_xval = []
        l_ytrain = []
        l_yval = []
        for _ in range(ensemble_size):
            t_xtrain,t_ytrain,t_xval,t_yval = utils.create_validation(xtrain,ytrain,(1/6))
            l_xtrain.append(t_xtrain)
            l_xval.append(t_xval)
            l_ytrain.append(t_ytrain)
            l_yval.append(t_yval)
        # Without adveserial training
        inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([network_model], pop_per_type=ensemble_size, merge_type="Average")
        losses = list(
            map( lambda m : ann.adveserial_loss(klosses.categorical_crossentropy,m,eps=0.01), model_list)
        )
        train_model.compile(optimizer="adam", loss=losses, metrics = ['acc'])
        train_model.fit(x=l_xtrain,y=l_ytrain, verbose=1,batch_size=100, epochs = epochs,validation_data=(l_xval,l_yval))
        merge_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
        c_error = ann.test_model(merge_model, [xtest]*ensemble_size, ytest, metric = 'accuracy' )
        entropy = ann.test_model(merge_model, [xtest]*ensemble_size, ytest, metric = 'entropy' )
        ensemble_cerror.append(c_error)
        entropy_ensemble.append(entropy)
        #entropy_vote.append(calc_vote_entropy(model_list,m,xtest))
        d_cerror,d_entropy = test_digits(merge_model,model_list,m,digits_data,digits_labels)
        digits_cerror.append(d_cerror)
        digits_entropy.append(d_entropy)

            #digits_vote.append(d_vote)
            # Adveserial training

            #inputs, outputs, train_model, model_list, merge_model = ann.build_ensemble([network_model], pop_per_type=m, merge_type="Average")
            #losses = list(
            #    map( lambda m : ann.adveserial_loss(klosses.categorical_crossentropy,m,eps=0.01), model_list)
            #)
            #train_model.compile(optimizer="adam", loss=losses, metrics = ['acc'])
            #train_model.fit(x=l_xtrain, y=l_ytrain, verbose=1, epochs = epochs ,validation_data=(l_xval,l_yval))
            #c_error = ann.test_model(merge_model, [xtest]*m, ytest, metric = 'c_error' )
            #entropy = ann.test_model(merge_model, [xtest]*m, ytest, metric = 'entropy' )
#
            #t_ensemble_adv_cerror.append(c_error)
            #t_entropy_adv_ensemble.append(entropy)
#
            #d_cerror,d_entropy,d_vote = test_digits(merge_model,model_list,m,digits_data,digits_labels)
#
            #t_digits_adv_cerror.append(d_cerror)
            #t_digits_adv_entropy.append(d_entropy)

        filename1 = 'adv_mnist_results_5-20-50-trial%s' % t
        filename2 = 'adv_digits_results_5-20-50-trial%s' % t

        mnist_results = {
            'ensemble_cerror' : ensemble_cerror,
            'ensemble_entropy' : entropy_ensemble
            #'ensemble_adv_cerror' : ensemble_adv_cerror,
            #'ensemble_adv_entropy' : entropy_adv_ensemble,
            #'voting_entropy' : entropy_vote
            #'voting_adv_entropy' : entropy_adv_vote
        }

        digits_results = {
            'ensemble_cerror' : digits_cerror,
            'ensemble_entropy' : digits_entropy
            #'ensemble_adv_cerror' : digits_adv_cerror,
            #'ensemble_adv_entropy' : digits_adv_entropy,
            #'voting_entropy' : digits_vote
            #'voting_adv_entropy' : digits_adv_vote
        }
    

        utils.save_processed_data(mnist_results,filename1)
        utils.save_processed_data(digits_results,filename2)

    return digits_taus, mnist_results, digits_results

utils.setup_gpu_session()

taus, mnist_results ,digits_results = experiment(network_model2, 'conv')
