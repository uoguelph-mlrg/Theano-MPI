'''
This test case is for testing the equivalence when building theano functions: 

        
        forward() + backward() = train()


so that we can separate train() function for use in "cdd" and hold the equivalence

result shows that:

     the result of "t" does not equal to "f+b" as of July 21 using Theano-0.8.0 (by sourcing set4theano.sh on cop1)

'''

import sys

sys.path.append("..")

from mlp_base.mlp import MLP
from mlp_base.logistic_sgd import load_data

import theano
import theano.tensor as T

def _compile(classifier, batch_size, x, y, index, train_set_x, train_set_y, L1_reg=0.00, L2_reg=0.0001, learning_rate=0.01):

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )


    gparams = [T.grad(cost, param) for param in classifier.params]


    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]


    train = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    shared_grads = [theano.shared(param.get_value()) for param in classifier.params]
    updates_save_grad = [(shared_grad, gparam) for shared_grad, gparam in zip (shared_grads,gparams)]
    get_gradient = theano.function(
        inputs=[index],
        outputs=[cost],
        updates=updates_save_grad,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    updates_descent = [
        (param, param - learning_rate * shared_grad)
        for param, shared_grad in zip(classifier.params, shared_grads)
    ]

    descent = theano.function([],[], updates = updates_descent)
    
    
    return train, get_gradient, descent
        
        

def test(n_epochs=10, dataset='../mlp_base/mnist.pkl.gz', batch_size=20, n_hidden=500):
             
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    

        
    
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    import numpy
    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier_0 = MLP(
        rng=rng,
        input=x, #.reshape((batch_size, 1, 28, 28)),
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )
        
        
    train_0, _ , _  = _compile(classifier_0, 1, x, y, index, train_set_x, train_set_y)
    
    
    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier_1 = MLP(
        rng=rng,
        input=x, #.reshape((batch_size, 1, 28, 28)),
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )
        
        
    _ , forward_1, backward_1 = _compile(classifier_1, 1, x, y, index, train_set_x, train_set_y)

    epoch = 0

    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            cost_0 = train_0(minibatch_index)
            
            # print cost and some updated parameter values for comparison
            print "t  :", cost_0, classifier_0.params[0].get_value()[0][1:5]
            
            
            ####----------------------------
            
            
            cost_1 = forward_1(minibatch_index)
            backward_1()
            
            # print cost and some updated parameter values for comparison
            print "f+b:", cost_1[0] ,classifier_1.params[0].get_value()[0][1:5]
            
            print ""


if __name__ == '__main__':
    
    test()
    


