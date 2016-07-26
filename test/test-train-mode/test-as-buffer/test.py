'''
This test case is for testing the possiblity of using as_buffer function on CudaNdarray: 

        
        "the pointer to the memory buffer got from CudaNdarray.as_buffer() should not change during graph execution"  


so that we need to call this only once to get the pointer and use it during parameter exchanging. (Will theano change this pointer during graph execution by garbage collection?)

result shows that:

     no, it's defined as mutable in cuda_ndarray.cuh file in struct CudaNdarray

'''

from mlp import MLP
from logistic_sgd import load_data

import theano
import theano.tensor as T

import numpy

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
        outputs=cost,
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
                        
    
    
    rng = numpy.random.RandomState(1234)

    # classifier working on full minibatch
    classifier_full = MLP(
        rng=rng,
        input=x, #.reshape((batch_size, 1, 28, 28)),
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )
    
    train_full, _ , _  = _compile(classifier_full, 2, x, y, index, train_set_x, train_set_y)
    
    
    rng = numpy.random.RandomState(1234)

     # classifier working on half minibatch
    classifier_half0 = MLP(
        rng=rng,
        input=x, #.reshape((batch_size, 1, 28, 28)),
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )
        
    train_half0, _ , _  = _compile(classifier_half0, 1, x, y, index, train_set_x, train_set_y)
    
    
    
    rng = numpy.random.RandomState(1234)

    # classifier working on half minibatch
    classifier_half1 = MLP(
        rng=rng,
        input=x, #.reshape((batch_size, 1, 28, 28)),
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )
        
    train_half1, _ , _  = _compile(classifier_half1, 1, x, y, index, train_set_x, train_set_y)
    
    
    

    epoch = 0

    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            cost_0 = train_full(minibatch_index)
            
            # print cost and some updated parameter values for comparison
            print "full:", cost_0, classifier_full.params[0].get_value()[0][1:5]
            
            
            ####----------------------------
            
            
            cost_00 = train_half0(minibatch_index*2)
            
            
            cost_01 = train_half1(minibatch_index*2+1)
            
            # print cost and some updated parameter values for comparison
            print "h0h1:", (cost_00+cost_01)/2. , (classifier_half0.params[0]+classifier_half1.params[0]).eval()[0][1:5]*0.5
            
            
            for param_0, param_1 in zip(classifier_half0.params,classifier_half1.params):
            
                avg_param = (param_0.get_value()+param_1.get_value())*0.5
            
                param_0.set_value(avg_param)
                param_1.set_value(avg_param)
                
                return param_0
                
            
            print ""
            
            
            
            



#if __name__ == '__main__':
    
arr=test().container.data



