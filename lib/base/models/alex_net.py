import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
import theano.tensor as T

import numpy as np

from layers import DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer



class AlexNet(object):

    def __init__(self, config):

        self.config = config
        self.verbose = self.config['rank'] == 0
        self.name = 'alexnet'
        batch_size = config['batch_size']
        flag_datalayer = config['use_data_layer']
        lib_conv = config['lib_conv']
        n_softmax_out=config['n_softmax_out']
        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        x = T.ftensor4('x')
        y = T.lvector('y')
        rand = T.fvector('rand')

        if self.verbose: print 'AlexNet 2/16'
        self.layers = []
        params = []
        weight_types = []

        if flag_datalayer:
            data_layer = DataLayer(input=x, image_shape=(3, 256, 256,
                                                         batch_size),
                                   cropsize=227, rand=rand, mirror=True,
                                   flag_rand=config['rand_crop'])

            layer1_input = data_layer.output
        else:
            layer1_input = x

        convpool_layer1 = ConvPoolLayer(input=layer1_input,
                                        image_shape=(3, 227, 227, batch_size),
                                        filter_shape=(3, 11, 11, 96),
                                        convstride=4, padsize=0, group=1,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, lrn=True,
                                        lib_conv=lib_conv,
                                        verbose = self.verbose
                                        )
        self.layers.append(convpool_layer1)
        params += convpool_layer1.params
        weight_types += convpool_layer1.weight_type

        convpool_layer2 = ConvPoolLayer(input=convpool_layer1.output,
                                        image_shape=(96, 27, 27, batch_size),
                                        filter_shape=(96, 5, 5, 256),
                                        convstride=1, padsize=2, group=2,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.1, lrn=True,
                                        lib_conv=lib_conv,
                                        verbose = self.verbose
                                        )
        self.layers.append(convpool_layer2)
        params += convpool_layer2.params
        weight_types += convpool_layer2.weight_type

        convpool_layer3 = ConvPoolLayer(input=convpool_layer2.output,
                                        image_shape=(256, 13, 13, batch_size),
                                        filter_shape=(256, 3, 3, 384),
                                        convstride=1, padsize=1, group=1,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        verbose = self.verbose
                                        )
        self.layers.append(convpool_layer3)
        params += convpool_layer3.params
        weight_types += convpool_layer3.weight_type

        convpool_layer4 = ConvPoolLayer(input=convpool_layer3.output,
                                        image_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 384),
                                        convstride=1, padsize=1, group=2,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        verbose = self.verbose
                                        )
        self.layers.append(convpool_layer4)
        params += convpool_layer4.params
        weight_types += convpool_layer4.weight_type

        convpool_layer5 = ConvPoolLayer(input=convpool_layer4.output,
                                        image_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 256),
                                        convstride=1, padsize=1, group=2,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        verbose = self.verbose
                                        )
        self.layers.append(convpool_layer5)
        params += convpool_layer5.params
        weight_types += convpool_layer5.weight_type

        fc_layer6_input = T.flatten(
            convpool_layer5.output.dimshuffle(3, 0, 1, 2), 2)
        fc_layer6 = FCLayer(input=fc_layer6_input, 
                            n_in=9216,
                            n_out=4096,
                            verbose = self.verbose
                            )
        self.layers.append(fc_layer6)
        params += fc_layer6.params
        weight_types += fc_layer6.weight_type

        dropout_layer6 = DropoutLayer(fc_layer6.output, 
                                      n_in=4096, 
                                      n_out=4096, 
                                      verbose = self.verbose)

        fc_layer7 = FCLayer(input=dropout_layer6.output, 
                            n_in=4096, 
                            n_out=4096,
                            verbose = self.verbose
                            )
        self.layers.append(fc_layer7)
        params += fc_layer7.params
        weight_types += fc_layer7.weight_type

        dropout_layer7 = DropoutLayer(fc_layer7.output, 
                                      n_in=4096, 
                                      n_out=4096,
                                      verbose = self.verbose)

        softmax_layer8 = SoftmaxLayer(input=dropout_layer7.output, 
                                      n_in=4096, 
                                      n_out=n_softmax_out,
                                      verbose = self.verbose)
        self.layers.append(softmax_layer8)
        params += softmax_layer8.params
        weight_types += softmax_layer8.weight_type

        # #################### NETWORK BUILT #######################
        self.p_y_given_x = softmax_layer8.p_y_given_x
        self.y_pred = softmax_layer8.y_pred
        
        
        self.cost = softmax_layer8.negative_log_likelihood(y)
        self.errors = softmax_layer8.errors(y)
        if n_softmax_out < 5:        
            self.errors_top_5 = softmax_layer8.errors_top_x(y, n_softmax_out)
        else:        
            self.errors_top_5 = softmax_layer8.errors_top_x(y, 5)       
        self.params = params
        self.x = x
        self.y = y
        self.rand = rand
        self.weight_types = weight_types
        self.batch_size = batch_size
        
        # training related
        self.lr = theano.shared(np.float32(config['learning_rate']))
        self.step_idx = 0
        self.mu = config['momentum'] # def: 0.9 # momentum
        self.eta = config['weight_decay'] #0.0002 # weight decay
        
        self.shared_x = theano.shared(np.zeros((3, config['input_width'], 
                                                  config['input_height'], 
                                                  config['batch_size']), 
                                                  dtype=theano.config.floatX),  
                                                  borrow=True)
                                              
        self.shared_y = theano.shared(np.zeros((config['batch_size'],), 
                                          dtype=int),   borrow=True)
                                          
        self.grads = T.grad(self.cost,self.params)
        
        # shared variable for storing momentum before exchanging momentum(delta w)
        self.vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in self.params]
        
        # shared variable for accepting momentum during exchanging momentum(delta w)
        self.vels2 = [theano.shared(param_i.get_value() * 0.)
            for param_i in self.params]
            
        self.train = None
        self.get_vel = None
        self.descent_vel = None
        self.val = None
        self.inference = None
        
    def set_dropout_off(self):
        
        DropoutLayer.SetDropoutOff()
    
    def set_dropout_on(self):
        
        DropoutLayer.SetDropoutOn()
        
    def compile_train(self, config, updates_dict):

        if self.verbose: print 'compiling training function...'
        
        x = self.x
        y = self.y
            
        cost = self.cost 
        error = self.errors
        errors_top_5 = self.errors_top_5
    
        shared_x, shared_y = self.shared_x, self.shared_y
               
        params = self.params  
        weight_types = self.weight_types 
        grads = self.grads
    
        updates_w,updates_v,updates_dv = updates_dict(config, model=self, 
                                    use_momentum=config['use_momentum'], 
                                    use_nesterov_momentum=config['use_nesterov_momentum'])  
                                      

    
        self.train= theano.function([], [cost,error], updates=updates_w,
                                              givens=[(x, shared_x), 
                                                      (y, shared_y)]
                                                                          )
    
   
        self.get_vel= theano.function([], [cost,error], updates=updates_v,
                                              givens=[(x, shared_x), 
                                                      (y, shared_y)]
                                                                          )
                                
                                                                                        
        self.descent_vel = theano.function([],[],updates=updates_dv)     

        
    def compile_inference(self):

        if self.verbose: print 'compiling inference function...'
    
        x = self.x
        
        output = self.output
    
        self.inference = theano.function([x],output)
        
    def compile_val(self):

        if self.verbose: print 'compiling validation function...'
    
        x = self.x
        y = self.y
            
        cost = self.cost 
        error = self.errors
        errors_top_5 = self.errors_top_5
        
        shared_x, shared_y = self.shared_x, self.shared_y
        
        self.val =  theano.function([], [cost,error,errors_top_5], updates=[], 
                                          givens=[(x, shared_x),
                                                  (y, shared_y)]
                                                                )                                                       
    def adjust_lr(self, epoch, val_error_list = None):
    
        if self.config['lr_policy'] == 'step':
            
            if epoch >=20 and epoch < 40 and self.step_idx==0:

                self.lr.set_value(
                    np.float32(self.lr.get_value() / 10))
                if self.verbose: print 'Learning rate divided by 10'
                self.step_idx = 1
                
            elif epoch >=40 and epoch < 60 and self.step_idx==1:
                
                self.lr.set_value(
                    np.float32(self.lr.get_value() / 10))
                if self.verbose: print 'Learning rate divided by 10'
                self.step_idx = 2
                
            elif epoch >=60 and epoch < 70 and self.step_idx==2:
                
                self.lr.set_value(
                    np.float32(self.lr.get_value() / 10))
                if self.verbose: print 'Learning rate divided by 10'
                self.step_idx = 3
            else:
                pass 
                
        if self.config['lr_policy'] == 'auto':
            if epoch>5 and (val_error_list[-3] - val_error_list[-1] <
                                self.config['lr_adapt_threshold']):
                self.lr.set_value(
                    np.float32(self.lr.get_value() / 10))
        
        if self.verbose: 
            print 'Learning rate now: %.10f' % np.float32(self.lr.get_value())
    
        

'''
def compile_models(model, config, flag_top_5=False):

    x = model.x
    y = model.y
    
#    y_pred = model.y_pred########ADDED##############
#    p_y_given_x = model.p_y_given_x########ADDED##############
    
    
    rand = model.rand
    weight_types = model.weight_types

    cost = model.cost
    params = model.params
    errors = model.errors
    errors_top_5 = model.errors_top_5
    batch_size = model.batch_size

    mu = config['momentum']
    eta = config['weight_decay']

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []

    learning_rate = theano.shared(np.float32(config['learning_rate']))
    lr = T.scalar('lr')  # symbolic learning rate

    if config['use_data_layer']:
        raw_size = 256
    else:
        raw_size = 227

    shared_x = theano.shared(np.zeros((3, raw_size, raw_size,
                                       batch_size),
                                      dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.zeros((batch_size,), dtype=int),
                             borrow=True)

    rand_arr = theano.shared(np.zeros(3, dtype=theano.config.floatX),
                             borrow=True)

    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]

    if config['use_momentum']:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in \
                zip(params, grads, vels, weight_types):

            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")

            if config['use_nesterov_momentum']:
                vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
            else:
                vel_i_next = mu * vel_i - real_lr * real_grad

            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i + vel_i_next))

    else:
        for param_i, grad_i, weight_type in zip(params, grads, weight_types):
            if weight_type == 'W':
                updates.append((param_i,
                                param_i - lr * grad_i - eta * lr * param_i))
            elif weight_type == 'b':
                updates.append((param_i, param_i - 2 * lr * grad_i))
            else:
                raise TypeError("Weight Type Error")

    # Define Theano Functions

    train_model = theano.function([], cost, updates=updates,
                                  givens=[(x, shared_x), (y, shared_y),
                                          (lr, learning_rate),
                                          (rand, rand_arr)])
    train_error = theano.function(
        [], errors, givens=[(x, shared_x), (y, shared_y), (rand, rand_arr)])
        
        
    validate_outputs = [cost, errors]
    if flag_top_5:
        validate_outputs.append(errors_top_5)

    validate_model = theano.function([], validate_outputs,
                                     givens=[(x, shared_x), (y, shared_y),
                                             (rand, rand_arr)])

#    y_error_info = theano.function([],[y_pred,p_y_given_x,y,errors_top_5], givens=[(x, shared_x), (y, shared_y),
#                                             (rand, rand_arr)])
                                             

    return (train_model, validate_model, train_error,
            learning_rate, shared_x, shared_y, rand_arr, vels)
            
            
'''
