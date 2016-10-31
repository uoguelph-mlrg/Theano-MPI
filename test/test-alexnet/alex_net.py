# This version of alex_net.py is modified based on the theano_alexnet project. See the original project here:
# https://github.com/uoguelph-mlrg/theano_alexnet, and its copy right:
# Copyright (c) 2014, Weiguang Ding, Ruoyan Wang, Fei Mao and Graham Taylor
# All rights reserved.

import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
import theano.tensor as T

import numpy as np

from layers import DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer

# from modelbase import ModelBase


class AlexNet(object):

    def __init__(self, config):
        ModelBase.__init__(self)

        self.config = config
        self.verbose = self.config['verbose']
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
        lr = T.scalar('lr')

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
        
        self.output = self.p_y_given_x
        
        
        self.cost = softmax_layer8.negative_log_likelihood(y)
        self.error = softmax_layer8.errors(y)
        if n_softmax_out < 5:        
            self.error_top_5 = softmax_layer8.errors_top_x(y, n_softmax_out)
        else:        
            self.error_top_5 = softmax_layer8.errors_top_x(y, 5)       
        self.params = params
        
        # inputs
        self.x = x
        self.y = y
        self.rand = rand
        self.lr = lr
        self.shared_x = theano.shared(np.zeros((3, config['input_width'], 
                                                  config['input_height'], 
                                                  config['file_batch_size']), # for loading large batch
                                                  dtype=theano.config.floatX),  
                                                  borrow=True)
                                              
        self.shared_y = theano.shared(np.zeros((config['file_batch_size'],), 
                                          dtype=int),   borrow=True)
        self.shared_lr = theano.shared(np.float32(config['learning_rate']))
        
        # training related
        self.base_lr = np.float32(config['learning_rate'])
        self.step_idx = 0
        self.mu = config['momentum'] # def: 0.9 # momentum
        self.eta = config['weight_decay'] #0.0002 # weight decay
        self.weight_types = weight_types
        self.batch_size = batch_size

                                          
        self.grads = T.grad(self.cost,self.params)
        
        subb_ind = T.iscalar('subb')  # sub batch index
        #print self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size].shape.eval()
        self.subb_ind = subb_ind
        self.shared_x_slice = self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
        self.shared_y_slice = self.shared_y[subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
        
    def set_dropout_off(self):
        
        DropoutLayer.SetDropoutOff()
    
    def set_dropout_on(self):
        
        DropoutLayer.SetDropoutOn()
        
    def compile_train(self):

        if self.verbose: print 'compiling training function...'
        
        for arg_list in self.compile_train_fn_list:
            self.compiled_train_fn_list.append(theano.function(**arg_list))
        
        if self.config['monitor_grad']:
            
            norms = [grad.norm(L=2) for grad in self.grads]
            
            self.get_norm = theano.function([self.subb_ind], norms,
                                              givens=[(self.x, self.shared_x_slice), 
                                                      (self.y, self.shared_y_slice)]
                                                                          )
        
    def compile_inference(self):

        if self.verbose: print 'compiling inference function...'
    
        self.inference = theano.function([self.x, self.y],self.error_top_5)
        
    def compile_val(self):

        if self.verbose: print 'compiling validation function...'
        
        self.val =  theano.function([self.subb_ind], [self.cost,self.error,self.error_top_5], updates=[], 
                                          givens=[(self.x, self.shared_x_slice),
                                                  (self.y, self.shared_y_slice)]
                                                                )                                                       
    def adjust_lr(self, epoch, val_error_list = None):
        
        # lr is calculated every time as a function of epoch and size
        
        if self.config['lr_policy'] == 'step':
            
            if epoch >=20 and epoch < 40:

                self.step_idx = 1
        
            elif epoch >=40 and epoch < 60:
                
                self.step_idx = 2

            elif epoch >=60 and epoch < 70:
                
                self.step_idx = 3
                
            else:
                pass
            
            tuned_base_lr = self.base_lr * 1.0/pow(10.0,self.step_idx) 
                
        if self.config['lr_policy'] == 'auto':
            if epoch>5 and (val_error_list[-3] - val_error_list[-1] <
                                self.config['lr_adapt_threshold']):
                tuned_base_lr = self.base_lr / 10.0
                
        self.shared_lr.set_value(np.float32(tuned_base_lr))
            
    def test(self):
        
        self.train(0)
        self.val(0)
        
        print 'test passed'
        
if __name__ == '__main__':
    
    
    import yaml
    with open('../../../run/config.yaml', 'r') as f:
        config = yaml.load(f)
        
    with open('../../../run/alexnet.yaml', 'r') as f:
        model_config = yaml.load(f)
    config = dict(config.items()+model_config.items())
    config['verbose'] = True
    config['batch_size']=1
    

    model = AlexNet(config)
    
    # inference demo
    model.compile_inference()
    
    test_image = np.zeros((3,227,227,1),dtype=theano.config.floatX) # inference on an image 
    
    y=np.ones(shape=(1,),dtype=np.int64)
    
    neq = model.inference(test_image,y)
    
    print neq, neq.dtype
    
    
     
     
