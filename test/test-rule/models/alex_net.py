# This version of alex_net.py is modified based on the theano_alexnet project. See the original project here:
# https://github.com/uoguelph-mlrg/theano_alexnet, and its copy right:
# Copyright (c) 2014, Weiguang Ding, Ruoyan Wang, Fei Mao and Graham Taylor
# All rights reserved.

import numpy as np
import sys
sys.path.append('../')

from lib.helper_funcs import get_rand3d
from lib.proc_load_mpi import crop_and_mirror

# model hyperparams
n_epochs = 70
momentum = 0.90
weight_decay = 0.0005
batch_size = 128
file_batch_size = 128
learning_rate = 0.01

lr_policy = 'step'
lr_step = [20, 40, 60]

use_momentum = True
use_nesterov_momentum = False

#cropping hyperparams
input_width = 227
input_height = 227

batch_crop_mirror = False
rand_crop = False

image_mean = 'img_mean'
dataname = 'imagenet'

# conv
lib_conv='cudnn'

class AlexNet(object):

    def __init__(self, config):

        self.verbose = config['verbose']
        
        import theano
        theano.config.on_unused_input = 'warn'
        self.name = 'AlexNet'
        
        # data
        from data.imagenet import ImageNet_data
        self.data = ImageNet_data(verbose=False)
        self.channels = self.data.channels # 'c' mean(R,G,B) = (103.939, 116.779, 123.68)
        self.input_width = input_width # '0' single scale training 224
        self.input_height = input_height # '1' single scale training 224
        self.batch_size = batch_size # 'b'
        self.file_batch_size = file_batch_size
        self.n_softmax_out = self.data.n_class
        
        # mini batching
        self.data.batch_data(file_batch_size)
        
        # training related
        self.n_epochs = n_epochs
        self.epoch = 0
        self.step_idx = 0
        self.mu = momentum # def: 0.9 # momentum
        self.use_momentum = use_momentum
        self.use_nesterov_momentum = use_nesterov_momentum
        self.eta = weight_decay #0.0002 # weight decay
        self.monitor_grad = config['monitor_grad']
        
        self.base_lr = np.float32(learning_rate)
        self.shared_lr = theano.shared(self.base_lr)
        self.shared_x = theano.shared(np.zeros((
                                                3,
                                                self.input_width, 
                                                self.input_height,
                                                file_batch_size
                                                ), 
                                                dtype=theano.config.floatX),  
                                                borrow=True)                           
        self.shared_y = theano.shared(np.zeros((file_batch_size,), 
                                          dtype=int),   borrow=True) 
        # slice batch if needed
        import theano.tensor as T                     
        subb_ind = T.iscalar('subb')  # sub batch index
        self.subb_ind = subb_ind
        self.shared_x_slice = self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
        self.shared_y_slice = self.shared_y[subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
        
        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        
        self.build_model()
        self.output = self.output_layer.output
        from layers2 import get_params, get_layers, count_params
        self.layers = get_layers(lastlayer = self.output_layer)
        self.params,self.weight_types = get_params(self.layers)
        count_params(self.params)
        self.grads = T.grad(self.cost,self.params)

        # To be compiled
        self.compiled_train_fn_list = []
        self.train_iter_fn = None
        self.val_iter_fn = None
        
        # iter related
        self.n_subb = file_batch_size/batch_size
        self.current_t = 0 # current filename pointer in the filename list
        self.last_one_t = False # if pointer is pointing to the last filename in the list
        self.subb_t = 0 # sub-batch index
        
        self.current_v=0
        self.last_one_v=False
        self.subb_v=0
        
        # preprocessing
        self.batch_crop_mirror = batch_crop_mirror
        self.input_width = input_width
        
    
    def build_model(self):
        
        if self.verbose: print self.name

        # start graph construction from scratch
        import theano.tensor as T
        from layers2 import ConvPoolLRN,Dropout,FC, \
                            Softmax,Flatten,LRN, Constant, Normal
        
        
        self.x = T.ftensor4('x')
        self.y = T.lvector('y')
        self.lr = T.scalar('lr')
                         

        convpool_layer1 = ConvPoolLRN(input=self.x,
                                        input_shape=(self.channels, 
                                                     self.input_width,
                                                     self.input_height,
                                                     self.batch_size),
                                                     
                                        filter_shape=(3, 11, 11, 96),
                                        convstride=4, padsize=0, group=1,
                                        poolsize=3, poolstride=2,
                                        b=0.0, lrn=True,
                                        lib_conv=lib_conv,
                                        printinfo = self.verbose
                                        #output_shape = (96, 27, 27, batch_size)
                                        )

        convpool_layer2 = ConvPoolLRN(input=convpool_layer1,
                                        #input_shape=(96, 27, 27, batch_size),
                                        filter_shape=(96, 5, 5, 256),
                                        convstride=1, padsize=2, group=2,
                                        poolsize=3, poolstride=2,
                                        b=0.1, lrn=True,
                                        lib_conv=lib_conv,
                                        printinfo = self.verbose
                                        #output_shape=(256, 13, 13, batch_size),
                                        )


        convpool_layer3 = ConvPoolLRN(input=convpool_layer2,
                                        #input_shape=(256, 13, 13, batch_size),
                                        filter_shape=(256, 3, 3, 384),
                                        convstride=1, padsize=1, group=1,
                                        poolsize=1, poolstride=0,
                                        b=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        printinfo = self.verbose
                                        #output_shape=(384, 13, 13, batch_size),
                                        )

        convpool_layer4 = ConvPoolLRN(input=convpool_layer3,
                                        #input_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 384),
                                        convstride=1, padsize=1, group=2,
                                        poolsize=1, poolstride=0,
                                        b=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        printinfo = self.verbose
                                        #output_shape=(384, 13, 13, batch_size),
                                        )

        convpool_layer5 = ConvPoolLRN(input=convpool_layer4,
                                        #input_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 256),
                                        convstride=1, padsize=1, group=2,
                                        poolsize=3, poolstride=2,
                                        b=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        printinfo = self.verbose
                                        #output_shape=(256, 6, 6, batch_size),
                                        )

        fc_layer6_input = Flatten(input=convpool_layer5.output.dimshuffle(3, 0, 1, 2),
                                  input_shape=(batch_size, 256, 6, 6),
                                  axis = 2,
                                  printinfo=self.verbose
                                  )
            
        fc_layer6      = FC(input=fc_layer6_input, 
                            # n_in=9216,
                            n_out=4096,
                            W=Normal((fc_layer6_input.output_shape[1], 4096), std=0.005),
                            b=Constant((4096,), val=0.1),
                            printinfo = self.verbose
                            )

        dropout_layer6 = Dropout(input=fc_layer6, 
                                  # n_in=4096,
                                  n_out=fc_layer6.output_shape[1], 
                                  prob_drop=0.5,
                                  printinfo = self.verbose)

        fc_layer7      = FC(input=dropout_layer6, 
                            # n_in=4096,
                            n_out=4096,
                            W = Normal((dropout_layer6.output_shape[1], 4096), std=0.005),
                            b = Constant((4096,), val=0.1),
                            printinfo = self.verbose
                            )

        dropout_layer7 = Dropout(input=fc_layer7, 
                                  #n_in=4096, 
                                  n_out=fc_layer7.output_shape[1],
                                  prob_drop=0.5,
                                  printinfo = self.verbose)

        softmax_layer8 = Softmax(input=dropout_layer7, 
                                      #n_in=4096, 
                                      n_out=self.n_softmax_out,
                                      W = Normal((dropout_layer7.output_shape[1], 
                                                  self.n_softmax_out), mean=0, std=0.01),
                                      b = Constant((self.n_softmax_out,),val=0),
                                      printinfo = self.verbose)
                                      
        self.output_layer = softmax_layer8
        
        self.cost = softmax_layer8.negative_log_likelihood(self.y)     
        self.error = softmax_layer8.errors(self.y)
        self.error_top_5 = softmax_layer8.errors_top_x(self.y)
        
    
    
    def compile_train(self, *args):
        
        # args is a list of dictionaries
        
        if self.verbose: print 'compiling training function...'
        
        import theano
        
        for arg_list in args:
            self.compiled_train_fn_list.append(theano.function(**arg_list))
        
        if self.monitor_grad:
            
            norms = [grad.norm(L=2) for grad in self.grads]
            
            self.get_norm = theano.function([self.subb_ind], norms,
                                              givens=[(self.x, self.shared_x_slice), 
                                                      (self.y, self.shared_y_slice)]
                                                                          )
    def compile_inference(self):

        if self.verbose: print 'compiling inference function...'
        
        import theano
        
        self.inf_fn = theano.function([self.x],self.output)
        
    def compile_val(self):

        if self.verbose: print 'compiling validation function...'
        
        import theano
        
        self.val_fn =  theano.function([self.subb_ind], [self.cost,self.error,self.error_top_5], updates=[], 
                                          givens=[(self.x, self.shared_x_slice),
                                                  (self.y, self.shared_y_slice)]
                                                                )
    
    def compile_iter_fns(self):
        
        from lib.opt import pre_model_iter_fn

        pre_model_iter_fn(self, sync_type='avg')

        
    def adjust_hyperp(self, epoch):
            
        '''
        borrowed from AlexNet
        '''
        # lr is calculated every time as a function of epoch and size
        
        if lr_policy == 'step':
            
            stp0,stp1,stp2 = lr_step
            
            if epoch >=stp0 and epoch < stp1:

                self.step_idx = 1
        
            elif epoch >=stp1 and epoch < stp2:
                
                self.step_idx = 2

            elif epoch >=stp2 and epoch < n_epochs:
                
                self.step_idx = 3
                
            else:
                pass
            
            tuned_base_lr = self.base_lr * 1.0/pow(10.0,self.step_idx) 
            
        else:
            raise NotImplementedError()
        
        self.shared_lr.set_value(np.float32(tuned_base_lr))
        
    def cleanup(self):
        
        pass
        
if __name__ == '__main__':
    
    
    # setting up device
    import os
    if 'THEANO_FLAGS' in os.environ:
        raise ValueError('Use theanorc to set the theano config')
    os.environ['THEANO_FLAGS'] = 'device={0}'.format('cuda0')
    import theano.gpuarray
    # This is a bit of black magic that may stop working in future
    # theano releases
    ctx = theano.gpuarray.type.get_context(None)
    
    
    import yaml
    with open('../config.yaml', 'r') as f:
        config = yaml.load(f)
    
    model = AlexNet(config)
    
    model.compile_iter_fns()
    
    # get recorder
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    from lib.recorder import Recorder
    recorder = Recorder(comm, printFreq=4, modelname='cifar10', verbose=True)
    
    
    
    
    
    
    
    
    
    
    # inference demo
    model.compile_inference()
    
    test_image = np.zeros((3,227,227,1),dtype=theano.config.floatX) # inference on an image 
    
    soft_prob = model.inference(test_image)
    
    num_top = 5
    
    y_pred_top_x = np.argsort(soft_prob, axis=1)[:, -num_top:] # prob sorted from small to large
    
    print ''
    print 'top-5 prob:'
    print y_pred_top_x[0]
    
    
    print ''
    print 'top-5 prob catagories:'
    print [soft_prob[0][index] for index in y_pred_top_x[0]]
    
    print ''
    # git clone https://github.com/hma02/show_batch.git
    # run mk_label_dict.py to generate label_dict.npy
    label_dict = np.load('label_dict.npy').tolist()
    
    print 'discription:'
    for cat in y_pred_top_x[0]:
        print "%s: %s" % (cat,label_dict[cat])
        print ''
    
    
    
     
     
