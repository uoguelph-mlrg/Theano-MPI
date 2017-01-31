import sys

import numpy as np

import numpy as np

import hickle as hkl

# model hyperparams
n_epochs = 90
momentum = 0.90
weight_decay = 0.0005
batch_size = 32
file_batch_size = 128
learning_rate = 0.005

lr_policy = 'poly'

use_momentum = True
use_nesterov_momentum = False

#cropping hyperparams
input_width = 224
input_height = 224

# apparently, training converges better with batch_crop_mirror=False. 
# 1200 6.898191 vs 1320 6.892865
batch_crop_mirror = True 
rand_crop = True

image_mean = 'img_mean'
dataname = 'imagenet'

# conv
lib_conv='cudnn' # cudnn or corrmm

monitor_grad = False

seed_weight_on_pid = False

from theanompi.models.layers2 import (Normal, Constant, Layer, Conv, Pool, LRN, 
                                      ConvPoolLRN, Dimshuffle, Flatten,
                                      Dropout, FC, Softmax, get_params, get_layers, count_params)
# class ConvPool_LRN(object):
#
#     def __init__(self, input, image_shape, filter_shape, convstride, padsize,
#                  poolsize, poolstride,poolpad, W, b, lrn=False,
#                  lib_conv='cudnn',
#                  ):
#         self.filter_size = filter_shape
#         self.convstride = convstride
#         self.padsize = padsize
#
#
#         self.channel = image_shape[0]
#         self.lrn = lrn
#         self.lib_conv = lib_conv
#
#         self.filter_shape = np.asarray(filter_shape)
#         self.image_shape = np.asarray(image_shape)
#
#
#         self.W = W#Weight(self.filter_shape)
#         self.b = b#Weight(self.filter_shape[3])#, bias_init, std=0)
#
#         input_shuffled = input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
#             # in01out to outin01
#             # print image_shape_shuffled
#             # print filter_shape_shuffled
#
#         W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
#         conv_out = dnn.dnn_conv(img=input_shuffled,
#                                 kerns=W_shuffled,
#                                 subsample=(convstride, convstride),
#                                 border_mode=padsize,
#                                 )
#         conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')
#
#         # ReLu
#         self.output = T.maximum(conv_out, 0)
#
#         # Pool
#         self.poolsize = poolsize
#         self.poolstride = poolstride
#         self.poolpad = poolpad
#
#         if self.poolsize != 1:
#             self.output = dnn.dnn_pool(self.output,
#                                        ws=(poolsize, poolsize),
#                                        stride=(poolstride, poolstride),
#                                        mode='max', pad=(poolpad, poolpad))
#
#         self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
#
#         # LRN
#         if self.lrn:
#             self.lrn_func = CrossChannelNormalization()
#             # lrn_input = gpu_contiguous(self.output)
#             self.output = self.lrn_func(self.output)
#
#         self.params = [self.W.val, self.b.val]
#         self.weight_type = ['W', 'b']
#         print "conv ({}) layer with shape_in: {}".format(lib_conv,
#                                                          str(image_shape))
                                                                 
            
class Incept(Layer):
    
    ''' Inception module.
    
    Parameters:
        l_in: Theano input graph tensor
    
    See:
        https://gist.github.com/benanne/ae2a7adaab133c61a059
    
    '''
    
    def __init__(self, input, input_shape = None, output_shape = None, 
                  n1x1=64, nr3x3=96, n3x3=128, nr5x5=16, n5x5=32, npj=32, 
                  lib_conv='cudnn', printinfo=False):  
                  
        
        self.get_input_shape(input,input_shape) 
        self.verbose = printinfo
        
        layers=[]
        outlayers=[]
    
        if n1x1 > 0:   

            l_1x1 =        Conv(input=input,# (128, 192,28,28), 
                                convstride=1, padsize=0,
                                W = Normal((n1x1, self.input_shape[1], 1, 1), mean = 0.0, std=0.03 ),
                                b = Constant((n1x1,), val = 0.2),
                                lib_conv=lib_conv,
                                printinfo=self.verbose
                                )   
    
    
            layers.append(l_1x1)
            outlayers.append(l_1x1)
            
        if n3x3 > 0:
            
            if nr3x3 > 0:


                l_r3x3 = Conv(input=input,
                                convstride=1, padsize=0,
                                W = Normal((nr3x3, self.input_shape[1], 1, 1),mean = 0.0,std=0.09),
                                b = Constant((nr3x3,), val = 0.2),
                                lib_conv=lib_conv,
                                printinfo=self.verbose
                                )   
        
        
                layers.append(l_r3x3)                                               
                
            else:
                l_r3x3 = l_in


            l_3x3 =        Conv(input=l_r3x3,
                                convstride=1, padsize=1,
                                W = Normal((n3x3, nr3x3, 3, 3), mean = 0.0, std=0.03 ),
                                b = Constant((n3x3,), val = 0.2),
                                lib_conv=lib_conv,
                                printinfo=self.verbose
                                )
    
    
            layers.append(l_3x3)
            outlayers.append(l_3x3)
        
    
        if n5x5 > 0:
        
            if nr5x5 > 0:
 

                l_r5x5 = Conv(input=input,
                                convstride=1, padsize=0,
                                W = Normal((nr5x5, self.input_shape[1], 1, 1), mean = 0.0, std=0.2 ),
                                b = Constant((nr5x5,), val = 0.2),
                                lib_conv=lib_conv,
                                printinfo=self.verbose                                
                                )   
        
        
                layers.append(l_r5x5)
                
            else:
                l_r5x5 = l_in


            l_5x5 =        Conv(input=l_r5x5,
                                convstride=1, padsize=2,
                                W = Normal((n5x5, nr5x5, 5, 5), mean = 0.0, std=0.03 ),
                                b = Constant((n5x5,), val = 0.2 ),
                                lib_conv=lib_conv,
                                printinfo=self.verbose  
                                )
                        
    
            layers.append(l_5x5)
            outlayers.append(l_5x5)
    
        if npj > 0:
                                            
            l_pool     =   Pool(input=input, 
                                poolsize=3, 
                                poolstride=1, 
                                poolpad=1,
                                mode = 'max',
                                printinfo=self.verbose
                                )                              

            l_pool_project=Conv(input=l_pool,
                                convstride=1, padsize=0,
                                W = Normal((npj, self.input_shape[1], 1, 1), mean = 0.0, std=0.1 ),
                                b = Constant((npj,), val = 0.2 ),
                                lib_conv=lib_conv,
                                printinfo=self.verbose  
                                )   
    
    
            layers.append(l_pool_project)
            outlayers.append(l_pool_project)          
        
        import theano.tensor as T
        self.output = T.concatenate([layer.output for layer in outlayers], axis=1)  # bc01 concaatenate on 'c'
        
        self.params, self.weight_type = get_params(layers)
            
        if output_shape:
            self.output_shape = output_shape 
        else:
            self.output_shape = self.get_output_shape(self.input_shape)
        
        self.name = 'Inception ({})'.format(lib_conv)
        if printinfo: self.print_shape()
        
class Aux_tower(Layer):
    '''    Auxilary classifier tower
    
    Parameters:
        input: tensor
        Theano input graph tensor
        
        input_shape: tuple
    
    '''
    def __init__(self, input, n_softmax_out, input_shape=None, output_shape= None,
                 lib_conv='cudnn', printinfo=False):
        
        
        self.get_input_shape(input,input_shape) 
        self.verbose = printinfo
        
        layers=[]
        outlayers=[]
        
        # input shape = (14x14x512or528)
        pool =           Pool(input=input, 
                              poolsize=5, 
                              poolstride=2, 
                              poolpad=0,
                              mode = 'average',
                              printinfo=self.verbose
                              )

        # output shape = (4x4x512or528)
        
        conv1x1        = Conv(input=pool,
                              convstride=1, padsize=0,
                              W = Normal((128, self.input_shape[1], 1, 1),mean=0.0, std=0.1),
                              b = Constant((128,),val = 0.2),
                              lib_conv='cudnn',
                              printinfo=self.verbose 
                              )
        layers.append(conv1x1)

        # output shape = (4x4x128)
        
                             
        l_flatten =   Flatten(input = conv1x1, #5
                              #input_shape=conv_5x5.output_shape, # (b, 64, 2, 2)
                              axis = 2, # expand dimensions after the first dimension
                              printinfo=self.verbose
                              #output_shape = (b,64*2*2)
                              )

        # output shape = (2048)
        
        fc            =     FC(input= l_flatten, 
                               #n_in=2048,
                               n_out=1024,
                               W = Normal((l_flatten.output_shape[1],1024),mean=0,std=0.01),
                               b = Constant((1024,),val=0),
                               printinfo=self.verbose
                               #input_shape = flatten.output_shape # (b, 9216)
                               )
        
        layers.append(fc)      
        
        drp =          Dropout(input=fc,
                              #n_in=1024,
                               n_out=fc.output_shape[1], 
                               prob_drop=0.7,
                               printinfo=self.verbose
                               )
        
        softmax_layer= Softmax(input=drp,  
                               n_out=n_softmax_out,
                               W = Normal((drp.output_shape[1], n_softmax_out), mean=0, std=0.01),
                               b = Constant((n_softmax_out,),val=0),
                               printinfo=self.verbose
                               )
        
        layers.append(softmax_layer)
        
        self.output = softmax_layer.p_y_given_x
        self.negative_log_likelihood = softmax_layer.negative_log_likelihood
        
        self.params, self.weight_type = get_params(layers)
            
        if output_shape:
            self.output_shape = output_shape 
        else:
            self.output_shape = self.get_output_shape(self.input_shape)
        
        self.name = 'AuxTower ({})'.format(lib_conv)
        if printinfo: self.print_shape()
         

class GoogLeNet(object):

    """    GoogleNet classifier for ILSVRC.
    
    Parameters:
    
        config:  dict
        training related and model related hyperparameter dict 
    
    References:
    
        [1] C Szegedy, W Liu, Y Jia, P Sermanet, S Reed, 
            D Anguelov, D Erhan, V Vanhoucke, A Rabinovich (2014):
            Going deeper with convolutions.
            The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1-9
            
        [2] https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
    """
    
    def __init__(self,config):

        self.verbose = config['verbose'] 
        self.rank = config['rank'] # will be used in sharding and distinguish rng
        self.size = config['size']
        self.no_paraload=False
        try: 
            self.no_paraload = config['no_paraload']
        except:
            pass
            
        import theano
        theano.config.on_unused_input = 'warn'
        
        self.name = 'GoogLeNet'
        
        # data
        from theanompi.models.data import ImageNet_data
        self.data = ImageNet_data(verbose=False)
        self.channels = self.data.channels # 'c' mean(R,G,B) = (103.939, 116.779, 123.68)
        self.input_width = input_width # '0' single scale training 224
        self.input_height = input_height # '1' single scale training 224
        # if self.size>1: # only use avg
#             self.batch_size = batch_size/self.size
#         else: # TODO find out if this works better
        self.batch_size = batch_size # 'b
        self.file_batch_size = file_batch_size
        self.n_softmax_out = self.data.n_class
        
        # mini batching
        self.data.batch_data(file_batch_size)
        #self.data.shuffle_data()
        
# training related
        self.n_epochs = n_epochs
        self.epoch = 0
        self.step_idx = 0
        self.mu = momentum # def: 0.9 # momentum
        self.use_momentum = use_momentum
        self.use_nesterov_momentum = use_nesterov_momentum
        self.eta = weight_decay #0.0002 # weight decay
        self.monitor_grad = monitor_grad
        
        self.base_lr = np.float32(learning_rate)
        self.shared_lr = theano.shared(self.base_lr)
        self.shared_x = theano.shared(np.zeros((
                                                3,
                                                self.input_width,#self.data.width, 
                                                self.input_height,#self.data.height,
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
        from theanompi.models.layers2 import get_params, get_layers, count_params
        #self.layers = get_layers(lastlayer = self.output_layer)
        self.params,self.weight_types = get_params(self.layers)
        count_params(self.params, verbose=self.verbose)
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
        
        if self.data.para_load and not self.no_paraload:
            
            self.data.spawn_load()
            self.data.para_load_init(self.shared_x)
            
        
    def build_model(self):
        
        if self.verbose: print self.name
        
        import theano.tensor as T
        if seed_weight_on_pid:
            import theanompi.models.layers2 as layers
            import os
            layers.rng = np.random.RandomState(os.getpid())
        
        self.x = T.ftensor4('x') # c01b
        self.y = T.lvector('y')
        self.lr = T.scalar('lr')    
        

        conv_7x7 = ConvPoolLRN(input=self.x,
                                input_shape=(self.channels,
                                             self.input_width,
                                             self.input_height,
                                             self.batch_size),
                                convstride=2, padsize=4,
                                poolsize=3, poolstride=2, poolpad=1,
                                W = Normal((3, 7, 7, 64), mean = 0.0, std=0.1),
                                b = 0.2, 
                                lrn=True,
                                group=1,
                                lib_conv='cudnn',
                                printinfo=self.verbose
                                )                 
        # output shape = (112x112x64)
        # output shape = (56x56x64)
        
        shuffle =    Dimshuffle(input=conv_7x7,
                                new_axis_order=(3,0,1,2), # c01b -> bc01
                                printinfo=self.verbose
                                )
                                     
        conv_r3x3  =        Conv(input=shuffle,
                                #image_shape=(batch_size, 64,56,56),
                                convstride=1, padsize=0,
                                W = Normal((64, 64, 1, 1),mean = 0.0,std=0.1),
                                b = Constant((64,), val = 0.2),
                                lib_conv='cudnn',
                                printinfo=self.verbose
                                )                                           
 
        # output shape = (56x56x64)
        
        shuffle =    Dimshuffle(input=conv_r3x3,
                                new_axis_order=(1,2,3,0), # bc01 -> c01b
                                printinfo=self.verbose
                                )
                                           
        conv_3x3 = ConvPoolLRN(input=shuffle,
                                #image_shape=(64,56,56, batch_size),
                                convstride=1, padsize=2,
                                poolsize=3, poolstride=2, poolpad=1,
                                W = Normal((64, 3, 3, 192),mean = 0.0,std=0.03),
                                b = 0.2, 
                                lrn=True,
                                group=1,
                                lib_conv='cudnn',
                                printinfo=self.verbose
                                )                                           

        # output shape = (56x56x192) 
        # output shape = (28x28x192)
        
        shuffle =    Dimshuffle(input=conv_3x3,
                                new_axis_order=(3,0,1,2), # c01b -> bc01
                                printinfo=self.verbose
                                )
                     

        incep3a =         Incept(input=shuffle,
                                #input_shape = (batch_size, 192,28,28)
                                n1x1=64, nr3x3=96, n3x3=128, 
                                nr5x5=16, n5x5=32, npj=32,
                                lib_conv=lib_conv,
                                printinfo = self.verbose
                                )
     
        # print 'incep3a output shape: (28x28x256)'
        # output shape = (28x28x256)
        
        incep3b =        Incept(input=incep3a,
                                #input_shape = (256,28,28,batch_size),
                                n1x1=128, nr3x3=128, n3x3=192, 
                                nr5x5=32, n5x5=96, npj=64,
                                lib_conv=lib_conv,
                                printinfo = self.verbose
                                )
               
        # print 'incep3b output shape: (28x28x480)'
        # output shape = (28x28x480)        

#        lrn3 = self.lrn_func(incep3b.output)
#        print 'LRN(added)'
        
        pool3 =            Pool(input=incep3b,
                                poolsize=3, 
                                poolstride=2, 
                                poolpad=1, 
                                printinfo=self.verbose
                                )        
        # output shape = (14x14x480)
        
        incep4a =       Incept(input=pool3, 
                               #input_shape = (480,14,14,batch_size), 
                               n1x1=192, nr3x3=96, n3x3=208, 
                               nr5x5=16, n5x5=48, npj=64,
                               lib_conv=lib_conv,
                               printinfo = self.verbose
                               )
              
        # print 'incep4a output shape: (14x14x512)'
        # output shape = (14x14x512)
        
        incep4b =       Incept(input=incep4a, 
                                    #input_shape = (512,14,14,batch_size), 
                               n1x1=160, nr3x3=112, n3x3=224, 
                               nr5x5=24, n5x5=64, npj=64,
                               lib_conv=lib_conv,
                               printinfo = self.verbose
                               )

        # print 'incep4b output shape: (14x14x512)'
        # output shape = (14x14x512)          
        
        incep4c =       Incept(input=incep4b, 
                                 #input_shape = (512,14,14,batch_size), 
                               n1x1=128, nr3x3=128, n3x3=256, 
                               nr5x5=24, n5x5=64, npj=64,
                               lib_conv=lib_conv,
                               printinfo = self.verbose
                               )
                              

        # print 'incep4c output shape: (14x14x512)'
        # output shape = (14x14x512) 

        incep4d =       Incept(input=incep4c, 
                                 #input_shape = (512,14,14,batch_size), 
                               n1x1=112, nr3x3=144, n3x3=288, 
                               nr5x5=32, n5x5=64, npj=64,
                               lib_conv=lib_conv,
                               printinfo = self.verbose
                               )
        

        # print 'incep4d output shape: (14x14x528)'
        # output shape = (14x14x528) 
         
        
        incep4e =       Incept(input=incep4d, 
                                #input_shape = (528,14,14,batch_size), 
                               n1x1=256, nr3x3=160, n3x3=320, 
                               nr5x5=32, n5x5=128, npj=128,
                               lib_conv=lib_conv,
                               printinfo = self.verbose
                               )
        
        # print 'incep4e output shape: (14x14x832)'
        # output shape = (14x14x832)                
        
        lrn4 =             LRN(input=incep4e,
                               printinfo=self.verbose) # turn on only this for 16data,53s/5120images
        # print 'LRN(added)'
        
        pool4 =           Pool(input=lrn4, #incep4e,
                               poolsize=3, 
                               poolstride=2, 
                               poolpad=1, 
                               printinfo = self.verbose)        
        # output shape = (7x7x832)        
        
        incep5a =       Incept(input=pool4, 
                            #input_shape = (832,7,7,batch_size), 
                               n1x1=256, nr3x3=160, n3x3=320, 
                               nr5x5=32, n5x5=128, npj=128,
                               lib_conv=lib_conv,
                               printinfo = self.verbose
                               )
        
        # print 'incep5a output shape: (7x7x832)'
        # output shape = (7x7x832)   
        
        incep5b =       Incept(input=incep5a, 
                           #input_shape = (832,7,7,batch_size), 
                               n1x1=384, nr3x3=192, n3x3=384, 
                               nr5x5=48, n5x5=128, npj=128,
                               lib_conv=lib_conv,
                               printinfo = self.verbose
                               )
                               
        # print 'incep5b output shape: (7x7x1024)'
        # output shape = (7x7x1024)
        
#        lrn5 = self.lrn_func(incep5b.output) # turn on only this for 16data, 51s/5120images
#        print 'LRN(added)'
        
        poolx =           Pool(input=incep5b,
                               poolsize=7, 
                               poolstride=1, 
                               poolpad=0, 
                               mode = 'average' ,
                               printinfo = self.verbose)
        # output shape = (1x1x1024)
        
        

        shuffle  =  Dimshuffle(input=poolx,
                               new_axis_order=(3, 0, 1, 2),
                               printinfo=self.verbose
                               )
                               
        l_flatten =    Flatten(input = shuffle,
                               axis = 2,
                               printinfo=self.verbose
                               )
        # output shape = (1024)                              
    
        dropout=       Dropout(input=l_flatten,
                               #n_in=1024, 
                               n_out=l_flatten.output_shape[1], 
                               prob_drop=0.4,
                               printinfo=self.verbose
                               )
        # output shape = (1024)
               
        
        softmax_layer = Softmax(input=dropout,
                                #n_in=1024, 
                                n_out=self.n_softmax_out,
                                printinfo=self.verbose
                                )
        # output shape = (n_softmax_out)              
        
        # auxilary classifier
        # print 'auxilary classifier 1:'
        aux1 =         Aux_tower(input=incep4a,
                                #input_shape=(512,14,14,batch_size),
                                 n_softmax_out=self.n_softmax_out,
                                 lib_conv=lib_conv,
                                 printinfo=self.verbose
                                 )
        
        # print 'auxilary classifier 2:'
        aux2 =         Aux_tower(input=incep4d,
                                 #input_shape=(528,14,14,batch_size),
                                 n_softmax_out=self.n_softmax_out,
                                 lib_conv=lib_conv,
                                 printinfo=self.verbose
                                 )
       
        self.output_layer = softmax_layer
        
        self.cost =    softmax_layer.negative_log_likelihood(self.y)+ \
                            0.3*aux1.negative_log_likelihood(self.y)+\
                            0.3*aux2.negative_log_likelihood(self.y)        
        self.error = softmax_layer.errors(self.y)
        self.error_top_5 = softmax_layer.errors_top_x(self.y)
        
        
        self.layers = get_layers(lastlayer = self.output_layer)
        self.layers.extend([aux1,aux2])
        
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
        
        import time
        
        start = time.time()
        
        from theanompi.lib.opt import pre_model_iter_fn

        pre_model_iter_fn(self, sync_type='avg')
        
        if self.verbose: print 'Compile time: %.3f s' % (time.time()-start)
            
    def reset_iter(self, mode):
        
        if mode=='train':
            
            self.current_t = 0
            self.subb_t=0
            self.last_one_t = False
        else:
            
            self.current_v = 0
            self.subb_v=0
            self.last_one_v = False
            
        
    def train_iter(self,count,recorder):
        
        '''use the train_iter_fn compiled'''
            
        if self.current_t==0: 
            self.data.shuffled=False
            self.data.shuffle_data()
        
        img= self.data.train_img
        labels = self.data.train_labels
            
        img_mean = self.data.rawdata[4]
        mode='train'
        function=self.train_iter_fn
         
        if self.subb_t == 0: # load the whole file into shared_x when loading sub-batch 0 of each file.
        
            recorder.start()
        
            arr = img[self.current_t] #- img_mean
            
            arr = np.rollaxis(arr,0,4)
                                    
            self.shared_x.set_value(arr)
            self.shared_y.set_value(labels[self.current_t])
            
            
            if self.current_t == self.data.n_batch_train - 1:
                self.last_one_t = True
            else:
                self.last_one_t = False
                
        
            recorder.end('wait')
                
        recorder.start()
        
        cost,error= function(self.subb_t)
        
        if self.verbose: 
            #print count+self.config['rank'], cost, error
            #if count+self.config['rank']>45: exit(0)
            if self.monitor_grad: 
                print np.array(self.get_norm(self.subb_t))
                #print [np.int(np.log10(i)) for i in np.array(self.get_norm(self.subb))]
            
        recorder.train_error(count, cost, error)
        recorder.end('calc')


            
        if (self.subb_t+1)//self.n_subb == 1: # test if next sub-batch is in another file
            
            if self.last_one_t == False:
                self.current_t+=1
            else:
                self.current_t=0
            
            self.subb_t=0
        else:
            self.subb_t+=1
        
    def val_iter(self, count, recorder):
        
        '''use the val_iter_fn compiled'''
            
        if self.current_v==0: self.data.shard_data(file_batch_size, self.rank, self.size)
            
        img= self.data.val_img_shard
        labels = self.data.val_labels_shard
            
        img_mean = self.data.rawdata[4]
        mode='val'
        function=self.val_iter_fn
        
        if self.subb_v == 0: # load the whole file into shared_x when loading sub-batch 0 of each file.
        
            arr = img[self.current_v] #- img_mean
        
            arr = np.rollaxis(arr,0,4)
                                
            self.shared_x.set_value(arr)
            self.shared_y.set_value(labels[self.current_v])
        
        
            if self.current_v == self.data.n_batch_val - 1:
                self.last_one_v = True
            else:
                self.last_one_v = False
        
        from theanompi.models.layers2 import Dropout, Crop       
        Dropout.SetDropoutOff()
        Crop.SetRandCropOff()
        cost,error,error_top5 = function(self.subb_v)
        Dropout.SetDropoutOn()
        Crop.SetRandCropOn()
        
        recorder.val_error(count, cost, error, error_top5)
        
        if (self.subb_v+1)//self.n_subb == 1: # test if next sub-batch is in another file
        
            if self.last_one_v == False:
                self.current_v+=1
            else:
                self.current_v=0
        
            self.subb_v=0
        else:
            self.subb_v+=1
                                                               
    def adjust_hyperp(self, epoch):
            
        # Poly lr policy according to
        # https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
        # the effective learning rate follows a polynomial decay, to be
        # zero by the max_iter. 
        # return base_lr * (1 - iter/max_iter) ^ (power)
        # power = 0.5
        # max_iter = 2400000
        
        # batch_len = len(train_batches) = 10008
        # since epoch* batch_len = iter
        # max_iter = 240 * batch_len
        # iter/max_iter = epoch/240

        # Poly lr policy according to
        # https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
        
        tuned_base_lr = self.base_lr
        
        for i in range(5, epoch+1):
            tuned_base_lr = tuned_base_lr * \
                pow( (1. -  1.* (i) /240.0), 0.5 )
                
        self.shared_lr.set_value(np.float32(tuned_base_lr))
        
    def cleanup(self):
        
        if self.data.para_load:
            
            self.data.para_load_close()
        
if __name__ == '__main__':
    
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    
    # setting up device
    import os
    if 'THEANO_FLAGS' in os.environ:
        raise ValueError('Use theanorc to set the theano config')
    os.environ['THEANO_FLAGS'] = 'device={0}'.format('cuda0')
    import theano.gpuarray
    # This is a bit of black magic that may stop working in future
    # theano releases
    ctx = theano.gpuarray.type.get_context(None)
    
    
    config={}
    config['verbose'] = True
    # config['device'] = 'cuda0'
    config['rank'] = comm.rank
    config['size'] = comm.size
    
    model = AlexNet(config)
    
    model.compile_iter_fns()
    
    # get recorder
    
    from theanompi.lib.recorder import Recorder
    recorder = Recorder(comm, printFreq=5120/model.batch_size, modelname='alexnet', verbose=True)
    
    
    # train
    
    for batch_i in range(model.data.n_batch_train):
        
        model.train_iter(batch_i, recorder)
        
        recorder.print_train_info(batch_i)
        
    # val
    for batch_j in range(model.data.n_batch_val):
        
        model.val_iter(batch_i, recorder)

    #recorder.gather_val_info()
    recorder.print_val_info(batch_i)
    
    model.epoch+=1
    print 'finish one epoch'
    
    model.adjust_hyperp(epoch=40)
    
    model.cleanup()
    
    exit(0)
    
    
    # inference demo
    
    batch_size = 1
    batch_crop_mirror = True
    
    # or
    # batch_size=batch_size
    # batch_crop_mirror=False
    
    trained_params = [param.get_value() for param in model.params]
    
    model = AlexNet(config)
    
    for p, p_old in zip(model.params, trained_params):
        p.set_value(p_old)
    
    model.compile_inference()
    
    
    
    test_image = np.zeros((3,227,227,1),dtype=theano.config.floatX) # inference on an image 
    
    soft_prob = model.inf_fn(test_image)
    
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
    
