# This version of alex_net.py is modified based on the theano_alexnet project. See the original project here:
# https://github.com/uoguelph-mlrg/theano_alexnet, and its copy right:
# Copyright (c) 2014, Weiguang Ding, Ruoyan Wang, Fei Mao and Graham Taylor
# All rights reserved.

import numpy as np

import hickle as hkl

# model hyperparams
n_epochs = 70
momentum = 0.90
weight_decay = 0.0005
batch_size = 128
file_batch_size = 128
learning_rate = 0.01
# size=1 converge at 1320 6.897861; if avg and not scale lr by size, then size=2 will converge at: 2360 6.898975
lr_policy = 'step'
lr_step = [20, 40, 60]

use_momentum = True
use_nesterov_momentum = False

#cropping hyperparams
input_width = 227
input_height = 227

# apparently, training converges better with batch_crop_mirror=False. 
# 1200 6.898191 vs 1320 6.892865
batch_crop_mirror = False 
rand_crop = True

image_mean = 'img_mean'
dataname = 'imagenet'

# conv
lib_conv='cudnn' # cudnn or corrmm

monitor_grad = False

seed_weight_on_pid = False

class AlexNet(object):

    def __init__(self, config):

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
        self.name = 'AlexNet'
        
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
        
        # mini batching and other data parallel common routine
        self.data.batch_data(file_batch_size)
        self.data.extend_data(rank=self.rank, size=self.size)
        self.data.shuffle_data(mode='train', common_seed=1234)
        self.data.shuffle_data(mode='val')
        self.data.shard_data(mode='train', rank=self.rank, size=self.size) # to update data.n_batch_train
        self.data.shard_data(mode='val', rank=self.rank, size=self.size) # to update data.n_batch_val
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
        self.layers = get_layers(lastlayer = self.output_layer)
        self.params,self.weight_types = get_params(self.layers)
        count_params(self.params, verbose=self.verbose)
        self.grads = T.grad(self.cost,self.params)

        # To be compiled
        self.compiled_train_fn_list = []
        self.train_iter_fn = None
        self.val_iter_fn = None
        
        # iter related
        self.n_subb = file_batch_size//batch_size
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
            self.data.para_load_init(self.shared_x, input_width, input_height, 
                                    rand_crop, batch_crop_mirror)
        
    
    def build_model(self):
        
        if self.verbose: print(self.name)

        # start graph construction from scratch
        import theano.tensor as T
        if seed_weight_on_pid:
            import theanompi.models.layers2 as layers
            import os
            layers.rng = np.random.RandomState(os.getpid())
        from theanompi.models.layers2 import (ConvPoolLRN,Dropout,FC, 
                                                Dimshuffle, Crop, Subtract,
                                                Softmax,Flatten,LRN, Constant, Normal)
        
        
        self.x = T.ftensor4('x')
        self.y = T.lvector('y')
        self.lr = T.scalar('lr')
        
        # subtract_layer = Subtract(input=self.x,
        #                           input_shape=(self.channels,
        #                                        self.data.width,
        #                                        self.data.height,
        #                                        self.batch_size),
        #                           subtract_arr = self.data.rawdata[4],
        #                           printinfo = self.verbose
        #                           )
        #
        # crop_layer = Crop(input=subtract_layer,
        #                   output_shape=(self.channels,
        #                                 self.input_width,
        #                                 self.input_height,
        #                                 self.batch_size),
        #                   flag_batch=batch_crop_mirror,
        #                   printinfo = self.verbose
        #                   )
                         
        convpool_layer1 = ConvPoolLRN(input=self.x,  #crop_layer,
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
        shuffle = Dimshuffle(input=convpool_layer5,
                             new_axis_order=(3,0,1,2),
                             printinfo=self.verbose
                             )
        
        fc_layer6_input = Flatten(input=shuffle,
                                  #input_shape=(batch_size, 256, 6, 6),
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
        
        if self.verbose: print('compiling training function...')
        
        import theano
        
        for arg_list in args:
            self.compiled_train_fn_list.append(theano.function(**arg_list))
        
        if self.monitor_grad:
            
            norms = [grad.norm(L=2) for grad in self.grads]
            import theano.tensor as T
            norms = T.log10(norms)
            
            self.get_norm = theano.function([self.subb_ind], [T.sum(norms), T.max(norms)],
                                              givens=[(self.x, self.shared_x_slice), 
                                                      (self.y, self.shared_y_slice)]
                                                                          )
    def compile_inference(self):

        if self.verbose: print('compiling inference function...')
        
        import theano
        
        self.inf_fn = theano.function([self.x],self.output)
        
    def compile_val(self):

        if self.verbose: print('compiling validation function...')
        
        import theano
        
        self.val_fn =  theano.function([self.subb_ind], [self.cost,self.error,self.error_top_5], updates=[], 
                                          givens=[(self.x, self.shared_x_slice),
                                                  (self.y, self.shared_y_slice)]
                                                                )
    
    def compile_iter_fns(self):
        
        import time
        
        start = time.time()
        
        from theanompi.lib.opt import pre_model_iter_fn

        pre_model_iter_fn(self,self.size)
        
        if self.verbose: print('Compile time: %.3f s' % (time.time()-start))
    
    def reset_iter(self, mode):
        
        '''used at the begininig of another mode'''
        
        if mode=='train':

            self.current_t = 0
            self.subb_t=0
            self.last_one_t = False
        else:

            self.current_v = 0
            self.subb_v=0
            self.last_one_v = False
        
        if self.data.para_load:
            
            self.data.icomm.isend(mode,dest=0,tag=40)
        
    def train_iter(self, count,recorder):
        
        '''use the train_iter_fn compiled'''
        '''use parallel loading for large or remote data'''

            
        if self.current_t==0 and self.subb_t == 0:  
            self.data.shuffle_data(mode='train',common_seed=self.epoch)
            self.data.shard_data(mode='train',rank=self.rank, size=self.size)
        
        img= self.data.train_img_shard
        labels = self.data.train_labels_shard

        mode = 'train'
        function = self.train_iter_fn
        
        # print len(img), 'current_t: %d, subb_t: %d' % (self.current_t,self.subb_t)
            
        if self.subb_t == 0: # load the whole file into shared_x when loading sub-batch 0 of each file.
        
            recorder.start()
            
            # parallel loading of shared_x
            if self.data.para_load:
                
                icomm = self.data.icomm
                
                if self.current_t == 0:
                    
                    # 3.0 give mode signal to adjust loading mode between train and val
                    icomm.isend('train',dest=0,tag=40)
                    # 3.1 give load signal to load the very first file
                    icomm.isend(img[self.current_t],dest=0,tag=40)
                    
                    
                if self.current_t == self.data.n_batch_train - 1:
                    self.last_one_t = True
                    # Only to get the last copy_finished signal from load
                    icomm.isend(img[self.current_t],dest=0,tag=40) 
                else:
                    self.last_one_t = False
                    # 4. give preload signal to load next file
                    icomm.isend(img[self.current_t+1],dest=0,tag=40)
                    
                # 5. wait for the batch to be loaded into shared_x
                msg = icomm.recv(source=0,tag=55) #
                assert msg == 'copy_finished'
                    
            
            else:
            
                img_mean = self.data.rawdata[-1]
                import hickle as hkl
                arr = hkl.load(img[self.current_t]) - img_mean

                from theanompi.models.data.utils import crop_and_mirror

                arr = crop_and_mirror(arr, mode, 
                                    rand_crop, 
                                    batch_crop_mirror, 
                                    input_width)
             
                self.shared_x.set_value(arr)
                
                if self.current_t == self.data.n_batch_train - 1:
                    self.last_one_t = True
                else:
                    self.last_one_t = False
                    
                
            # direct loading of shared_y
            self.shared_y.set_value(labels[self.current_t])
                
        
            recorder.end('wait')
                
        recorder.start()
        
        if self.verbose: 
            if self.monitor_grad: 
                #print np.array(self.get_norm(self.subb_t))
                print(np.array(self.get_norm(self.subb_t)).tolist()[:2])
                
        cost,error= function(self.subb_t)
        
        for p in self.params: p.container.value.sync()
        
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
        
    def val_iter(self, count,recorder):
        
        '''use the val_iter_fn compiled'''
        
        if self.current_v==0 and self.subb_v == 0:
            self.data.shuffle_data(mode='val')
            self.data.shard_data(mode='val',rank=self.rank, size=self.size)
        
        img= self.data.val_img_shard
        labels = self.data.val_labels_shard
        
        mode='val'
        function=self.val_iter_fn
        
        if self.subb_v == 0: # load the whole file into shared_x when loading sub-batch 0 of each file.
        
            # parallel loading of shared_x
            if self.data.para_load:
                
                icomm = self.data.icomm
            
                if self.current_v == 0:
                
                    # 3.0 give mode signal to adjust loading mode between train and val
                    icomm.isend('val',dest=0,tag=40)
                    # 3.1 give load signal to load the very first file
                    icomm.isend(img[self.current_v],dest=0,tag=40)
                
                
                if self.current_v == self.data.n_batch_val - 1:
                    
                    self.last_one_v = True
                    # Only to get the last copy_finished signal from load
                    icomm.isend(img[self.current_v],dest=0,tag=40) 
                    
                else:
                    
                    self.last_one_v = False
                    # 4. give preload signal to load next file
                    icomm.isend(img[self.current_v+1],dest=0,tag=40)
                    
                
                # 5. wait for the batch to be loaded into shared_x
                msg = icomm.recv(source=0,tag=55) #
                assert msg == 'copy_finished'
                
        
            else:
        
    
                img_mean = self.data.rawdata[-1]
                import hickle as hkl
                arr = hkl.load(img[self.current_v]) - img_mean

                from theanompi.models.data.utils import crop_and_mirror

                arr = crop_and_mirror(arr, mode, 
                                    rand_crop, 
                                    batch_crop_mirror, 
                                    input_width)
        
                # arr = np.rollaxis(arr,0,4)
                                
                self.shared_x.set_value(arr)
                
                
            # direct loading of shared_y    
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
        
        'to be called once per epoch'
        
        if lr_policy == 'step':
            
            if epoch in lr_step: 
                
                tuned_base_lr = self.shared_lr.get_value() /10.
        
                self.shared_lr.set_value(np.float32(tuned_base_lr))
        
    def cleanup(self):
        
        if self.data.para_load:
            
            self.data.para_load_close()
        
if __name__ == '__main__':
    
    raise RuntimeError('to be tested using test_model.py:\n$ python test_model.py alex_net AlexNet')

