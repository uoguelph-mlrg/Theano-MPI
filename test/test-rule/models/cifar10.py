import numpy as np
import sys
sys.path.append('../')

from lib.helper_funcs import get_rand3d
from lib.proc_load_mpi import crop_and_mirror
        
# model hyperparams

n_epochs = 70
momentum = 0.90
weight_decay = 0.0001
file_batch_size = 128
batch_size = 128
learning_rate = 0.01

lr_policy = 'step'
lr_step = [30, 50, 65]

use_momentum = True
use_nesterov_momentum = False

#cropping hyperparams
input_width = 28
input_height = 28
batch_crop_mirror = False
rand_crop = True

image_mean = 'img_mean'
dataname = 'cifar10'

class Cifar10_model(object): # c01b input
    
    def __init__(self,config): 

        self.verbose = config['verbose']
        
        import theano
        self.name = 'Cifar10_model'
        
        # data
        from data.cifar10 import Cifar10_data
        self.data = Cifar10_data(self.verbose)
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
        # build model
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
        from layers2 import Conv,Pool,Dropout,FC,Softmax,Flatten,LRN, \
                            HeUniform, HeNormal, Constant, Normal
        
        self.x = T.ftensor4('x')
        
        self.y = T.lvector('y')
        
        self.lr = T.scalar('lr')
        
        x_shuffled = self.x.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        
        conv_5x5 = Conv(input=x_shuffled,
                        input_shape=(self.batch_size,
                                    self.channels,
                                    self.input_width,
                                    self.input_height), # (b, 3, 28, 28)
                        convstride=1,
                        padsize=0,
                        W = Normal((64, self.channels, 5, 5), std=0.05), # bc01
                        b = Constant((64,), val=0),
                        printinfo=self.verbose
                        #output_shape = (b, 64, 24, 24)
                        )

        pool_2x2 = Pool(input=conv_5x5, 
                        #input_shape=conv_3x3.output_shape, # (b, 64, 24, 24)
                        poolsize=2, 
                        poolstride=2, 
                        poolpad=0,
                        mode = 'max',
                        printinfo=self.verbose
                        #output_shape = (b, 64, 12, 12)
                        )
                        
        conv_5x5 = Conv(input=pool_2x2,
                        #input_shape=conv_2x2.output_shape, # (b, 64, 12, 12) 
                        convstride=1,
                        padsize=0,
                        W = Normal((128, pool_2x2.output_shape[1], 5, 5), std=0.05), # bc01
                        b = Constant((128,), val=0),
                        printinfo=self.verbose
                        #output_shape = (b, 128, 8, 8)
                        )
                        
        pool_2x2 = Pool(input=conv_5x5, 
                        #input_shape=conv_5x5.output_shape, # (b, 128, 8, 8)
                        poolsize=2, 
                        poolstride=2, 
                        poolpad=0,
                        mode = 'max',
                        printinfo=self.verbose
                        #output_shape = (b, 128, 4, 4)
                        )
                        
        conv_5x5 = Conv(input=pool_2x2,
                        #input_shape=pool_2x2.output_shape, # (b, 128, 4, 4)
                        convstride=1,
                        padsize=0,
                        W = Normal((64, pool_2x2.output_shape[1], 3, 3), std=0.05), # bc01
                        b = Constant((64,), val=0),
                        printinfo=self.verbose
                        #output_shape = (b, 64, 2, 2)
                        )
        
        # bc01 from now on

        flatten = Flatten(input = conv_5x5, #5
                        #input_shape=conv_5x5.output_shape, # (b, 64, 2, 2)
                        axis = 2, # expand dimensions after the first dimension
                        printinfo=self.verbose
                        #output_shape = (b,64*2*2)
                        )
                        
                        
        fc_256  = FC(input= flatten, 
                        n_out=256,
                        W = Normal((flatten.output_shape[1], 256), std=0.001),
                        b = Constant((256,),val=0),
                        printinfo=self.verbose
                        #input_shape = flatten.output_shape # (b, 9216)
                        )
        dropout= Dropout(input=fc_256,
                        n_out=fc_256.output_shape[1], 
                        prob_drop=0.5,
                        printinfo=self.verbose
                        #input_shape = fc_4096.output_shape # (b, 4096)
                        )
                        
                        
        softmax = Softmax(input=dropout,  
                        n_out=self.n_softmax_out,
                        W = Normal((dropout.output_shape[1], self.n_softmax_out), std=0.005),
                        b = Constant((self.n_softmax_out,),val=0),
                        printinfo=self.verbose
                        #input_shape = dropout.output_shape # (b, 4096)
                        )
        
        self.output_layer = softmax
        
        self.cost = softmax.negative_log_likelihood(self.y)     
        self.error = softmax.errors(self.y)
        self.error_top_5 = softmax.errors_top_x(self.y)
        
    
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

        pre_model_iter_fn(model, sync_type='avg')
            
    def iter_reset(self, mode):
        
        self.current = 0
        self.subb=0
        
    def train_iter(self,count,recorder):
        

        img= self.data.train_img
        labels = self.data.train_labels
        img_mean = self.data.rawdata[4]
        mode='train'
        function=self.train_iter_fn
            
        if self.subb_t == 0: # load the whole file into shared_x when loading sub-batch 0 of each file.
              
            if self.current_t == 0:

                self.data.shuffle_data()
        
            recorder.start()
        
            arr = img[self.current_t] - img_mean
            
            arr = np.rollaxis(arr,0,4)
            
            rand_arr = get_rand3d(rand_crop, mode)

            arr = crop_and_mirror(arr, rand_arr, \
                                    flag_batch=self.batch_crop_mirror, \
                                    cropsize = self.input_width)
                                    
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
        
        img= self.data.val_img
        labels = self.data.val_labels
        img_mean = self.data.rawdata[4]
        mode='val'
        function=self.val_iter_fn
        
        if self.subb_v == 0: # load the whole file into shared_x when loading sub-batch 0 of each file.
        
    
            arr = img[self.current_v] - img_mean
        
            arr = np.rollaxis(arr,0,4)
        
            rand_arr = get_rand3d(rand_crop, mode)

            arr = crop_and_mirror(arr, rand_arr, \
                                    flag_batch=self.batch_crop_mirror, \
                                    cropsize = self.input_width)
                                
            self.shared_x.set_value(arr)
            self.shared_y.set_value(labels[self.current_v])
        
        
            if self.current_v == self.data.n_batch_val - 1:
                self.last_one_v = True
            else:
                self.last_one_v = False
            
        cost,error,error_top5 = function(self.subb_v)
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
    
    model = Cifar10_model(config)
    
    model.compile_iter_fns()
    
    # get recorder
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    from lib.recorder import Recorder
    recorder = Recorder(comm, printFreq=4, modelname='cifar10', verbose=True)
    
    # train
    
    for batch_i in range(model.data.n_batch_train):
        
        model.train_iter(batch_i, recorder)
        
        #recorder.print_train_info(batch_i)
        
        print batch_i
        
    # val
    
    for batch_j in range(model.data.n_batch_val):
        
        model.val_iter(batch_i, recorder)
        
        print batch_j
        
    recorder.print_val_info(batch_i)
    
    print 'finish one epoch'
    
    model.batch_size=1
    model.compile_inference()
    
    print model.inf_fn(model.shared_x.get_value()[:,:,:,:1])
    
    model.adjust_hyperp(epoch=40)