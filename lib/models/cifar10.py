import theano
import theano.tensor as T
import numpy as np
from layers2 import Conv,Pool,Dropout,FC,Softmax,Flatten,LRN, \
                    HeUniform, HeNormal, Constant, Normal, \
                    get_params, get_layers
from modelbase import ModelBase

# other tools minerva, chainer


class CNNCifar10(ModelBase): # c01b input
    
    def __init__(self,config): 
        ModelBase.__init__(self)  

        self.config = config
        self.verbose = self.config['verbose']
        
        self.name = 'cnncifar10'
        
        # input shape in c01b 
        from data.cifar10 import Cifar10_data
        
        data = Cifar10_data()
        
        self.channels = data.channels # 'c' mean(R,G,B) = (103.939, 116.779, 123.68)
        self.input_width = data.input_width # '0' single scale training 224
        self.input_height = data.input_height # '1' single scale training 224
        self.batch_size = data.batch_size # 'b'
        
        # output dimension
        self.n_softmax_out = data.n_class
        
        # training related
        self.base_lr = np.float32(self.config['learning_rate'])
        self.shared_lr = theano.shared(self.base_lr)
        self.step_idx = 0
        self.mu = self.config['momentum'] # def: 0.9 # momentum
        self.eta = self.config['weight_decay'] #0.0002 # weight decay
        
        self.shared_x = theano.shared(np.zeros((
                                                3,
                                                self.input_width, 
                                                self.input_height,
                                                self.config['file_batch_size']
                                                ), 
                                                dtype=theano.config.floatX),  
                                                borrow=True)
                                              
        self.shared_y = theano.shared(np.zeros((self.config['file_batch_size'],), 
                                          dtype=int),   borrow=True)                                  
        
        # build model
        self.build_model()
        
        
        self.output_layer = softmax
        
        self.output = self.output_layer.output
        
        self.layers = get_layers(lastlayer = self.output_layer)
        
        
        self.layers = [layer for layer in self.layers \
            if layer.name not in ['LRN\t','Pool\t','Flatten\t','Dropout'+ str(0.5)]]
            
        self.params,self.weight_types = get_params(self.layers)

        # if multi-stream layers exist, redefine and abstract into one layer class in layers2.py
        
        # count params
        self.count_params()
            
        self.grads = T.grad(self.cost,self.params)
        
        self.cost = softmax.negative_log_likelihood(self.y)     
        self.error = softmax.errors(self.y)
        self.error_top_5 = softmax.errors_top_x(self.y)
        
        self.grads = T.grad(self.cost,self.params)
        
        subb_ind = T.iscalar('subb')  # sub batch index
        #print self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size].shape.eval()
        self.subb_ind = subb_ind
        self.shared_x_slice = self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
        self.shared_y_slice = self.shared_y[subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
        
    def build_model(self):
        
        
        if self.verbose: print 'Customized MLP'

        # start graph construction from scratch
        
        self.x = T.ftensor4('x')
        
        self.y = T.lvector('y')
        
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
                        W = Normal((128, self.channels, 5, 5), std=0.05), # bc01
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
                        W = Normal((64, self.channels, 3, 3), std=0.05), # bc01
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
                        
                        
        fc256  = FC(input= flatten, 
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
                        
                        
                        
        fc_10  = FC(input= dropout,  
                        n_out=10,
                        W = Normal((dropout.output_shape[1], 10), std=0.005),
                        b = Constant((10,),val=0),
                        printinfo=self.verbose
                        #input_shape = dropout.output_shape # (b, 4096)
                        )
        dropout= Dropout(input=fc_10, 
                        n_out=fc_10.output_shape[1], 
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
        
    def count_params(self):
        
        size=0
        for param in self.params:
            
            size+=param.size.eval()
            
            print param.shape.eval()
            
        self.model_size = size
            
        print 'model size %d' % int(self.model_size)
    
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
    
        self.inference = theano.function([self.x],self.output)
        
    def compile_val(self):

        if self.verbose: print 'compiling validation function...'
        
        self.val =  theano.function([self.subb_ind], [self.cost,self.error,self.error_top_5], updates=[], 
                                          givens=[(self.x, self.shared_x_slice),
                                                  (self.y, self.shared_y_slice)]
                                                                )
    def set_dropout_off(self):
        
        Dropout.SetDropoutOff()
    
    def set_dropout_on(self):
        
        Dropout.SetDropoutOn()
                                                                                                      
    def adjust_lr(self, epoch):
            
        '''
        borrowed from AlexNet
        '''
        # lr is calculated every time as a function of epoch and size
        
        if self.config['lr_policy'] == 'step':
            
            stp0,stp1,stp2 = self.config['lr_step']
            
            if epoch >=stp0 and epoch < stp1:

                self.step_idx = 1
        
            elif epoch >=stp1 and epoch < stp2:
                
                self.step_idx = 2

            elif epoch >=stp2 and epoch < self.config['n_epochs']:
                
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
                  
                            
                            
if __name__ == '__main__': 
    
    import yaml
    with open('../../../run/config.yaml', 'r') as f:
        config = yaml.load(f)
        
    with open('../../../run/customized.yaml', 'r') as f:
        model_config = yaml.load(f)
    config = dict(config.items()+model_config.items())
    
    # import pycuda.driver as drv
    # drv.init()
    
    model = Customized(config)
    
    # simple test
    
    model.compile_train()
    
    print model.train(0)
    
    model.compile_val()
    
    print model.val(0)
    
    model.compile_inference()
    
    print model.inference(model.shared_x.get_value()[0])
    
    model.adjust_lr(epoch=40,size=1)