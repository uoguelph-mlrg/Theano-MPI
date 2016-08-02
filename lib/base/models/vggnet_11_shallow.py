import theano
import theano.tensor as T
import numpy as np
from layers2 import Conv,Pool,Dropout,FC,Softmax,Flatten,LRN, \
                    HeUniform, HeNormal, Constant, Normal, \
                    get_params, get_layers
from modelbase import ModelBase

# other tools minerva, chainer


class VGGNet_11(ModelBase): # c01b input
    
    def __init__(self,config): 
        ModelBase.__init__(self)  

        self.config = config
        self.verbose = self.config['verbose']
        self.build_model()
        
    def build_model(self):
        
        print 'VGGNet_11 (shallow) 3/19'
        
        self.name = 'vggnet'
        
        # input shape in c01b 
        self.channels = 3 # 'c' mean(R,G,B) = (103.939, 116.779, 123.68)
        self.input_width = self.config['input_width'] # '0' single scale training 224
        self.input_height = self.config['input_height'] # '1' single scale training 224
        self.batch_size = self.config['batch_size'] # 'b'
        b = self.batch_size
        
        # output dimension
        self.n_softmax_out = self.config['n_softmax_out']
        
        # start graph construction from scratch
        self.x = T.ftensor4('x')
        
        self.y = T.lvector('y')
        
        x_shuffled = self.x.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        
        layers = []
        params = []
        weight_types = [] # for distinguishing w and b later      
        
        # bc01 from now on
        
        conv_3x3 = Conv(input=x_shuffled,
                        input_shape=(b,
                                    self.channels,
                                    self.input_width,
                                    self.input_height), # (b, 3, 224, 224)
                        convstride=1,
                        padsize=1,
                        W = Normal((64, self.channels, 3, 3), std=0.3), # bc01
                        b = Constant((64,), val=0.2),
                        printinfo=self.verbose
                        #output_shape = (b, 64, 224, 224)
                        )

        pool_2x2 = Pool(input=conv_3x3, 
                        #input_shape=conv_3x3.output_shape, # (b, 64, 224, 224)
                        poolsize=2, 
                        poolstride=2, 
                        poolpad=0,
                        mode = 'max',
                        printinfo=self.verbose
                        #output_shape = (b, 64, 112, 112)
                        )

        conv_3x3 = Conv(input=pool_2x2,
                        #input_shape=pool_2x2.output_shape, # (b, 64, 112, 112)
                        convstride=1,
                        padsize=1,
                        W = Normal((128, pool_2x2.output_shape[1], 3, 3), std=0.1), # bc01
                        b = Constant((128,), val=0.02),
                        printinfo=self.verbose
                        #output_shape = (b, 128, 112, 112)
                        )

        pool_2x2 = Pool(input=conv_3x3, 
                        #input_shape=conv_3x3.output_shape, # (b, 128, 112, 112)
                        poolsize=2, 
                        poolstride=2, 
                        poolpad=0,
                        mode = 'max',
                        printinfo=self.verbose
                        #output_shape = (b, 128, 56, 56)
                        )

        conv_3x3 = Conv(input=pool_2x2,
                        #input_shape=pool_2x2.output_shape, # (b, 128, 56, 56)
                        convstride=1,
                        padsize=1,
                        W = Normal((256, pool_2x2.output_shape[1], 3, 3), std=0.05), # bc01
                        b = Constant((256,), val=0.02),
                        printinfo=self.verbose
                        #output_shape = (b, 256, 56, 56)
                        )
        conv_3x3 = Conv(input=conv_3x3,
                        #input_shape=conv_3x3.output_shape, # (b, 256, 56, 56)
                        convstride=1,
                        padsize=1,
                        W = Normal((256, conv_3x3.output_shape[1], 3, 3), std=0.05), # bc01
                        b = Constant((256,), val=0.01),
                        printinfo=self.verbose
                        #output_shape = (b, 256, 56, 56)
                        )

        pool_2x2 = Pool(input=conv_3x3, 
                        #input_shape=conv_3x3.output_shape, # (b, 256, 56, 56)
                        poolsize=2, 
                        poolstride=2, 
                        poolpad=0,
                        mode = 'max',
                        printinfo=self.verbose
                        #output_shape = (b, 256, 28, 28)
                        )

        conv_3x3 = Conv(input=pool_2x2,
                        #input_shape=pool_2x2.output_shape, # (b, 256, 28, 28)
                        convstride=1,
                        padsize=1,
                        W = Normal((512, pool_2x2.output_shape[1], 3, 3), std=0.05), # bc01
                        b = Constant((512,), val=0.02),
                        printinfo=self.verbose
                        #output_shape = (b, 512, 28, 28)
                        )
        conv_3x3 = Conv(input=conv_3x3,
                        #input_shape=conv_3x3.output_shape, # (b, 512, 28, 28)
                        convstride=1,
                        padsize=1,
                        W = Normal((512, conv_3x3.output_shape[1], 3, 3), std=0.01), # bc01
                        b = Constant((512,), val=0.01),
                        printinfo=self.verbose
                        #output_shape = (b, 512, 28, 28)
                        )

        pool_2x2 = Pool(input=conv_3x3, 
                        #input_shape=conv_3x3.output_shape, # (b, 512, 28, 28)
                        poolsize=2, 
                        poolstride=2, 
                        poolpad=0,
                        mode = 'max',
                        printinfo=self.verbose
                        #output_shape = (b, 512, 14, 14)
                        )

        conv_3x3 = Conv(input=pool_2x2,
                        #input_shape=pool_2x2.output_shape, # (b, 512, 14, 14)
                        convstride=1,
                        padsize=1,
                        W = Normal((512, pool_2x2.output_shape[1], 3, 3), std=0.005), # bc01
                        b = Constant((512,)),
                        printinfo=self.verbose
                        #output_shape = (b, 512, 14, 14)
                        )
        conv_3x3 = Conv(input=conv_3x3,
                        #input_shape=conv_3x3.output_shape, # (b, 512, 14, 14)
                        convstride=1,
                        padsize=1,
                        W = Normal((512, conv_3x3.output_shape[1], 3, 3), std=0.005), # bc01
                        b = Constant((512,)),
                        printinfo=self.verbose
                        #output_shape = (b, 512, 14, 14)
                        )
 
        pool_2x2 = Pool(input=conv_3x3, 
                        #input_shape=conv_3x3.output_shape, # (b, 512, 14, 14)
                        poolsize=2, 
                        poolstride=2, 
                        poolpad=0,
                        mode = 'max',
                        printinfo=self.verbose
                        #output_shape = (b, 512, 7, 7)
                        )

        flatten = Flatten(input = pool_2x2, #5
                        #input_shape = pool_2x2.output_shape, # (b, 512, 7, 7)
                        axis = 2, # expand dimensions after the first dimension
                        printinfo=self.verbose
                        #output_shape = (b, 25088)
                        )
        fc_4096 = FC(input= flatten, 
                        n_out=4096,
                        W = Normal((flatten.output_shape[1], 4096), std=0.001),
                        b = Constant((4096,),val=0.01),
                        printinfo=self.verbose
                        #input_shape = flatten.output_shape # (b, 25088)
                        )
        dropout= Dropout(input=fc_4096,
                        n_out=fc_4096.output_shape[1], 
                        prob_drop=0.5,
                        printinfo=self.verbose
                        #input_shape = fc_4096.output_shape # (b, 4096)
                        )
        fc_4096 = FC(input= dropout,  
                        n_out=4096,
                        W = Normal((dropout.output_shape[1], 4096), std=0.005),
                        b = Constant((4096,),val=0.01),
                        printinfo=self.verbose
                        #input_shape = dropout.output_shape # (b, 4096)
                        )
        dropout= Dropout(input=fc_4096, 
                        n_out=fc_4096.output_shape[1], 
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
        
        self.output = self.output_layer.output
        
        self.layers = get_layers(lastlayer = self.output_layer)
        
        self.layers = [layer for layer in self.layers \
            if layer.name not in ['LRN\t','Pool\t','Flatten\t','Dropout'+ str(0.5)]]
        
        self.params,self.weight_types = get_params(self.layers)
        
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
        
        self.grads = T.grad(self.cost,self.params)
        
        subb_ind = T.iscalar('subb')  # sub batch index
        #print self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size].shape.eval()
        self.subb_ind = subb_ind
        self.shared_x_slice = self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
        self.shared_y_slice = self.shared_y[subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
    
    def compile_train(self):
        
        print 'compiling training function...'
        
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
        
        self.shared_lr.set_value(tuned_base_lr)
            
    def test(self):
        
        self.train(0)
        #self.val(0)
                  
                            
                            
if __name__ == '__main__': 
    
    import yaml
    with open('../../../run/config.yaml', 'r') as f:
        config = yaml.load(f)
        
    with open('../../../run/vggnet.yaml', 'r') as f:
        model_config = yaml.load(f)
    config = dict(config.items()+model_config.items())
    
    import pycuda.driver as drv
    drv.init()
    
    model = VGGNet_16(config)
    
    # simple test
    
    model.compile_train()
    
    free, total = drv.mem_get_info()
    
    print '%.1f %% of device memory is free.' % ((free/float(total))*100)
    
    print model.train(0)
    
    free, total = drv.mem_get_info()
    
    print '%.1f %% of device memory is free.' % ((free/float(total))*100)
    
    print model.train(0)
    
    free, total = drv.mem_get_info()
    
    print '%.1f %% of device memory is free.' % ((free/float(total))*100)
    
    model.compile_val()
    
    free, total = drv.mem_get_info()
    
    print '%.1f %% of device memory is free.' % ((free/float(total))*100)
    
    print model.val(0)
    
    free, total = drv.mem_get_info()
    
    print '%.1f %% of device memory is free.' % ((free/float(total))*100)
    
    model.compile_inference()
    
    print model.inference(model.shared_x.get_value()[0])
    
    model.adjust_lr(epoch=40,size=1)
    
    
                                              



    
    

    
    
    
    
    
