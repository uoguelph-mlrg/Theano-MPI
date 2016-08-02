import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer,DropoutLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear



def build_model_vgg16(input_shape, verbose):
    
    '''
    See Lasagne Modelzoo:
    https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
    
    '''
    
    if verbose: print 'VGG16 (from lasagne model zoo)'
    
    
    net = {}
    net['input'] = InputLayer(input_shape)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
                                    
    # for layer in net.values():
    #     print str(lasagne.layers.get_output_shape(layer))
        
    return net
    

def build_model_vgg_cnn_s(input_shape, verbose):
    
    '''
    See Lasagne Modelzoo:
    https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg_cnn_s.py
    
    '''
    if verbose: print 'VGG_cnn_s (from lasagne model zoo)'
    
    net = {}
    net['input'] = InputLayer(input_shape)
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
    net['norm1'] = LRNLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
                                    
    if verbose:
        for layer in net.values():
            print str(lasagne.layers.get_output_shape(layer))
        
    return net
    

import numpy as np
import theano
import theano.tensor as T
rng = np.random.RandomState(23455)

import sys
sys.path.append('../lib/base/models/')
from modelbase import ModelBase 

class VGG(ModelBase): # c01b input

    '''

    overwrite those methods in the ModelBase class


    '''
    
    def __init__(self,config): 
        ModelBase.__init__(self)
        
        self.config = config
        self.verbose = config['verbose']
        
        self.name = 'vggnet'
        
        # input shape in c01b 
        self.channels = 3 # 'c' mean(R,G,B) = (103.939, 116.779, 123.68)
        self.input_width = self.config['input_width'] # '0' single scale training 224
        self.input_height = self.config['input_height'] # '1' single scale training 224
        self.batch_size = self.config['batch_size'] # 'b'
        
        # output dimension
        self.n_softmax_out = self.config['n_softmax_out']
        
        
        
        # training related
        self.base_lr = np.float32(self.config['learning_rate'])
        self.shared_lr = theano.shared(self.base_lr)
        self.step_idx = 0
        self.mu = config['momentum'] # def: 0.9 # momentum
        self.eta = config['weight_decay'] #0.0002 # weight decay
        
        self.x = T.ftensor4('x')
        self.y = T.lvector('y')
        self.lr = T.scalar('lr')      
        
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
        net = self.build_model(input_shape=(self.batch_size, 3, self.input_width, self.input_height)) # bc01
        #self.output_layer = net['fc8'] 
        self.output_layer = net['prob']
        
        from lasagne.layers import get_all_params
        self.params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
        self.extract_weight_types()
        self.pack_layers()
        
        # count params
        if self.verbose: self.count_params()
        
        from lasagne.layers import get_output
        self.output = lasagne.layers.get_output(self.output_layer, self.x, deterministic=False)
        self.cost = lasagne.objectives.categorical_crossentropy(self.output, self.y).mean()
        self.error = self.errors(self.output, self.y)
        
        
        self.grads = T.grad(self.cost,self.params)
                                          
        subb_ind = T.iscalar('subb')  # sub batch index
        #print self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size].shape.eval()
        self.subb_ind = subb_ind
        self.shared_x_slice = self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size].dimshuffle(3, 0, 1, 2) # c01b to bc01
        self.shared_y_slice = self.shared_y[subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
        
    def build_model(self, input_shape):
        
        return build_model_vgg16(input_shape, self.verbose)
        
    def count_params(self):
        
        size=0
        for param in self.params:
            
            size+=param.size.eval()
            
            #print param.shape.eval()
            
        self.model_size = size
            
        print 'model size %d' % int(self.model_size)
        
    def extract_weight_types(self):
        
        self.weight_types = []
        for param in self.params:
            
            if len(param.shape.eval())>1:
                
                weight_type= 'W'
            else:
                weight_type= 'b'
                
            self.weight_types.append(weight_type)
                
            #print param.shape.eval(), weight_type
            
    def pack_layers(self):
        
        from layers2 import Layer
        from layers2 import Weight
        from pprint import pprint
        
        self.layers = []
        
        for param in self.params:
            
            if len(param.shape.eval())>1:
                
                layer = Layer()
                
                W = Weight()
                W.val = param
                W.shape = tuple(param.shape.eval())
                layer.W = W
            else:
                
                b = Weight()
                b.val = param
                b.shape = tuple(param.shape.eval())
                layer.b = b
                
                self.layers.append(layer)
                
           # if self.verbose: pprint(vars(layer))
            
        # for layer in self.layers:
        #
        #     print hasattr(layer.W.val, 'get_value')
        #     print hasattr(layer.b.val, 'get_value')
        #
        #
        # print len(self.layers)
        #
        # exit(0)                  
        
    def errors(self, p_y_given_x, y):
        
        y_pred = T.argmax(p_y_given_x, axis=1)
        
        if y.ndim != y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(y_pred, y))
        else:
            raise NotImplementedError()

        
    def errors_top_x(self, p_y_given_x, y, num_top=5):                       
                                    
        if num_top != 5: print 'val errors from top %d' % num_top        
        
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_pred_top_x = T.argsort(p_y_given_x, axis=1)[:, -num_top:]
            y_top_x = y.reshape((y.shape[0], 1)).repeat(num_top, axis=1)
            return T.mean(T.min(T.neq(y_pred_top_x, y_top_x), axis=1))
        else:
            raise NotImplementedError()             
        
    def compile_train(self, updates_dict=None):

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

        print 'compiling inference function...'
        
        from lasagne.layers import get_output
        
        self.output_inference = lasagne.layers.get_output(self.output_layer, self.x, deterministic=True)
    
        self.inference = theano.function([self.x],self.output_inference)
        
    def compile_val(self):

        print 'compiling validation function...'
        
        from lasagne.layers import get_output
        
        self.output_val = lasagne.layers.get_output(self.output_layer, self.x, deterministic=True)
        
        self.cost_val = lasagne.objectives.categorical_crossentropy(self.output_val, self.y).mean()
        
        self.error_val = self.errors(self.output_val, self.y)
            
        self.error_top_5_val = self.errors_top_x(self.output_val, self.y, num_top=5)
        
        self.val =  theano.function([self.subb_ind], [self.cost_val,self.error_val,self.error_top_5_val], updates=[], 
                                          givens=[(self.x, self.shared_x_slice),
                                                  (self.y, self.shared_y_slice)]
                                                                )
                                                                
    def set_dropout_off(self):
        
        '''
        no need to call this function, since it's taken care of in lasagne by specifying (deterministic=True)
        '''
        
        pass
    
    def set_dropout_on(self):
        '''
        no need to call this function, since it's taken care of in lasagne
        '''
        
        pass
                                                                                     
    def adjust_lr(self, epoch, size):
            
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
            
        
    def load_params(self):
        
        # wget !wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl
        
        import pickle

        with open('vgg_cnn_s.pkl') as f:
            model = pickle.load(f)
        
        # CLASSES = model['synset words']
        # MEAN_IMAGE = model['mean image']

        lasagne.layers.set_all_param_values(self.output_layer, model['values'])
    
    