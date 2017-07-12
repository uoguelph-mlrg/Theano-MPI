import numpy as np
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax

import pickle

def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    names : list of string
        Names of the layers in block

    num_filters : int
        Number of filters in convolution layer

    filter_size : int
        Size of filters in convolution layer

    stride : int
        Stride of convolution layer

    pad : int
        Padding of convolution layer

    use_bias : bool
        Whether to use bias in conlovution layer

    nonlin : function
        Nonlinearity type of Nonlinearity layer

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, stride, pad,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block

    ratio_size : float
        Scale factor of filter size

    has_left_branch : bool
        if True, then left branch contains simple block

    upscale_factor : float
        Scale factor of filter bank at the output of residual block

    ix : int
        Id of residual block

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix
    

def build_model_resnet50(input_shape): 
    net = {}
    net['input'] = InputLayer(input_shape)
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 2, 3, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)
    
    # block_size = ['a'] + ['b'+str(i+1) for i in range(7)]
    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)
    
    # block_size = ['a'] + ['b'+str(i+1) for i in range(35)]
    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)
    
    print('Total number of layers:', len(lasagne.layers.get_all_layers(net['prob'])))

    return net

# model hyperparams
n_epochs = 120 # 60E4/5004 = 119.9
momentum = 0.90
weight_decay = 0.0001
batch_size = 64  # ResNet50: 16->3.5G Mem, 64->10G Mem
file_batch_size = 128
learning_rate = 0.025 # 0.1 for 256 batch size; 0.1/8 for 32 batch size

lr_policy = 'step'
lr_step = [30, 60, 110]

use_momentum = True
use_nesterov_momentum = False

#cropping hyperparams
input_width = 224
input_height = 224

batch_crop_mirror = True 
rand_crop = True
dataname = 'imagenet'

monitor_grad = False

class ResNet50(object):
    '''
    Residual Network on ImageNet
    
    # adjusted from https://github.com/kundan2510/resnet152-pre-trained-imagenet/blob/master/resnet152.py
    # and https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/ImageNet%20Pretrained%20Network%20(ResNet-50).ipynb
    
    # "Deep Residual Learning for Image Recognition"
    # http://arxiv.org/pdf/1512.03385v1.pdf
    # License: see https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE
    
    '''
    def __init__(self, config):
        
        self.verbose = config['verbose']
        self.rank = config['rank']
        self.size = config['size']
        self.no_paraload=False
        try: 
            self.no_paraload = config['no_paraload']
        except:
            pass

        import theano
        theano.config.on_unused_input = 'warn'
        self.name = 'ResNet50'

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
                                                self.input_width, 
                                                self.input_height,
                                                self.file_batch_size
                                                ), 
                                                dtype=theano.config.floatX),  
                                                borrow=True)                           
        self.shared_y = theano.shared(np.zeros((self.file_batch_size,), 
                                          dtype=int),   borrow=True) 
        # slice batch if needed
        import theano.tensor as T                     
        subb_ind = T.iscalar('subb')  # sub batch index
        self.subb_ind = subb_ind
        self.shared_x_slice = self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]
        self.shared_y_slice = self.shared_y[subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]                             

        # build model                                 
        self.build_model() # bc01

        from lasagne.layers import get_all_params
        self.params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
        from theanompi.models.layers2 import count_params, extract_weight_types
        self.weight_types = extract_weight_types(self.params)
        if self.verbose: count_params(self.params, self.verbose)

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
                                  
        subb_ind = T.iscalar('subb')  # sub batch index
        #print self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size].shape.eval()
        self.subb_ind = subb_ind
        self.shared_x_slice = self.shared_x[:,:,:,subb_ind*self.batch_size:(subb_ind+1)*self.batch_size].dimshuffle(3, 0, 1, 2) # c01b to bc01
        self.shared_y_slice = self.shared_y[subb_ind*self.batch_size:(subb_ind+1)*self.batch_size]


        if self.data.para_load and not self.no_paraload:
    
            self.data.spawn_load()
            self.data.para_load_init(self.shared_x, input_width, input_height, 
                                    rand_crop, batch_crop_mirror)
        
    def build_model(self):
        
        import theano.tensor as T
        self.x = T.ftensor4('x')
        self.y = T.lvector('y')
        self.lr = T.scalar('lr')
        
        net = build_model_resnet50(input_shape=(None, 3, 224, 224))
        
        self.output_layer = net['prob']
        
        from lasagne.layers import get_output
        self.output = lasagne.layers.get_output(self.output_layer, self.x, deterministic=False)
        self.cost = lasagne.objectives.categorical_crossentropy(self.output, self.y).mean()
        from lasagne.objectives import categorical_accuracy
        self.error = 1-categorical_accuracy(self.output, self.y, top_k=1).mean()
        self.error_top_5 = 1-categorical_accuracy(self.output, self.y, top_k=5).mean()
        
    def compile_train(self, *args):
        
        # args is a list of dictionaries
        
        if self.verbose: print('compiling training function...')
        
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

        if self.verbose: print('compiling inference function...')
        
        from lasagne.layers import get_output
        
        output_inference = lasagne.layers.get_output(self.output_layer, self.x, deterministic=True)
    
        self.inf_fn = theano.function([self.x],output_inference)
        
    def compile_val(self):

        if self.verbose: print('compiling validation function...')
        
        import theano
        
        from lasagne.layers import get_output
        
        output_val = lasagne.layers.get_output(self.output_layer, self.x, deterministic=True)
        
        from lasagne.objectives import categorical_accuracy, categorical_crossentropy
        
        cost = categorical_crossentropy(output_val, self.y).mean()
        error = 1-categorical_accuracy(output_val, self.y, top_k=1).mean()
        error_top_5 = 1-categorical_accuracy(output_val, self.y, top_k=5).mean()
        
        self.val_fn=  theano.function([self.subb_ind], [cost,error,error_top_5], updates=[], 
                                          givens=[(self.x, self.shared_x_slice),
                                                  (self.y, self.shared_y_slice)]
                                                                )
    
    def compile_iter_fns(self, sync_type):
        
        import time
        
        start = time.time()
        
        from theanompi.lib.opt import pre_model_iter_fn

        pre_model_iter_fn(self, self.size)
        
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
            
                arr = hkl.load(img[self.current_t]) #- img_mean
             
                self.shared_x.set_value(arr)
                
                if self.current_t == self.data.n_batch_train - 1:
                    self.last_one_t = True
                else:
                    self.last_one_t = False
                    
                
            # direct loading of shared_y
            self.shared_y.set_value(labels[self.current_t])
                
        
            recorder.end('wait')
                
        recorder.start()
        
        cost,error= function(self.subb_t)
        
        if self.verbose: 
            if self.monitor_grad: 
                print(np.array(self.get_norm(self.subb_t)))
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
        
    
                arr = hkl.load(img[self.current_v]) #- img_mean
        
                # arr = np.rollaxis(arr,0,4)
                                
                self.shared_x.set_value(arr)
                
                
            # direct loading of shared_y    
            self.shared_y.set_value(labels[self.current_v])
        
        
            if self.current_v == self.data.n_batch_val - 1:
                self.last_one_v = True
            else:
                self.last_one_v = False
        
        # no need to set drop out off, it is taken care of by setting "deterministic=True"
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
        
    def scale_lr(self, size):
        
        self.shared_lr.set_value(np.array(self.shared_lr.get_value()*size, dtype='float32'))
        
    def cleanup(self):
        
        if self.data.para_load:
            
            self.data.para_load_close()
        
        
    def load(self):
        
        model = pickle.load(open(model_pkl_file,'rb'))
        net = build_model()
        lasagne.layers.set_all_param_values(net['prob'], model['values'])
        return net, model['mean_image'], model['synset_words']
        
        
if __name__ == '__main__':
    
    raise RuntimeError('to be tested using test_model.py:\n$ python test_model.py lasagne_model_zoo.resnet50 ResNet50')



