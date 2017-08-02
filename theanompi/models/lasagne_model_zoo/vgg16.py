import numpy as np
import lasagne
from lasagne.layers import InputLayer, DimshuffleLayer
from lasagne.layers import DenseLayer,DropoutLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
    
def build_model_vgg16(input_shape, verbose):
    
    '''
    See Lasagne Modelzoo:
    https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
    
    '''
    
    if verbose: print('VGG16 (from lasagne model zoo)')
    
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
    

# def build_model_vgg_cnn_s(input_shape, verbose):
#
#     '''
#     See Lasagne Modelzoo:
#     https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg_cnn_s.py
#
#     '''
#     if verbose: print 'VGG_cnn_s (from lasagne model zoo)'
#
#     net = {}
#     net['input'] = InputLayer(input_shape)
#     net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
#     net['norm1'] = LRNLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
#     net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
#     net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
#     net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
#     net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
#     net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
#     net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
#     net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
#     net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
#     net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
#     net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
#     net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
#     net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
#
#     if verbose:
#         for layer in net.values():
#             print str(lasagne.layers.get_output_shape(layer))
#
#     return net


# model hyperparams
n_epochs = 40
momentum = 0.90
weight_decay = 0.0005
batch_size = 64
file_batch_size = 128
learning_rate = 0.002
# size=1 converge at 1320 6.897861; if avg and not scale lr by size, then size=2 will converge at: 2360 6.898975
lr_policy = 'step'
lr_step = [15, 30, 35]

use_momentum = True
use_nesterov_momentum = False

#cropping hyperparams
input_width = 224
input_height = 224

batch_crop_mirror = True 
rand_crop = True

image_mean = np.array([103.939, 116.779, 123.68],dtype='float32')[:,np.newaxis,np.newaxis,np.newaxis]
dataname = 'imagenet'

monitor_grad = False


class VGG16(object): # c01b input

    '''

    overwrite those methods in the ModelBase class


    '''
    
    def __init__(self,config): 
        
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
        self.name = 'VGG16'
        
        # data
        from theanompi.models.data import ImageNet_data
        self.data = ImageNet_data(verbose=False)
        self.data.rawdata[4] = image_mean
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
        
        net = build_model_vgg16(input_shape=(None, 3, 224, 224), verbose=self.verbose)
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
        
        print('Compile time: %.3f s' % (time.time()-start))
            
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
            
                img_mean = self.data.rawdata[4]
                img_std = self.data.rawdata[5]
                import hickle as hkl
                arr = (hkl.load(img[self.current_t]) - img_mean)/255./img_std

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
        
        cost,error= function(self.subb_t)
        
        for p in self.params: p.container.value.sync()
        
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
        
    
                img_mean = self.data.rawdata[4]
                img_std = self.data.rawdata[5]
                import hickle as hkl
                arr = (hkl.load(img[self.current_v]) - img_mean)/255./img_std

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
        
        'to be called once per epoch'
        
        if lr_policy == 'step':
            
            if epoch in lr_step: 
                
                tuned_base_lr = self.shared_lr.get_value() /10.
        
                self.shared_lr.set_value(np.float32(tuned_base_lr))
        
    def scale_lr(self, size):
        
        self.shared_lr.set_value(np.array(self.shared_lr.get_value()*size, dtype='float32'))
        
    def cleanup(self):
        
        if self.data.para_load:
            
            self.data.para_load_close()
        
    def load(self):
        
        # wget !wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
        
        import pickle

        with open('vgg16.pkl') as f:
            model = pickle.load(f)
        
        # CLASSES = model['synset words']
        # MEAN_IMAGE = model['mean image']

        lasagne.layers.set_all_param_values(self.output_layer, model['values'])
        
if __name__ == '__main__':
    
    raise RuntimeError('to be tested using test_model.py:\n$ python test_model.py lasagne_model_zoo.vgg VGG16')