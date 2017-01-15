import sys
#from theano.sandbox.cuda import dnn
from theano.gpuarray import dnn
import numpy as np
import theano
import theano.tensor as T
theano.config.on_unused_input = 'warn'
import warnings
warnings.filterwarnings("ignore")
# from pylearn2.expr.normalize import CrossChannelNormalization
#
# LRN=CrossChannelNormalization

rng = np.random.RandomState(23455)
# set a fixed number for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights


class Weight(object):

    def __init__(self): #mean=0, std=0.01
        super(Weight, self).__init__()
        
        self.val = None
        self.shape = None
        
    def save_weight(self, dir, name):
        #print 'weight saved: ' + name
        np.save(dir + name + '.npy', self.val.get_value())

    def load_weight(self, dir, name):
        #print 'weight loaded: ' + name
        np_values = np.load(dir + name + '.npy')
        
        print self.shape, np_values.shape
        
        if self.shape != np_values.shape:
            raise ValueError('The weight to be loaded must be of the same shape. %s != %s' % (self.shape,np_values.shape))
        else:
            #print '%s == %s' % (self.shape,np_values.shape)
            pass
        self.np_values = np_values
        self.val.set_value(self.np_values)
        
class Constant(Weight):
    def __init__(self, shape, val=0):
        
        self.np_values = \
            val * np.ones(shape,
            dtype=theano.config.floatX)
        self.val = theano.shared(value=self.np_values)
        self.shape = shape
    
class Normal(Weight):
    def __init__(self, shape, mean=0, std=0.01):
    
        self.np_values = np.asarray(
            rng.normal(mean, std, shape), 
            dtype=theano.config.floatX)
        self.val = theano.shared(value=self.np_values)
        self.shape = shape

class Uniform(Weight):
    def __init__(self, shape, low, high):
        
        self.np_values = np.asarray(
            rng.uniform(low, high, shape), 
            dtype=theano.config.floatX)
        self.val = theano.shared(value=self.np_values)
        self.shape = shape
        
class GlorotNormal(Weight):
    def __init__(self, shape, gain=np.sqrt(2)):  
        
        # the choice of gain is related to 
        # the choice of the type of activation units
        # gain = sqrt(2) for 'relu'
        
        if len(shape)==4:
            fan_in = shape[1] # c
            fan_out = shape[0] # b
            std = gain*np.sqrt(6.0 / ((fan_in + fan_out) * shape[2]*shape[3]))
        elif len(shape)==2:
            fan_in = shape[1] # c
            fan_out = shape[0] # b
            std = gain*np.sqrt(6.0 / (fan_in + fan_out))
        else:
            raise NotImplementedError
        
        self.np_values = np.asarray(
            rng.normal(0, std, shape), 
            dtype=theano.config.floatX)
        self.val = theano.shared(value=self.np_values)
        self.shape = shape # bc01

class GlorotUniform(Weight):
    def __init__(self, shape, gain=np.sqrt(2)): 
        
        # the choice of gain is related to 
        # the choice of the type of activation units
        # gain = sqrt(2) for relu
        if len(shape)==4:
            fan_in = shape[1] # c
            fan_out = shape[0] # b
            std = gain*np.sqrt(6.0 / ((fan_in + fan_out) * shape[2]*shape[3]))
        elif len(shape)==2:
            fan_in = shape[1] # c
            fan_out = shape[0] # b
            std = gain*np.sqrt(6.0 / (fan_in + fan_out))
        else:
            raise NotImplementedError
            
        low = -1 * std
        high = std
        self.np_values = np.asarray(
            rng.uniform(low, high, shape), 
            dtype=theano.config.floatX)
        self.val = theano.shared(value=self.np_values)
        self.shape = shape # bc01
        
class HeUniform(Weight):
    def __init__(self,shape,gain=np.sqrt(2)):
    
        # the choice of gain is related to 
        # the choice of the type of activation units
        # gain = sqrt(2) for relu
        if len(shape)==4:
            fan_in = shape[1]*shape[2]*shape[3] # c*0*1
            std = gain*np.sqrt(3.0 / fan_in)
        elif len(shape)==2:
            fan_in = shape[1] # c
            std = gain*np.sqrt(3.0 / fan_in)
        else:
            raise NotImplementedError
            
        low = -1 * std
        high = std
        self.np_values = np.asarray(
            rng.uniform(low, high, shape), 
            dtype=theano.config.floatX)
        self.val = theano.shared(value=self.np_values)
        self.shape = shape # bc01

class HeNormal(Weight):
    def __init__(self, shape, gain=np.sqrt(2)):  
        
        # the choice of gain is related to 
        # the choice of the type of activation units
        # gain = sqrt(2) for relu
        if len(shape)==4:
            fan_in = shape[1]*shape[2]*shape[3] # c*0*1
            std = gain*np.sqrt(3.0 / fan_in)
        elif len(shape)==2:
            fan_in = shape[1] # c
            std = gain*np.sqrt(3.0 / fan_in)
        else:
            raise NotImplementedError
        
        self.np_values = np.asarray(
            rng.normal(0, std, shape), 
            dtype=theano.config.floatX)
        self.val = theano.shared(value=self.np_values)
        self.shape = shape # bc01
        
        
class Layer(object):
    '''
    Base Layer class with shape evaluation tools
    '''
    def __init__(self):
        
        self.W=None
        self.b=None
        
    def get_input_shape(self,input,input_shape):
    
        if hasattr(input, 'output'):
            self.input_layer = input
            self.input = self.input_layer.output
            self.input_shape = np.array(self.input_layer.output_shape) #bc01
        elif input_shape:
            self.input = input
            self.input_shape = input_shape
        else:
            raise AttributeError('input layer should have *output* attribute')
        
    def get_output_shape(self,input_shape):
    
        '''
        helper functin to calcuate the output shape of a layer on the run
    
        the output shape of a Conv layer can also be calculated based on the fomular in
    
        http://cs231n.github.io/convolutional-networks/
    
        input_shape = (W_in, H_in, D_in) # 01c
        K filters with shape (F,F), stride S, surrounding padding P
        e.g.,P = (F-1)/2 to preserve input shape
    
        output_shape = (
        W_out = (W_in - F + 2P)/S + 1, # integer division
        H_out = (H_in - F + 2P)/S + 1, # integer division
        D_out = K
        )
    
        '''

        input = self.input
        x_in = np.zeros(input_shape, dtype=np.float32)
        output_shape = tuple(self.output.shape.eval({input:x_in}))
        del x_in
        
        return output_shape
        
    def print_shape(self):
        
        print 'Layer %s \t in %s --> out %s' % (self.name, 
                                self.input_shape, self.output_shape)
    

class Crop(Layer):
    def __init__(self, input, input_shape, output_shape, 
                 flag_rand=True, flag_batch=True, printinfo=True):
        
        self.get_input_shape(input,input_shape)
        
        
        if flag_rand:
                
            import theano.sandbox.rng_mrg as RNG_MRG
            MRG = RNG_MRG.MRG_RandomStreams(rng.randint(234))
        
            rand = MRG.uniform(size=(3,), low=0.001, high=0.999)
            #rand = MRG.normal(size=(num_sam, self.dim_sample), avg=0., std=1.)
        
        else:
            
            rand = np.float32([0.5, 0.5, 0])
        
        
        if flag_batch:
    
            mirror = self.input[:,:,::-1,:] # TODO find out if it's height or width that is being mirrored
            self.input = T.concatenate([self.input, mirror], axis=0)
            
            cropsize = output_shape[2] # works for both c01b and bc01
    
            # crop images
            center_margin = (self.input_shape[2] - cropsize) / 2
            if flag_rand:
                mirror_rand = T.cast(rand[2], 'int32')
                crop_xs = T.cast(rand[0] * center_margin * 2, 'int32')
                crop_ys = T.cast(rand[1] * center_margin * 2, 'int32')
            else:
                mirror_rand = 0
                crop_xs = center_margin
                crop_ys = center_margin
    
            self.output = self.input[mirror_rand * 3:(mirror_rand + 1) * 3, :, :, :]
            self.output = self.output[ :, crop_xs:crop_xs + cropsize, 
                                          crop_ys:crop_ys + cropsize, :]
    
        else:
        
            raise NotImplementedError()
                
            
        if output_shape:
            self.output_shape = output_shape 
        else:
            self.output_shape = self.get_output_shape(self.input_shape)
            
        self.name = 'Crop\t'
        if printinfo: self.print_shape()

class Conv(Layer):
    def __init__(self, input, convstride, padsize, 
                 b, W = None, filter_shape = None, 
                 lib_conv='cudnn', printinfo=True, 
                 input_shape=None, output_shape=None):
        
        if W == None and filter_shape == None:
            raise AttributeError('need to specify at least one of W and filtershape')
        
        self.get_input_shape(input,input_shape)
         
        self.filter_shape = filter_shape
        self.convstride = convstride
        self.padsize = padsize
        self.lib_conv = lib_conv

        if W:
            self.W = W #Weight(self.filter_shape,)
        else:
            self.W = Normal(filter_shape, mean = 0.0, std=0.1)
            
        self.b = b #Weight(self.filter_shape[3])
        
        if filter_shape:
            assert W.shape == filter_shape

        conv_out = dnn.dnn_conv(img=self.input, # bc01
                                kerns=self.W.val, #bc01
                                subsample=(convstride, convstride),
                                border_mode=padsize,
                                )
        conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x') # broadcasting b     
        
        # ReLu
        self.output = T.maximum(conv_out, 0)

        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']
        if output_shape:
            self.output_shape = output_shape 
        else:
            self.output_shape = self.get_output_shape(self.input_shape)
        
        self.name = 'Conv ({})'.format(lib_conv)
        if printinfo: self.print_shape()   
                                             
    
        
class Pool(Layer):
    def __init__(self, input, poolsize, poolstride, 
                 poolpad, mode = 'max', printinfo=True,
                 input_shape=None,output_shape=None):                 
        
        self.get_input_shape(input,input_shape)
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.poolpad = poolpad
        
        if self.poolsize != 1:
            self.output = dnn.dnn_pool(self.input,
                                       ws=(poolsize, poolsize),
                                       stride=(poolstride, poolstride),
                                       mode=mode, pad=(poolpad, poolpad))
        else:
            self.output = input
        
        if output_shape:
            self.output_shape = output_shape 
        else:
            self.output_shape = self.get_output_shape(self.input_shape)
        
        self.name = 'Pool\t'
        if printinfo: self.print_shape()


class ConvPoolLRN(Layer):

    def __init__(self, input, convstride, padsize, poolsize, poolstride, group,
                 b, W = None, filter_shape = None, 
                 poolpad=0, mode = 'max', 
                 lrn=False, lib_conv='cudnn', printinfo=True,
                 input_shape=None, output_shape=None,
                 ):
                 
                
        '''
                 ConvPoolLRN layer
                 
        To be used in AlexNet
        lib_conv can be cudnn (recommended)or cudaconvnet
        
        '''
                 
        if W == None and filter_shape == None:
            raise AttributeError('need to specify at least one of W and filtershape')
        
        self.get_input_shape(input,input_shape)
        
        self.filter_shape = np.asarray(filter_shape)
        self.convstride = convstride
        self.padsize = padsize
        self.lib_conv = lib_conv
        
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.poolpad = poolpad
        self.lrn = lrn
        if self.lrn:
            self.lrn_func = CrossChannelNormalization()
        
        
        assert group in [1, 2]
        
        
        if group == 1:
            
            self.W = Normal(self.filter_shape, mean=0, std=0.01)
            self.b = Constant(self.filter_shape[3], val=b)
        
        else:
            
            self.filter_shape[0] = self.filter_shape[0] / 2
            self.filter_shape[3] = self.filter_shape[3] / 2
            # self.input_shape[0] = self.input_shape[0] / 2
            # self.input_shape[3] = self.input_shape[3] / 2
            channel = self.input_shape[0]
            self.W0 = Normal(self.filter_shape, mean=0, std=0.01)
            self.W1 = Normal(self.filter_shape, mean=0, std=0.01)
            self.b0 = Constant(self.filter_shape[3], val=b)
            self.b1 = Constant(self.filter_shape[3], val=b)                             
                                                

        if lib_conv == 'cudnn':
            

            input_shuffled = self.input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
            
            # in01out to outin01
            # print image_shape_shuffled
            # print filter_shape_shuffled
            if group == 1:
                W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out = dnn.dnn_conv(img=input_shuffled,
                                        kerns=W_shuffled,
                                        subsample=(convstride, convstride),
                                        border_mode=padsize,
                                        )
                conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')
            else:
                W0_shuffled = \
                    self.W0.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                    
                # print W0_shuffled.shape.eval()# c01b to bc01  # 96, 5, 5, 256 -> 128, 48, 5, 5
                #
                # x_in = np.zeros((96, 27, 27, 128), dtype=np.float32) # c01b to bc01  # 96, 27, 27, 128 -> 128, 48, 27, 27
                # test = input_shuffled[:, :self.channel / 2,:, :]
                #
                # print test.shape
                    
                conv_out0 = \
                    dnn.dnn_conv(img=input_shuffled[:, :channel/2,
                                                    :, :],
                                 kerns=W0_shuffled,
                                 subsample=(convstride, convstride),
                                 border_mode=padsize,
                                 )
                conv_out0 = conv_out0 + \
                    self.b0.val.dimshuffle('x', 0, 'x', 'x')
                W1_shuffled = \
                    self.W1.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out1 = \
                    dnn.dnn_conv(img=input_shuffled[:, channel/2:,
                                                    :, :],
                                 kerns=W1_shuffled,
                                 subsample=(convstride, convstride),
                                 border_mode=padsize,
                                 )
                conv_out1 = conv_out1 + \
                    self.b1.val.dimshuffle('x', 0, 'x', 'x')
                conv_out = T.concatenate([conv_out0, conv_out1], axis=1)

            # ReLu
            self.output = T.maximum(conv_out, 0)

            # Pooling
            if poolsize != 1:
                self.output = dnn.dnn_pool(self.output,
                                           ws=(poolsize, poolsize),
                                           stride=(poolstride, poolstride))

            self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            
        # elif lib_conv == 'cudaconvnet':
#
#             from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
#
#             self.conv_op = FilterActs(pad=self.padsize, stride=self.convstride,
#                                       partial_sum=1)
#
#             from theano.gpuarray.basic_ops import gpu_contiguous
#
#             # Conv
#             if group == 1:
#                 contiguous_input = gpu_contiguous(self.input)
#                 contiguous_filters = gpu_contiguous(self.W.val)
#                 conv_out = self.conv_op(contiguous_input, contiguous_filters)
#                 conv_out = conv_out + self.b.val.dimshuffle(0, 'x', 'x', 'x')
#             else:
#                 contiguous_input0 = gpu_contiguous(
#                     self.input[:channel/2, :, :, :])
#                 contiguous_filters0 = gpu_contiguous(self.W0.val)
#                 conv_out0 = self.conv_op(
#                     contiguous_input0, contiguous_filters0)
#                 conv_out0 = conv_out0 + \
#                     self.b0.val.dimshuffle(0, 'x', 'x', 'x')
#
#                 contiguous_input1 = gpu_contiguous(
#                     self.input[channel/2:, :, :, :])
#                 contiguous_filters1 = gpu_contiguous(self.W1.val)
#                 conv_out1 = self.conv_op(
#                     contiguous_input1, contiguous_filters1)
#                 conv_out1 = conv_out1 + \
#                     self.b1.val.dimshuffle(0, 'x', 'x', 'x')
#                 conv_out = T.concatenate([conv_out0, conv_out1], axis=0)
#
#             # ReLu
#             conv_out = gpu_contiguous(conv_out)
#             self.output = T.maximum(conv_out, 0)
#
#             # Pooling
#             if poolsize != 1:
#                 from pylearn2.sandbox.cuda_convnet.pool import MaxPool
#                 self.pool_op = MaxPool(ds=poolsize, stride=poolstride)
#                 self.output = self.pool_op(self.output)

        elif lib_conv == 'corrmm':

            from theano.gpuarray.basic_ops import gpu_contiguous
            from theano.gpuarray.blas import GpuCorrMM

            border_mode = 'half' if padsize == (filter_shape[1]-1)/2 else (padsize, padsize)
            self.corr_mm_op = GpuCorrMM(subsample=(convstride,convstride),
                                                border_mode=border_mode)

            input_shuffled = self.input.dimshuffle(3, 0, 1, 2)  # c01b to bc01


            if group==1:

                filters = self.W.val.dimshuffle(3, 0, 1, 2)

                # flip top-down, left-right to compute convolution instead of correlation  
                contiguous_filters = gpu_contiguous(filters[:, :, ::-1, ::-1])
                contiguous_input = gpu_contiguous(input_shuffled)

                conv_out = self.corr_mm_op(contiguous_input, contiguous_filters)
                conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')

            else:

                W0_shuffled = self.W0.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01

                contiguous_filters0 = gpu_contiguous(W0_shuffled[:, :, ::-1, ::-1])
                contiguous_input0 = gpu_contiguous(input_shuffled[:, :channel / 2,:, :])

                conv_out0 = self.corr_mm_op(contiguous_input0, contiguous_filters0)
                conv_out0 = conv_out0 + self.b0.val.dimshuffle('x', 0, 'x', 'x')

                W1_shuffled = self.W1.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01

                contiguous_filters1 = gpu_contiguous(W1_shuffled[:, :, ::-1, ::-1])
                contiguous_input1 = gpu_contiguous(input_shuffled[:, channel / 2:,:, :])

                conv_out1 = self.corr_mm_op(contiguous_input1, contiguous_filters1)
                conv_out1 = conv_out1 + self.b1.val.dimshuffle('x', 0, 'x', 'x')
                conv_out = T.concatenate([conv_out0, conv_out1], axis=1)

            # ReLu
            self.output = T.maximum(conv_out, 0)
   

            # Pooling
            if poolsize != 1:
                from theano.gpuarray.pool import GpuPool
                
                
                ds_op = GpuPool(ignore_border=False, mode='max', ndim=2)
                
                self.output = ds_op(inp=self.output, ws=(poolsize,poolsize),
                                    stride=(poolstride,poolstride), pad=(0,0))
                
            self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b

        else:
            NotImplementedError("lib_conv can only be cudnn or cudaconvnet for now")

        # LRN
        if self.lrn:
            # lrn_input = gpu_contiguous(self.output)
            self.output = self.lrn_func(self.output)

        if group == 1:
            self.params = [self.W.val, self.b.val]
            self.weight_type = ['W', 'b']
        else:
            self.params = [self.W0.val, self.b0.val, self.W1.val, self.b1.val]
            self.weight_type = ['W', 'b', 'W', 'b']

        if output_shape:
            self.output_shape = output_shape 
        else:
            self.output_shape = self.get_output_shape(self.input_shape)
        
        self.name = 'ConvPoolLRN(%s) ' % lib_conv
        if printinfo: self.print_shape()                           
                                                         

class BatchNormal(object): #TODO

    def __init__(self):
        pass

class CrossChannelNormalization(object):
    """
    See "ImageNet Classification with Deep Convolutional Neural Networks"
    Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton
    NIPS 2012
    Section 3.3, Local Response Normalization
    .. todo::
        WRITEME properly
    f(c01b)_[i,j,k,l] = c01b[i,j,k,l] / scale[i,j,k,l]
    scale[i,j,k,l] = (k + sqr(c01b)[clip(i-n/2):clip(i+n/2),j,k,l].sum())^beta
    clip(i) = T.clip(i, 0, c01b.shape[0]-1)
    
    reproduced from https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py to remove pylearn2 dependency

    """

    def __init__(self, alpha = 1e-4, k=2, beta=0.75, n=5):
        self.__dict__.update(locals())
        del self.self

        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n for now")

    def __call__(self, c01b):
        """
        .. todo::
            WRITEME
        """
        half = self.n // 2

        sq = T.sqr(c01b)

        ch, r, c, b = c01b.shape

        extra_channels = T.alloc(0., ch + 2*half, r, c, b)

        sq = T.set_subtensor(extra_channels[half:half+ch,:,:,:], sq)

        scale = self.k

        for i in xrange(self.n):
            scale += self.alpha * sq[i:i+ch,:,:,:]

        scale = scale ** self.beta

        return c01b / scale
        
class LRN(Layer):
    
    def __init__(self, input, input_shape=None, printinfo=True):
        
        self.lrn_func = CrossChannelNormalization()
            
        self.get_input_shape(input,input_shape)
            
        self.output = self.lrn_func(self.input)
        self.output_shape = self.input_shape
        self.name = 'LRN\t'
        if printinfo: self.print_shape()
        

class Dimshuffle(Layer):
    
    def __init__(self, input, new_axis_order, 
                 printinfo=True, 
                 input_shape=None, output_shape=None):
        
        self.get_input_shape(input,input_shape)
        
        self.output = self.input.dimshuffle(*new_axis_order)
        
        if output_shape:
            self.output_shape = output_shape 
        else:
            self.output_shape = self.get_output_shape(self.input_shape)
            
        self.name = 'Dimshuffle    '
        if printinfo: self.print_shape()
    

class Flatten(Layer):
    
    def __init__(self, input, axis, 
                 printinfo=True, 
                 input_shape=None, output_shape=None):
        
        self.get_input_shape(input,input_shape)
        
        self.output = T.flatten(self.input, axis)
        
        if output_shape:
            self.output_shape = output_shape 
        else:
            self.output_shape = self.get_output_shape(self.input_shape)
            
        self.name = 'Flatten\t'
        if printinfo: self.print_shape()
        
        
                             
class Dropout(Layer):
    seed_common = np.random.RandomState(0)  # for deterministic results
    # seed_common = np.random.RandomState()
    layers = []

    def __init__(self, input, n_out, 
                 prob_drop=0.5, 
                 printinfo=True, input_shape=None):
        
        
        self.get_input_shape(input,input_shape)
        n_in = self.input_shape[-1]
            
        self.prob_drop = prob_drop
        self.prob_keep = 1.0 - prob_drop
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on

        seed_this = Dropout.seed_common.randint(0, 2**31-1)
        mask_rng = theano.tensor.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=self.input.shape)

        self.output = \
            self.flag_on * T.cast(self.mask, theano.config.floatX) * self.input + \
            self.flag_off * self.prob_keep * self.input

        Dropout.layers.append(self)

        self.output_shape = self.get_output_shape(self.input_shape)
        
        #print 'dropout layer with P_drop: ' + str(self.prob_drop)

        self.name = 'Dropout'+ str(self.prob_drop)
        if printinfo: self.print_shape()

    @staticmethod
    def SetDropoutOn():
        for i in range(0, len(Dropout.layers)):
            Dropout.layers[i].flag_on.set_value(1.0)

    @staticmethod
    def SetDropoutOff():
        for i in range(0, len(Dropout.layers)):
            Dropout.layers[i].flag_on.set_value(0.0)   
            
            

class FC(Layer):

    def __init__(self, input, n_out, W, b,
                 printinfo=True, input_shape=None):
        
        self.get_input_shape(input,input_shape)
        n_in = self.input_shape[-1]
        
        if W and b:
            self.W = W
            self.b = b
        else:
            self.W = Normal((n_in, n_out),std=0.005)
            self.b = Constant((n_out,), val=0.1)

        lin_output = T.dot(self.input, self.W.val) + self.b.val
        #ReLU
        self.output = T.maximum(lin_output, 0)
        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']
        self.output_shape = self.get_output_shape(self.input_shape)
        
        self.name = 'FC\t'
        if printinfo: self.print_shape()

class Softmax(Layer):

    def __init__(self, input, n_out, W, b,
                 printinfo=True, input_shape=None):
        
        self.get_input_shape(input,input_shape)
        n_in = self.input_shape[-1]
        
        if W and b:
            self.W = W
            self.b = b
        else:
            self.W = Normal((n_in, n_out))
            self.b = Constant((n_out,), val=0)

        self.p_y_given_x = T.nnet.softmax(
            T.dot(self.input, self.W.val) + self.b.val)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        self.output = self.p_y_given_x

        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        self.output_shape = self.get_output_shape(self.input_shape)
        
        self.name = 'Softmax\t'
        if printinfo: self.print_shape()

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def errors_top_x(self, y, num_top=5):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))                            
                                    
        if num_top != 5: print 'val errors from top %d' % num_top ############TOP 5 VERSION##########        
        
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_pred_top_x = T.argsort(self.p_y_given_x, axis=1)[:, -num_top:]
            y_top_x = y.reshape((y.shape[0], 1)).repeat(num_top, axis=1)
            return T.mean(T.min(T.neq(y_pred_top_x, y_top_x).astype('int8'), axis=1))
        else:
            raise NotImplementedError()
        
    
def get_layers(lastlayer):
    
    # if multi-stream layers exist, redefine and abstract into one layer class in layers2.py
    
    assert hasattr(lastlayer, 'output')

    layers = [lastlayer]

    while hasattr(lastlayer, 'input_layer'):
    
        lastlayer = lastlayer.input_layer

        layers.append(lastlayer)
    layers = layers[::-1]

    return layers
    
def get_params(layers):
    
    #layers = [layer for layer in layers if layer.name not in ['LRN\t','Pool\t','Flatten\t','Dropout'+ str(0.5)]]
    
    params = []
    weight_types = []
    for layer in layers:
        if hasattr(layer,'params'):
            params+=layer.params
            assert hasattr(layer,'weight_type')
            weight_types+=layer.weight_type
            
    return params, weight_types
    
    
def count_params(params, verbose):
        
    model_size=0
    for param in params:
        
        model_size+=param.size.eval()
        
        if verbose: print param.shape.eval()
        
    if verbose: print 'model size %d M floats' % (int(model_size)/(1024*1024))
    
    
            
        

