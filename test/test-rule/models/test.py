# setting up device
import os
if 'THEANO_FLAGS' in os.environ:
    raise ValueError('Use theanorc to set the theano config')
os.environ['THEANO_FLAGS'] = 'device={0}'.format('cuda1')
import theano.gpuarray
# This is a bit of black magic that may stop working in future
# theano releases
ctx = theano.gpuarray.type.get_context(None)


# test Crop Layer

import theano.tensor as T
from layers2 import Crop
x = T.ftensor4('x')

crop_l = Crop(input=x,
              input_shape=(3, 
                           4,
                           4,
                           2),
              output_shape=(3, 
                            2,
                            2,
                            2),
              printinfo = True,
              
              flag_batch=False
              )

import numpy as np              
x_in_b1 = np.array([
                [1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]
                            ]).astype('float32')
x_in_b2 = np.array([
                [101,102,103,104],
                [105,106,107,108],
                [109,110,111,112],
                [113,114,115,116]
                            ]).astype('float32')
                            
x_in1 = np.array([[x_in_b1,x_in_b1,x_in_b1]]) # shape = (1,3,4,4)

x_in1= np.rollaxis(x_in1, 0, 4 )

x_in2 = np.array([[x_in_b2,x_in_b2,x_in_b2]]) # shape = (1,3,4,4)

x_in2= np.rollaxis(x_in2, 0, 4 )

x_in = np.concatenate((x_in1,x_in2),axis=3)

out=crop_l.output.eval({x:x_in})

print out[0,:,:,:]

print '================'

out=crop_l.output.eval({x:x_in})

print out[0,:,:,:]

print '================'

out=crop_l.output.eval({x:x_in})

print out[0,:,:,:]

print '================'

out=crop_l.output.eval({x:x_in})

print out[0,:,:,:]

print '================'

out=crop_l.output.eval({x:x_in})

print out[0,:,:,:]
