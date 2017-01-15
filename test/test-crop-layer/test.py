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
                           1),
              output_shape=(3, 
                            2,
                            2,
                            1),
              printinfo = True
              )
              
x_in = np.array([
                [1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]
                            ]).astype('float32')
                            
x_in = np.array([[x_in,x_in,x_in]]) # shape = (1,3,4,4)

x_in= np.rollaxis(x_in, 0, 4 )

out=crop_l.output.eval({x:x_in})

print out

out=crop_l.output.eval({x:x_in})

print out

out=crop_l.output.eval({x:x_in})

print out

out=crop_l.output.eval({x:x_in})

print out

out=crop_l.output.eval({x:x_in})

print out
