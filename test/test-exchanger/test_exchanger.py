import numpy as np
# import time

def init_device(device='gpu0'):
  
    if device.startswith('cuda'):
        
        import os
        if 'THEANO_FLAGS' in os.environ:
            raise ValueError('Use theanorc to set the theano config')
        
        os.environ['THEANO_FLAGS'] = 'device={0}'.format(device)
        import theano.gpuarray
        # This is a bit of black magic that may stop working in future
        # theano releases
        ctx = theano.gpuarray.type.get_context(None)
        drv = None
        
    elif device.startswith('gpu'):
        
        gpuid = int(device[-1])

        import pycuda.driver as drv
        drv.init()
        dev = drv.Device(gpuid)
        ctx = dev.make_context()
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(device)
        import theano
    else:
        drv=None
        ctx=None
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(device)
        import theano
        
    from theano import function, config, shared, sandbox, tensor

    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = np.random.RandomState(22)
    arr = rng.rand(vlen)

    shared_x = theano.shared(np.asarray(arr, config.floatX))
    shared_xx = theano.shared(np.asarray(arr, config.floatX))
    
    x=tensor.fvector("x")
    # compile a function so that shared_x will be set to part of a computing graph on GPU (CUDAndarray)
    f = function([], tensor.exp(x), givens=[(x,shared_x)]) 
    
    
    if np.any([isinstance(x.op, tensor.Elemwise) and
                  ('Gpu' not in type(x.op).__name__)
                  for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')

    # if np.any([isinstance(x.op, tensor.Elemwise) for x in f.maker.fgraph.toposort()]) and device!='cpu':
    #     raise TypeError('graph not compiled on GPU') 

    return drv,ctx, arr, shared_x, shared_xx
    
    
    
def clean_device(ctx):
    ctx.pop()