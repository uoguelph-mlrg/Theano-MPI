from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy as np
# import time

def init_device(device='gpu0'):
  
    if device!='cpu':
        
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

    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = np.random.RandomState(22)
    arr = rng.rand(vlen)

    shared_x = theano.shared(np.asarray(arr, config.floatX))
    shared_xx = theano.shared(np.asarray(arr, config.floatX))
    x=T.fvector("x")
    # compile a function so that shared_x will be set to part of a computing graph on GPU (CUDAndarray)
    f = function([], T.exp(x), givens=[(x,shared_x)]) 

    if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]) and device!='cpu':
        raise TypeError('graph not compiled on GPU')  

    return drv,ctx, arr, shared_x, shared_xx
    
    
    
def clean_device(ctx):
    ctx.pop()