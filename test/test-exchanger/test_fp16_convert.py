import pygpu
import numpy as np

device='cuda'+str(0)

from test_exchanger import init_device, clean_device
_,ctx,arr,shared_x,shared_xx = init_device(device=device)



float2half = pygpu.elemwise.GpuElemwise(expr="a = b",
                                             args=[pygpu.elemwise.arg("a", 'float16', write=True),\
                                             pygpu.elemwise.arg("b", 'float32', read=True)],
                                             convert_f16=True,
                                             ctx=ctx)



numpy_float = np.array([1.444, 2.555, 7.999], dtype=np.float32)
numpy_half = np.array([0, 0, 0], dtype=np.float16)

ga_float = pygpu.asarray(numpy_float, dtype=np.float32,
                           context=ctx)
                           
ga_half = pygpu.asarray(numpy_half, dtype=np.float16,
                           context=ctx)

print 'before convert', ga_float

float2half(ga_half, ga_float)

print 'after convert', ga_half
                           

                           
                           
                           
                           