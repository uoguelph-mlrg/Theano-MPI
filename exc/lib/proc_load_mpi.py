'''
Load data in parallel with train.py
'''

import time
import math

import numpy as np
import zmq
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import hickle as hkl
from helper_funcs import get_rand3d

def get_params_crop_and_mirror(param_rand, data_shape, cropsize):

    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = round(param_rand[0] * center_margin * 2)
    crop_ys = round(param_rand[1] * center_margin * 2)
    if False:
        # this is true then exactly replicate Ryan's code, in the batch case
        crop_xs = math.floor(param_rand[0] * center_margin * 2)
        crop_ys = math.floor(param_rand[1] * center_margin * 2)

    flag_mirror = bool(round(param_rand[2]))

    return crop_xs, crop_ys, flag_mirror


def crop_and_mirror(data, param_rand, flag_batch=True, cropsize=227):
    '''
    when param_rand == (0.5, 0.5, 0), it means no randomness
    '''
    # print param_rand

    # if param_rand == (0.5, 0.5, 0), means no randomness and do validation
    if param_rand[0] == 0.5 and param_rand[1] == 0.5 and param_rand[2] == 0:
        flag_batch = True

    if flag_batch:
        # mirror and crop the whole batch
        crop_xs, crop_ys, flag_mirror = \
            get_params_crop_and_mirror(param_rand, data.shape, cropsize)

        # random mirror
        if flag_mirror:
            data = data[:, :, ::-1, :]

        # random crop
        data = data[:, crop_xs:crop_xs + cropsize,
                    crop_ys:crop_ys + cropsize, :]

    else:
        # mirror and crop each batch individually
        # to ensure consistency, use the param_rand[1] as seed
        np.random.seed(int(10000 * param_rand[1]))

        data_out = np.zeros((data.shape[0], cropsize, cropsize,
                             data.shape[3])).astype('float32')

        for ind in range(data.shape[3]):
            # generate random numbers
            tmp_rand = np.float32(np.random.rand(3))
            tmp_rand[2] = round(tmp_rand[2])

            # get mirror/crop parameters
            crop_xs, crop_ys, flag_mirror = \
                get_params_crop_and_mirror(tmp_rand, data.shape, cropsize)

            # do image crop/mirror
            img = data[:, :, :, ind]
            if flag_mirror:
                img = img[:, :, ::-1]
            img = img[:, crop_xs:crop_xs + cropsize,
                      crop_ys:crop_ys + cropsize]
            data_out[:, :, :, ind] = img

        data = data_out

    return np.ascontiguousarray(data, dtype='float32')


if __name__ == '__main__':
    
    from mpi4py import MPI
    
    import sys
    gpuid = sys.argv	

    print gpuid[1]

    icomm = MPI.Comm.Get_parent()
    
    # 0. Receive config
    config = icomm.recv(source=MPI.ANY_SOURCE,tag=99)
    config['icomm']=icomm
    size = config['size']
    rank = config['rank']
    verbose = False #rank==0
    file_batch_size = config['file_batch_size']
    batch_size = config['batch_size']
    subb = file_batch_size//batch_size
    
    
    drv.init()
    dev = drv.Device(int(gpuid[1]))

    ctx = dev.make_context()

    import socket
    addr = socket.gethostbyname(socket.gethostname())
    if verbose: print addr, rank

    sock = zmq.Context().socket(zmq.PAIR)
    try:
        sock.bind('tcp://*:{0}'.format(config['sock_data']))
    except zmq.error.ZMQError:
        print 'rank %d zmq error' % rank
        sock.close()
        zmq.Context().term()
        raise
    finally:
        pass

    shape, dtype, h = sock.recv_pyobj()
    if verbose: print '[load] 1. shared_x information received'

    gpu_data_remote = gpuarray.GPUArray(shape, dtype,
                                        gpudata=drv.IPCMemoryHandle(h))
    gpu_data = gpuarray.GPUArray(shape, dtype)

    img_mean = icomm.recv(source=MPI.ANY_SOURCE, tag=66)
    if verbose: print '[load] 2. img_mean received'

    count=0
    import time
    while True:
        
        mode = icomm.recv(source=MPI.ANY_SOURCE, tag=43)
        if verbose: print '[load] 3. mode received: %s' % mode
        
        filename_list = icomm.recv(source=MPI.ANY_SOURCE, tag=40)
        if verbose: print '[load] 4. filename list received'

        for filename in filename_list:

            data = hkl.load(str(filename)) - img_mean
            
            rand_arr = get_rand3d(config)

            data = crop_and_mirror(data, rand_arr, flag_batch=True, cropsize = config['input_width'])

            gpu_data.set(data)
            
            # 5. wait for computation on last minibatch to finish  
            msg = icomm.recv(source=MPI.ANY_SOURCE,tag=35)
            assert msg == 'calc_finished'

            drv.memcpy_dtod(gpu_data_remote.ptr,
                            gpu_data.ptr,
                            gpu_data.dtype.itemsize *
                            gpu_data.size,
                            )

            ctx.synchronize()

            # 6. tell train proc to start train on this batch
            icomm.isend("copy_finished",dest=0,tag=55)
            
    icomm.Disconnect()


