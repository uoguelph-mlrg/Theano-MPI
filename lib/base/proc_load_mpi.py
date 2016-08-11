# This version of proc_load.py is modified based on the theano_alexnet project. See the original project here:
# https://github.com/uoguelph-mlrg/theano_alexnet, and its copy right:
# Copyright (c) 2014, Weiguang Ding, Ruoyan Wang, Fei Mao and Graham Taylor
# All rights reserved.

'''
Load data in parallel with train.py
'''

import time
import math

import numpy as np
import zmq
import hickle as hkl
import pygpu
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
        # mirror and crop each batch(each image?) individually
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
    
    verbose = False #rank==0
    
    import sys
    
    gpuid = sys.argv[1]

    if verbose: print gpuid

    icomm = MPI.Comm.Get_parent()

    # 0. Receive config
    config = icomm.recv(source=MPI.ANY_SOURCE,tag=99)
    config['icomm']=icomm
    size = config['size']
    rank = config['rank']

    file_batch_size = config['file_batch_size']
    batch_size = config['batch_size']
    subb = file_batch_size//batch_size

    ctx = pygpu.init(gpuid)

    import socket
    addr = socket.gethostbyname(socket.gethostname())
    if verbose: print '[load] ', addr, rank

    sock = zmq.Context().socket(zmq.PAIR)
    try:
        sock.bind('tcp://*:{0}'.format(config['sock_data']))
    except zmq.error.ZMQError:
        print '[load] rank %d port %d zmq error' % (rank,config['sock_data'])
        sock.close()
        zmq.Context().term()
        raise
    finally:
        pass

    shape, dtype, h = sock.recv_pyobj()
    if verbose: print '[load] 1. shared_x information received'

    gpu_data_remote_b = pygpu.gpuarray.open_ipc_handle(ctx, h, np.prod(shape)*dtype.itemsize)
    gpu_data_remote = pygpu.gpuarray.from_gpudata(gpu_data_remote_b, 0, dtype, shape, ctx)
    gpu_data = pygpu.empty(shape, dtype, context=ctx)

    img_mean = icomm.recv(source=MPI.ANY_SOURCE, tag=66)
    if verbose: print '[load] 2. img_mean received'

    count=0
    mode=None
    import time
    while True:
        
        # 3. load the very first filename in 'train' or 'val' mode
        message = icomm.recv(source=0, tag=40)
        
        if message == 'stop':
            break
        elif message == 'train':
            mode = 'train'
            continue
        elif message == 'val':
            mode = 'val'
            continue
        else:
            filename = message
            
        if mode==None:
            raise ValueError('[load] need to specify a mode (train or val) to proceed')
        
        while True:

            data = hkl.load(str(filename)) - img_mean
        
            rand_arr = get_rand3d(config, mode)

            data = crop_and_mirror(data, rand_arr, \
                                    flag_batch=config['batch_crop_mirror'], \
                                    cropsize = config['input_width'])
                                    

            gpu_data.write(data)

            # 4. wait for computation on last minibatch to
            # finish and get the next filename
            message = icomm.recv(source=0,tag=40)
            
            # this is used when switching to 'val' 
            # or 'stop' during 'train' mode
            if message =='stop': 
                mode = None
                break
            elif message == 'train':
                mode = 'train'
                break
            elif message == 'val':
                mode = 'val'
                break
            else:
                filename = message

            gpu_data_remote[:] = gpu_data

            # 5. tell train proc to start train on this batch
            icomm.isend("copy_finished",dest=0,tag=55)
            
    icomm.Disconnect()
    if verbose: print '[load] paraloading process closed'


