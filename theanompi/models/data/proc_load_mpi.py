# This version of proc_load.py is modified based on the theano_alexnet project. See the original project here:
# https://github.com/uoguelph-mlrg/theano_alexnet, and its copy right:
# Copyright (c) 2014, Weiguang Ding, Ruoyan Wang, Fei Mao and Graham Taylor
# All rights reserved.

'''
Load data in parallel for imagenet
'''


import numpy as np
import zmq
import hickle as hkl
import pygpu

if __name__ == '__main__':
    
    from theanompi.models.data.utils import crop_and_mirror
    
    from mpi4py import MPI

    icomm = MPI.Comm.Get_parent()
    
    
    config = icomm.recv(source=MPI.ANY_SOURCE,tag=99)
    
    gpuid = config['gpuid']
    verbose = config['verbose']
    sock_data = config['sock_data']
    
    input_width  =    config['input_width']      
    input_height  =    config['input_height']     
    rand_crop  =    config['rand_crop']        
    batch_crop_mirror  =    config['batch_crop_mirror']
    img_mean = config['img_mean']
    img_std=config['img_std']
    import os
    if "CPULIST_train" in os.environ:
        cpulist = os.environ['CPULIST_train']
        from theanompi.lib.hwloc_utils import bind_to_socket_mem, detect_socket_num
        bind_to_socket_mem(cpulist, label='load')
        detect_socket_num(debug=True, label='load')

    ctx = pygpu.init(gpuid)

    import socket
    addr = socket.gethostbyname(socket.gethostname())

    sock = zmq.Context().socket(zmq.PAIR)
    try:
        sock.bind('tcp://*:{0}'.format(sock_data))
    except zmq.error.ZMQError:
        import os
        print('[load] %s port %d zmq error' % (os.getpid(),sock_data))
        sock.close()
        zmq.Context().term()
        raise
    finally:
        pass

    shape, dtype, h = sock.recv_pyobj()
    if verbose: print('[load] 1. shared_x information received')

    gpu_data_remote_b = pygpu.gpuarray.open_ipc_handle(ctx, h, np.prod(shape)*dtype.itemsize)
    gpu_data_remote = pygpu.gpuarray.from_gpudata(gpu_data_remote_b, 0, dtype, shape, ctx)
    gpu_data = pygpu.empty(shape, dtype, context=ctx)

    # img_mean = icomm.recv(source=MPI.ANY_SOURCE, tag=66)
    # if verbose: print '[load] 2. img_mean received'

    import os
    print('loading %s started' % os.getpid())
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

            arr = hkl.load(str(filename)).astype('float32')
            
            arr = (arr - img_mean)/255./img_std
            
            arr = crop_and_mirror(arr, mode, 
                                rand_crop, 
                                batch_crop_mirror, 
                                input_width)
            # data = np.ascontiguousarray(data)

            gpu_data.write(arr)

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
    if verbose: print('[load] paraloading process closed')


