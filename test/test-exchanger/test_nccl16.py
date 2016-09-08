import sys
sys.path.append('../../lib/base/')

# inter-node comm

def get_internode_comm():
    
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    
    return comm

# intra-node comm

def  get_intranode_comm(rank,size, ctx):
    
    from pygpu import collectives
    
    _local_id = collectives.GpuCommCliqueId(context=ctx)

    string =  _local_id.comm_id.decode('utf-8')

    import os
    pid = str(os.getpid())
    len_pid =len(pid)

    # replace the process-unique id to be the universal id "0......" so that a intranode gpucomm can be created
    replacement = ''.join('0' for i in range(len_pid))
    _string = string.replace(pid, replacement)
    
    # if rank==0:
    #     comm.send(_string, dest=1)
    # else:
    #     res = comm.recv(source=0)
    #
    #     print res == _string
    #
    # comm.Barrier()

    _local_id.comm_id = bytearray(_string.encode('utf-8'))
    _local_size = size # how many intra-node workers, in the case of copper maximum 8 workers per node, assuming running within a node here 
    _local_rank = rank # assuming running within a node here 
     
    gpucomm = collectives.GpuComm(_local_id,_local_size,_local_rank)
    
    return gpucomm



if __name__ == '__main__':
    
    comm = get_internode_comm()
    
    rank=comm.rank
    device='cuda'+str(rank)
    size=comm.size

    from test_exchanger import init_device, clean_device
    _,ctx,arr,shared_x,shared_xx = init_device(device=device)
    
    gpucomm = get_intranode_comm(rank,size, ctx)
                           

    if rank==0: print 'original array %s' % arr

    # prepare nccl16 exchanger

    from exchanger_strategy import Exch_nccl16

    exch = Exch_nccl16(intercomm=comm, intracomm=gpucomm, avg=False)

    exch.prepare(ctx, [shared_x])

    exch.exchange()

    if rank==0: print 'nccl16 summation: %s' % shared_x.get_value()


    # prepare ar exchanger

    from exchanger_strategy import Exch_allreduce
    exch = Exch_allreduce(comm, avg=False) 

    exch.prepare([shared_xx])

    exch.exchange()

    if rank==0: print 'ar summation: %s' % shared_xx.get_value()

    # clean_device(ctx=ctx)
