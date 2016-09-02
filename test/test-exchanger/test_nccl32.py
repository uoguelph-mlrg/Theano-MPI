import sys
sys.path.append('../../lib/base/')

# inter-node comm

from mpi4py import MPI
comm=MPI.COMM_WORLD
rank=comm.rank
device='cuda'+str(rank)
size=comm.size

from test_exchanger import init_device, clean_device
_,ctx,arr,shared_x,shared_xx = init_device(device=device)

# intra-node comm
import pygpu
from pygpu import collectives
_local_id = pygpu.collectives.GpuCommCliqueId(context=ctx)
# string =  _local_id.comm_id.decode('utf-8')
#
# _local_id.comm_id = bytearray(string.encode('utf-8'))
_local_size = size%9 # how many intra-node workers, in the case of copper maximum 8 workers per node, assuming running within a node here 
_local_rank = rank # assuming running within a node here 

print '-1'
gpucomm = collectives.GpuComm(_local_id,_local_size,_local_rank)                              

if rank==0: print 'original array %s' % arr

# prepare copper exchanger

from nccl_exch import Exch_nccl32

exch = Exch_nccl32(intercomm=comm, intracomm=gpucomm, avg=False)

exch.prepare(ctx, [shared_x])

exch.exchange()

if rank==0: print 'nccl32 summation: %s' % shared_x.get_value()


# prepare ar exchanger

from exchanger_strategy import Exch_allreduce
exch = Exch_allreduce(comm, avg=False) 

exch.prepare([shared_xx])

exch.exchange()

if rank==0: print 'ar summation: %s' % shared_xx.get_value()

clean_device(ctx=ctx)
