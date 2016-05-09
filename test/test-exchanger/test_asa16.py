
import sys
sys.path.append('../../lib/base/')
device=sys.argv[1]

from mpi4py import MPI
comm=MPI.COMM_WORLD
rank=comm.rank
size=comm.size
# device='gpu'+str(rank)

from test_exchanger import init_device, clean_device

drv,ctx,arr,shared_x,shared_xx = init_device(device=device)

if rank==0: print 'original array %s' % arr

# prepare asa16 exchanger

from exchanger_strategy import Exch_asa16
exch = Exch_asa16(comm, avg=False)

exch.prepare([shared_x], ctx, drv)
exch.exchange()

if rank==0: print 'asa16 summation: %s' % shared_x.get_value()


# prepare ar exchanger

from exchanger_strategy import Exch_allreduce
exch = Exch_allreduce(comm, avg=False) 

exch.prepare([shared_xx])

exch.exchange()

if rank==0: print 'ar summation: %s' % shared_xx.get_value()

clean_device(ctx=ctx)