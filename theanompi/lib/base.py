from __future__ import absolute_import

from mpi4py import MPI

class MPI_GPU_Process(object):
    
    def __init__(self, device):
        
        self.device = device
        
        self.get_internode_comm()
        
        self.init_device()
        
    def get_internode_comm(self):
        
        self.comm=MPI.COMM_WORLD
        
        self.rank=self.comm.rank
        self.size=self.comm.size
        
    def get_intranode_comm(self):
        
        '''a gpucomm between all synchronous workers'''
        
        from pygpu import collectives
    
        _local_id = collectives.GpuCommCliqueId(context=self.ctx)

        string =  _local_id.comm_id.decode('utf-8')

        comm=self.comm
        rank=comm.rank
        size=comm.size

        if rank==0:
            _string=string
        else:
            _string=None
        
        _string=comm.bcast(_string, root=0)

        _local_id.comm_id = bytearray(_string.encode('utf-8'))
        _local_size = size # how many intra-node workers, in the case of copper maximum 8 workers per node, assuming running within a node here 
        _local_rank = rank # assuming running within a node here 
 
        self.gpucomm = collectives.GpuComm(_local_id,_local_size,_local_rank)
    
    def get_intranode_pair_comm(self, pair):
        
        '''a gpucomm between the server and a worker'''
        # pair is the a size-two tuple of the MPI ranks of the server (rank=0) and a worker 
    
        from pygpu import collectives
    
        _local_id = collectives.GpuCommCliqueId(context=self.ctx)

        string =  _local_id.comm_id.decode('utf-8')

        comm=self.comm
        rank=comm.rank
        size=comm.size

        # if rank==0:
        #     _string=string
        #     comm.send(_string, dest=1)
        # else:
        #
        #     _string = comm.recv(source=0)
            
            
        if rank==0:
            
            _string = comm.recv(source=MPI.ANY_SOURCE, tag=220)
            
        else:

            _string=string
            comm.send(_string, dest=0, tag=220)
            
        #print _string,  string,  _string==string
            
        # len_pid =len(str(pid))
        #
        # # replace the process-unique id to be the universal id "0......" so that a intranode gpucomm can be created
        #
        # pair_index=0
        #
        # replacement = ''.join(('%d' % pair_index) for i in range(len_pid))
        # _string = string.replace(str(pid), replacement)
    


        _local_id.comm_id = bytearray(_string.encode('utf-8'))
        _local_size = len(pair) # how many intra-node processes, pair usually means 2
    
        if self.rank==pair[0]:
            _local_rank=0
        else:
            _local_rank=1
     
        gpucomm = collectives.GpuComm(_local_id,_local_size,_local_rank)
    
        #print 'on rank %d, pair %s generated' % (self.rank, pair)
        
        return gpucomm
        
    def init_device(self):
        import os
        if 'THEANO_FLAGS' in os.environ:
            raise ValueError('Use theanorc to set the theano config')
        os.environ['THEANO_FLAGS'] = 'device={0}'.format(self.device)
        import theano.gpuarray
        # This is a bit of black magic that may stop working in future
        # theano releases
        self.ctx = theano.gpuarray.type.get_context(None)