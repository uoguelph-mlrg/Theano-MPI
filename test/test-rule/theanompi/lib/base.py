from __future__ import absolute_import

class MPI_GPU_Process(object):
    
    def __init__(self, device):
        
        self.device = device
        
        self.get_internode_comm()
        
        self.init_device()
        
    def get_internode_comm(self):
        
        from mpi4py import MPI
        self.comm=MPI.COMM_WORLD
        
        self.rank=self.comm.rank
        self.size=self.comm.size
        
    def init_device(self):
        import os
        if 'THEANO_FLAGS' in os.environ:
            raise ValueError('Use theanorc to set the theano config')
        os.environ['THEANO_FLAGS'] = 'device={0}'.format(self.device)
        import theano.gpuarray
        # This is a bit of black magic that may stop working in future
        # theano releases
        self.ctx = theano.gpuarray.type.get_context(None)