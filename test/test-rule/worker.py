class Worker(object):
    
    def __init__(self, device, sync_type):
        
        self.device = device
        
        self.sync_type = sync_type
        
        self.get_internode_comm()
        
        self.init_device()
        
        self.get_intranode_comm()
        
    def get_internode_comm(self):
        
        from mpi4py import MPI
        self.comm=MPI.COMM_WORLD
        
        self.rank=self.comm.rank
        self.size=self.comm.size
        
    def get_intranode_comm(self):
        
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
        
        
    def init_device(self):
        import os
        if 'THEANO_FLAGS' in os.environ:
            raise ValueError('Use theanorc to set the theano config')
        os.environ['THEANO_FLAGS'] = 'device={0}'.format(self.device)
        import theano.gpuarray
        # This is a bit of black magic that may stop working in future
        # theano releases
        self.ctx = theano.gpuarray.type.get_context(None)
        
    def build(self, model, config):
        
        try:
            
            assert hasattr(model, 'params') == True
            
            assert isinstance(model.params, list)
            
            import theano
            
            assert isinstance(model.params[0], theano.gpuarray.type.GpuArraySharedVariable)
            
            assert hasattr(model, 'data') == True
            
            assert hasattr(model, 'train_iter') == True
            
            assert hasattr(model, 'val_iter') == True
            
            assert hasattr(model, 'compile_train') == True and callable(getattr(model, 'compile_train')) == True
            
            assert hasattr(model, 'compile_val') == True and callable(getattr(model, 'compile_val')) == True
            
            assert hasattr(model, 'adjust_hyperp') == True and callable(getattr(model, 'adjust_hyperp')) == True
            
            assert hasattr(model, 'save') == True and  callable(getattr(model, 'save')) == True
            
            assert hasattr(model, 'load') == True and  callable(getattr(model, 'load')) == True
        
        except AssertionError:
            
            print 'Model def lacks some attributes and/or methods'
            raise
            
        
        # construct model train function based on sync rule
        from lib.opt import pre_model_fn
        pre_model_fn(model, self.sync_type)
        
        # choose between Iterator and Iterator_hkl. 
        # Iterator_hkl supports parallel loading hkl files
        from lib.iterator import Iterator 
        model.train_iter = Iterator(model, self.sync_type, 'train')
        model.val_iter = Iterator(model, self.sync_type, 'val')
        
        from lib.recorder import Recorder
        self.recorder = Recorder(config)
        
        # choose the type of exchanger
        from lib.exchanger import BSP_Exchanger
        self.recorder = BSP_Exchanger(self.comm, self.gpucomm, config['exch_strategy'], self.sync_type, self.ctx, model)

        
    def run(self):
        
        pass
        
if __name__ == '__main__':
    
    import sys
    
    device = sys.argv[1]
    
    sync_type = sys.argv[2]
    
    worker = Worker(device, sync_type)
    
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    
    config['verbose'] = (worker.rank==0)

    from models.cifar10 import Cifar10_model

    model = Cifar10_model(config)

    worker.build(model, config)