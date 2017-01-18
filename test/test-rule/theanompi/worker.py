from __future__ import absolute_import

class Worker(object):
    
    def __init__(self, device, sync_type, exch_strategy):
        
        self.device = device
        
        self.sync_type = sync_type
        
        self.exch_strategy = exch_strategy
        
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
        
        from theanompi.lib.helper_funcs import check_model
        
        # check model has necessary attributes
        check_model(model)
        
        # construct model train function based on sync rule
        model.compile_iter_fns()
        
        self.verbose = (self.rank==0)
        from theanompi.lib.recorder import Recorder
        self.recorder = Recorder(self.comm, printFreq=40, modelname='alexnet', verbose=self.verbose)
        
        # choose the type of exchanger
        from theanompi.lib.exchanger import BSP_Exchanger
        self.exchanger = BSP_Exchanger(self.comm, self.gpucomm, self.exch_strategy, self.sync_type, self.ctx, model)
            
            
    def BSP_run(self, model):
        
        self.comm.Barrier()
        
        exchange_freq = 1 # iterations
        snapshot_freq = 2 # epochs
        snapshot_path = './snapshots/'
        recorder=self.recorder
        exchanger=self.exchanger
        
        from theanompi.lib.helper_funcs import save_model

        for epoch in range(model.n_epochs):
            
            model.epoch=epoch
            
            recorder.start_epoch()
            
            # train
    
            for batch_i in range(model.data.n_batch_train):
        
                model.train_iter(batch_i, recorder)
                
                if batch_i % exchange_freq == 0: 
                    exchanger.exchange(recorder)
        
                recorder.print_train_info(batch_i * self.size)
            
            model.reset_iter('train')
        
            # val
            
            self.comm.Barrier()
    
            for batch_j in range(model.data.n_batch_val):
        
                model.val_iter(batch_i, recorder)

                
            model.reset_iter('val')
            
            recorder.gather_val_info()
        
            recorder.print_val_info(batch_i)
            model.current_info = recorder.get_latest_val_info()
            
            if self.rank==0: recorder.save(batch_i, model.shared_lr.get_value())
            
            if epoch % snapshot_freq == 0 and self.rank==0: save_model(model, snapshot_path, verbose=self.verbose)
            
            model.adjust_hyperp(epoch)
            
            recorder.end_epoch(batch_i, epoch)
            
        model.cleanup()

        
if __name__ == '__main__':
    
    import sys
    device = sys.argv[1]
    sync_type = sys.argv[2]
    exch_strategy = sys.argv[3]
    modelfile = sys.argv[4]
    modelclass = sys.argv[5]
    
    worker = Worker(device, sync_type, exch_strategy)
    
    config={}
    config['verbose'] = (worker.rank==0)
    config['rank'] = worker.rank
    config['size'] = worker.size
    
    import importlib
    mod = importlib.import_module(modelfile)
    modcls = getattr(mod, modelclass)

    model = modcls(config)

    worker.build(model, config)
    
    worker.BSP_run(model)