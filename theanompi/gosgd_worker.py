from __future__ import absolute_import

from theanompi.lib.base import MPI_GPU_Process

class GOSGD_Worker(MPI_GPU_Process):
    
    '''
    An implementation of a worker process in the Gossip SGD rule
    https://arxiv.org/abs/1611.09726

    '''
    
    def __init__(self, device):
        MPI_GPU_Process.__init__(self, device) # setup ctx, comm
        
        self.get_intranode_comm_dict() # setup D_gpucomm synchronously
        
    def build(self, model, config):
        
        from theanompi.lib.helper_funcs import check_model
        
        # check model has necessary attributes
        check_model(model)
        
        # construct model train function based on sync rule
        model.compile_iter_fns()
        
        self.verbose = (self.rank==0)
        from theanompi.lib.recorder import Recorder
        self.recorder = Recorder(self.comm, printFreq=40, modelname=model.name, verbose=self.verbose)
        
        # choose the type of exchanger
        from theanompi.lib.exchanger import GOSGD_Exchanger
        self.exchanger = GOSGD_Exchanger(self.comm, self.D_gpucomm, self.ctx, model)
            
            
    def run(self, model):
        
        self.comm.Barrier()
        
        exchange_freq = 1 # iterations
        snapshot_freq = 5 # epochs
        snapshot_path = './snapshots/'
        recorder=self.recorder
        exchanger=self.exchanger
        
        from theanompi.lib.helper_funcs import save_model

        for epoch in range(model.n_epochs):
            
            model.epoch=epoch
            
            recorder.start_epoch()
            
            # train
    
            for batch_i in range(model.data.n_batch_train):
                
                exchanger.process_messages(recorder) # process those params and alpha's from other workers (use comm first and then gpucomm )
        
                model.train_iter(batch_i, recorder)
                
                if exchanger.draw()==True: # drawing a Success Bernoulli variable
                    # Choosing another worker from M-1 workers
                    dest_rank = exchanger.choose() 
                    # push self.params and self.alpha
                    exchanger.push_message(dest_rank, recorder)
        
                recorder.print_train_info(batch_i*self.size)
            
            model.reset_iter('train')
        
            # val
            
            self.comm.Barrier()
    
            for batch_j in range(model.data.n_batch_val):
        
                model.val_iter(batch_i, recorder)

                
            model.reset_iter('val')
            
            #recorder.gather_val_info()
        
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
    
    worker = GOSGD_Worker(device)
    
    config={}
    config['verbose'] = True
    config['rank'] = 0
    config['size'] = 1
    
    import importlib
    mod = importlib.import_module(modelfile)
    modcls = getattr(mod, modelclass)

    model = modcls(config)

    worker.build(model, config)
    
    worker.run(model)