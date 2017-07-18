from __future__ import absolute_import

from theanompi.lib.base import MPI_GPU_Process

class BSP_Worker(MPI_GPU_Process):
    
    def __init__(self, device, sync_type, exch_strategy):
        MPI_GPU_Process.__init__(self, device) # setup ctx, comm
        
        self.get_intranode_comm() # setup gpucomm
        
        self.sync_type = sync_type
        
        self.exch_strategy = exch_strategy 
        
        self.verbose = (self.rank==0)
        
    def build(self, model, config):
        
        from theanompi.lib.helper_funcs import check_model, check_model_cdd
        
        # check model has necessary attributes
        check_model(model)
        # construct model train function based on sync rule
        model.compile_iter_fns(self.sync_type)
        
        from theanompi.lib.recorder import Recorder
        self.recorder = Recorder(self.comm, printFreq=40, modelname=config['mname'], verbose=self.verbose)
        
        # choose the type of exchanger
        from theanompi.lib.exchanger import BSP_Exchanger
        self.exchanger = BSP_Exchanger(self.comm, self.gpucomm, self.exch_strategy, self.sync_type, self.ctx, model)
    
    def lr_warmup(self, model, epoch):  
        
        if epoch == 0:
            
            self.warmup_epochs=5.
        
            self.power_base = pow(self.size, 1./self.warmup_epochs)  # power(b,5) = size
            
            # epoch0 : lr
            # epoch1 : lr * b
            # epoch2 : lr * b * b 
            # epoch3 : lr * b * b * b 
            # epoch4 : lr * b * b * b * b
            # epoch5 : lr * b * b * b * b * b = lr * size
            # epoch6 : lr * size
            
        else:
            
            if epoch<=self.warmup_epochs:
                
                current_lr = model.shared_lr.get_value()
                
                if self.verbose: print('warming up lr from %f to %f' % (current_lr, current_lr*self.power_base))
                import numpy as np
                model.shared_lr.set_value(np.array(current_lr*self.power_base, dtype='float32'))
                
            
    def BSP_run(self, model):
        
        self.comm.Barrier()
        
        exchange_freq = 1 # iterations
        snapshot_freq = 5 # epochs
        snapshot_path = './snapshots/'
        recorder=self.recorder
        exchanger=self.exchanger
        self.stop = False
        
        from theanompi.lib.helper_funcs import save_model

        for epoch in range(model.n_epochs):
            
            model.epoch=epoch
            
            recorder.start_epoch()
            
            self.lr_warmup(model,epoch)
            
            # train
            exch_iteration=0
            batch_i=0
            while batch_i <model.data.n_batch_train:
                
                for subb_i in range(model.n_subb):
        
                    model.train_iter(batch_i, recorder)
                    
                    if exch_iteration % exchange_freq == 0: 
                        exchanger.exchange(recorder)
                    exch_iteration+=1
                    
                batch_i+=1
                
                recorder.print_train_info(batch_i*self.size)
            
            recorder.clear_train_info()
            
            model.reset_iter('train')
            
        
            # val
            
            self.comm.Barrier()
            
            batch_j=0
            while batch_j <model.data.n_batch_val:
            # for batch_j in range(model.data.n_batch_val):
                
                for subb_i in range(model.n_subb):
                    
                    out = model.val_iter(batch_i*self.size, recorder)
                    
                    if out=='stop': 
                        self.stop=True
                        break
                    elif out !=None:
                        batch_j=out
                    else:
                        batch_j+=1
                
            model.reset_iter('val')
            
            recorder.gather_val_info()
        
            recorder.print_val_info(batch_i*self.size)
            model.current_info = recorder.get_latest_val_info()
            
            if self.rank==0: recorder.save(batch_i*self.size, 0)
            
            if epoch % snapshot_freq == 0 and self.rank==0: save_model(model, snapshot_path, verbose=self.verbose)
            
            model.adjust_hyperp(epoch)
            
            if hasattr(model,'print_info'):
                model.print_info(recorder, verbose=self.verbose)
            
            recorder.end_epoch(batch_i*self.size, epoch)
            
            if self.stop==True:
                break
            
        model.cleanup()

        
if __name__ == '__main__':
    
    import sys
    device = sys.argv[1]
    sync_type = sys.argv[2]
    exch_strategy = sys.argv[3]
    modelfile = sys.argv[4]
    modelclass = sys.argv[5]
    try:
        cpulist = sys.argv[6]
    except:
        pass
    else: # optional binding cores using hwloc
        from theanompi.lib.hwloc_utils import bind_to_socket_mem,detect_socket_num
        bind_to_socket_mem(cpulist, label='train')
        detect_socket_num(debug=True, label='train')
    
    worker = BSP_Worker(device, sync_type, exch_strategy)
    
    config={}
    config['verbose'] = (worker.rank==0)
    config['rank'] = worker.rank
    config['size'] = worker.size
    config['mname'] = modelclass
    
    import importlib
    mod = importlib.import_module(modelfile)
    modcls = getattr(mod, modelclass)

    model = modcls(config)

    worker.build(model, config)
    
    worker.BSP_run(model)