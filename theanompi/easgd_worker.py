from __future__ import absolute_import

from theanompi.lib.base import MPI_GPU_Process

worker_alpha = 0.5

class EASGD_Worker(MPI_GPU_Process):
    
    def __init__(self, device):
        MPI_GPU_Process.__init__(self, device) # setup ctx, comm
        
        self.server_rank=0
        
        import os
        self.worker_id = os.getpid()
        self.worker_rank = self.rank
        
        self.register_worker() # setup gpucomm between this worker and the server
        
    def comm_request(self, message):
        
        if self.comm == None:
            
            print 'Worker communicator not initialized'
            
            return
            
            
        request = {'id': self.worker_id, 'rank': self.rank, 'message':message }
        
        self.comm.send(request, dest=self.server_rank, tag=199)
        
        reply = self.comm.recv(source=self.server_rank, tag=200)
        
        return reply

        
    def comm_action(self, message, action=None, action_args=None):
        
        if self.comm == None:
            
            print 'MPIClient not initialized'
            
            return
        
        request = {'id': self.worker_id, 'rank': self.rank, 'message': message }
        
        self.comm.send(request, dest=self.server_rank, tag=199)
        
        reply = self.comm.recv(source=self.server_rank, tag=200)
        
        if action: 
            if action_args:
                action(action_args)
            else:
                action()
        
        
    def register_worker(self):
        
        first = self.comm_request('sync_register')
        
        self.verbose = (first == 'first')
        
        self.gpucomm = self.get_intranode_pair_comm(pair=(0,self.worker_rank))
    
    def exchange(self):
        
        self.exchanger.gpucomm = self.gpucomm
        
        self.comm_action(message = 'exchange', 
                         action=self.exchanger.exchange, 
                         action_args=self.recorder)
        
    def copy_to_local(self):
        
        self.exchanger.gpucomm = self.gpucomm
        
        self.comm_action(message = 'copy_to_local', 
                    action=self.exchanger.copy_to_local)
                    
        #if self.verbose: print '\nSynchronized param with server'
        
    def test_run(self, model):
        
        model.train_iter(0, self.recorder)
        
        model.train_iter(1, self.recorder)
        
        self.comm_request(dict(done=2))
        
        self.exchange()
        
        self.copy_to_local()
        
        if self.verbose: self.comm_request('stop')

        print 'success'
        
        
    def build(self, model, config):
        
        from theanompi.lib.helper_funcs import check_model
        
        # check model has necessary attributes
        check_model(model)
        
        # construct model train function based on sync rule
        model.compile_iter_fns()
        
        from theanompi.lib.recorder import Recorder
        self.recorder = Recorder(self.comm, printFreq=40, 
                                 modelname=model.name, verbose=self.verbose)
        
        # choose the type of exchanger
        from theanompi.lib.exchanger import EASGD_Exchanger
        self.exchanger = EASGD_Exchanger(alpha=worker_alpha, 
                                         param_list=model.params, 
                                         etype='worker')
                
    def run(self, model):
        
        exchange_freq = 10 # iterations
        snapshot_freq = 2 # epochs
        snapshot_path = './snapshots/'
        recorder=self.recorder
        exchanger=self.exchanger
        epoch_start = False
        batch_i=0
        uepoch=0
        
        from theanompi.lib.helper_funcs import save_model
        
        
        while True:
            
            mode = self.comm_request('next')
            
            if mode == 'train':
                
                if epoch_start == False:
                    recorder.start_epoch()
                    epoch_start = True
                    
                    
                for i in range(exchange_freq):
                    
                    model.train_iter(batch_i, recorder)
                    batch_i+=1
                    recorder.print_train_info(batch_i)
                    
                self.comm_request(dict(done=exchange_freq))

                self.exchange()
                
            elif mode == 'adjust_hyperp':
                
                model.adjust_hyperp(uepoch)
                
                
            elif mode == 'val':
                
                self.copy_to_local()
                
                model.reset_iter('val')
                
                for batch_j in range(model.data.n_batch_val):
        
                    model.val_iter(uepoch, recorder)
                    
                model.reset_iter('val')
                
                recorder.print_val_info(batch_i)
                
                model.current_info = recorder.get_latest_val_info()
                
                if self.verbose: recorder.save(batch_i, model.shared_lr.get_value())
    
                uepoch, n_workers = self.comm_request('uepoch')
                
                model.epoch=uepoch
                
                if self.verbose and uepoch % snapshot_freq == 0: 
                    save_model(model, snapshot_path, verbose=self.verbose)
                    
                self.copy_to_local()
                
                
                if epoch_start == True:
                    recorder.end_epoch(batch_i, uepoch)
                    epoch_start = False
                    
            elif mode=='stop':
                
                self.copy_to_local()
                
                for batch_j in range(model.data.n_batch_val):
        
                    model.val_iter(uepoch, recorder)
                    
                model.reset_iter('val')
                
                recorder.print_val_info(batch_i)
                
                if epoch_start == True:
                    recorder.end_epoch(batch_i, uepoch)
                    epoch_start = False
                    
                break
                            
        model.cleanup()

        
if __name__ == '__main__':
    
    import sys
    device = sys.argv[1]
    modelfile = sys.argv[2]
    modelclass = sys.argv[3]
    
    worker = EASGD_Worker(device)
    
    config={}
    config['verbose'] = worker.verbose
    config['rank'] = worker.rank
    config['size'] = 1
    
    import importlib
    mod = importlib.import_module(modelfile)
    modcls = getattr(mod, modelclass)

    model = modcls(config)

    worker.build(model, config)
    
    worker.run(model)