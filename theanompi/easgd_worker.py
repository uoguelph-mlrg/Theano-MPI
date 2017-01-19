from __future__ import absolute_import

from theanompi.lib.base import MPI_GPU_Process

class EASGD_Worker(MPI_GPU_Process):
    
    def __init__(self, device):
        MPI_GPU_Process.__init__(self, device)
        
        self.server_rank=0
        
        import os
        self.worker_id = os.getpid()
        
    def get_intranode_pair_comm(self, pair):
    
        from pygpu import collectives
    
        _local_id = collectives.GpuCommCliqueId(context=self.ctx)

        string =  _local_id.comm_id.decode('utf-8')

        pid = str(self.worker_id)
        len_pid =len(pid)

        # replace the process-unique id to be the universal id "0......" so that a intranode gpucomm can be created
        
        pair_index=0

        replacement = ''.join(('%d' % pair_index) for i in range(len_pid))
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
        _local_size = len(pair) # how many intra-node workers, in the case of copper maximum 8 workers per node, assuming running within a node here 
    
        if self.interrank==pair[0]:
            _local_rank=0
        else:
            _local_rank=1
        
        _local_rank = _local_rank # assuming running within a node here 
     
        self.gpucomm = collectives.GpuComm(_local_id,_local_size,_local_rank)
    
        print 'on rank %d, pair %s generated' % (self.interrank, pair)
        
        
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
        
        if action: action(action_args)
        
        
    def register_worker(self):
        
        first = self.comm_request('sync_register')
        
        self.verbose = (first == 'first')
            
    def test_run(self):
        
        self.comm.Barrier()
        
        reply = self.comm_request('stop')

        print 'success', reply
        
        
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
        self.exchanger = EASGD_Exchanger(self.comm, self.gpucomm,
                                         self.ctx, model, etype='worker')
                                         
        #TODO reimplement EASGD using pygpu.bcast
    
    def copy_to_local(self):
        
        self.comm_action(message = 'copy_to_local', \
                    action=self.exchanger.copy_to_local)
        if self.verbose: print '\nSynchronized param with server'
                
    def EASGD_run(self, model):
        
        # after the barrier, run asynchronously
        self.comm.Barrier()
        
        exchange_freq = 1 # iterations
        snapshot_freq = 2 # epochs
        snapshot_path = './snapshots/'
        recorder=self.recorder
        exchanger=self.exchanger
        epoch_start = False
        batch_i=0
        self.uepoch=0
        
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
                    
                self.comm_request(dict(done=self.train_len))

                self.comm_action(message = 'exchange', \
                            action=self.exchanger.exchange, action_args=recorder)
                
            elif mode == 'adjust_hyperp':
                
                model.adjust_hyperp(self.uepoch)
                
            elif mode == 'val':
                
                self.copy_to_local()
                
                model.reset_iter('val')
                
                for batch_j in range(model.data.n_batch_val):
        
                    model.val_iter(self.uepoch, recorder)
                    
                model.reset_iter('val')
                
                recorder.print_val_info(batch_i)
                
                model.current_info = recorder.get_latest_val_info()
                
                if self.verbose: recorder.save(batch_i, model.shared_lr.get_value())
    
                self.uepoch, n_workers = self.comm_request('uepoch')
                
                if self.verbose and self.uepoch % snapshot_freq == 0: 
                    save_model(model, snapshot_path, verbose=self.verbose)
                    
                self.copy_to_local()
                
                
                if epoch_start == True:
                    recorder.end_epoch(batch_i, self.uepoch)
                    epoch_start = False
                    
            elif mode='stop':
                
                self.copy_to_local()
                
                if epoch_start == True:
                    recorder.end_epoch(batch_i, self.uepoch)
                    epoch_start = False
                    
                break
                            
        model.cleanup()
            
        

        
if __name__ == '__main__':
    
    import sys
    device = sys.argv[1]
    modelfile = sys.argv[2]
    modelclass = sys.argv[3]
    
    worker = EASGD_Worker(device)
    
    worker.test_run()
    
    exit(0)
    
    config={}
    config['verbose'] = (worker.rank==0)
    config['rank'] = worker.rank
    config['size'] = worker.size
    
    import importlib
    mod = importlib.import_module(modelfile)
    modcls = getattr(mod, modelclass)

    model = modcls(config)

    worker.build(model, config)
    
    worker.EASGD_run(model)