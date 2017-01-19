from __future__ import absolute_import

from theanompi.lib.base import MPI_GPU_Process

from mpi4py import MPI

class EASGD_Server(MPI_GPU_Process):
    
    def __init__(self, device):
        MPI_GPU_Process.__init__(self, device)
        
        self.worker_gpucomm = {}
        self.worker_id = {}
        self.first_worker_id = None
        
    def process_request(self, worker_id, worker_rank, message):

        # override Server class method, for connection related request
        reply = None
    
        if message in ['sync_register']:
            
            if self.first_worker_id == None:
                self.first_worker_id = worker_id
                print '[Server] recording worker is %s' % worker_id
                reply = 'first'
            
            self.worker_id[str(worker_rank)] = int(worker_id)
            
            print '[Server] registered worker %d' % worker_id
            
            return reply
            
        try:
            valid = self.valid['%s' % worker_id]
            amount = self.uidx['%s' % worker_id]
            adj_lr = self.adj_lr['%s' % worker_id]
        except KeyError:
            self.valid['%s' % worker_id] = False
            self.adj_lr['%s' % worker_id] = False
            self.uidx['%s' % worker_id] = 0
            self.adj_lr = self.adj_lr.fromkeys(self.adj_lr, True) # when a new worker joins
            
        # Training related requests
        
        if message == 'next':
            
            if self.start_time is None:
                self.start_time = time.time()
        
            if sum(self.uidx.values()) >= self.max_mb: # stop when finish all epochs
                print "[Server] Total training time %.2fh" % ((time.time() - self.start_time)/3600.0)
                reply = 'stop'
        
            elif self.valid['%s' % worker_id]:
                self.valid['%s' % worker_id] = False
                reply = 'val'
        
            elif self.adj_lr['%s' % worker_id]:
                self.adj_lr['%s' % worker_id] = False
                reply = 'adjust_lr'
        
            else:
                reply = 'train' 
        
        elif 'done' in message:
    
            self.uidx['%s' % worker_id] += message['done']
            #print '[Server] uidx %d' % sum(self.uidx.values())

        elif message == 'uepoch':

            reply = [self.uepoch, len(self.worker_comm)]
        
        if message in ['next', 'uepoch'] or 'done' in message:       
    
            now_uidx = sum(self.uidx.values())
            self.uepoch = int(now_uidx/self.validFreq)
            if self.last_uepoch != self.uepoch:
                #print "[Server] now global epoch %d" % self.uepoch
                self.last_uepoch = self.uepoch 
                self.adj_lr = self.adj_lr.fromkeys(self.adj_lr, True) # when a epoch is finished
                #self.valid = self.valid.fromkeys(self.valid, True)
                self.valid["%s" % self.first_worker_id] = True # only the first worker validates
        
                # tunning server alpha
                a_step1, a_step2, a_step3 = self.config['alpha_step']
                if self.uepoch>a_step1 and self.uepoch< a_step2:
                    step_idx = 1
                elif self.uepoch>a_step2 and self.uepoch< a_step3:
                    step_idx = 2
                elif self.uepoch>a_step3:
                    step_idx = 3
                else:
                    step_idx = 0
                self.exchanger.alpha=self.config['server_alpha'] - self.config['alpha_minus']*step_idx
                print 'server alpha changed to %f' % self.exchanger.alpha
        
        
            if self.last == None:
                self.last = float(time.time())
        
            if now_uidx - self.last_uidx >= 400:
        
                now = float(time.time())

                print '[Server] %d time per 5120 images: %.2f s' % \
                        (self.uepoch, (now - self.last)/(now_uidx - self.last_uidx)*40.0)

                self.last_uidx = now_uidx
                self.last = now
    
        return reply
    
    def action_after(self, worker_id,  worker_rank, message):
        
        if message == 'disconnect':

            self.worker_comm.pop(str(worker_id))
        
            print '[Server] disconnected with worker %d' % worker_id
            
        elif message == 'stop':
            
            print '[Server] stopped by %d' % worker_id
            
            import sys
            sys.exit(0)
            
            
        if message in ['sync_register']:
            
            gpucomm = self.get_intranode_comm_pair(worker_rank)
            
            self.worker_comm[str(worker_id)]= gpucomm
            
        elif message == 'exchange':
    
            self.exchanger.gpucomm = self.worker_comm[str(worker_id)]
            # self.exchanger.dest = worker_rank
    
            self.exchanger.exchange()
    
        elif message == 'copy_to_local':
    
            self.exchanger.gpucomm = self.worker_comm[str(worker_id)]
            # self.exchanger.dest = worker_rank
    
            self.exchanger.copy_to_local()
        
    
    def build(self, model):
        
        from theanompi.lib.helper_funcs import check_model
        
        # check model has necessary attributes
        check_model(model)
        
        # choose the type of exchanger
        from theanompi.lib.exchanger import EASGD_Exchanger
        self.exchanger = EASGD_Exchanger(self.config, 
                                    self.drv, 
                                    self.model.params, 
                                    etype='server')
                
    def EASGD_run(self):
        
        if self.comm == None:
            
            print 'Server communicator not initialized'
            
            return
            
        print 'server started'
        
        # after the barrier, run asynchronously
        self.comm.Barrier()

        while True:
            #  Wait for next request from client
            
            request = self.comm.recv(source=MPI.ANY_SOURCE, tag=199)
                
            #  Do some process work and formulate a reply
            reply = self.process_request(request['id'],request['rank'],\
                                                    request['message'])

            #  Send reply back to client
            self.comm.send(reply, dest=request['rank'], tag=200)
            
            # Do some action work after reply
            self.action_after(request['id'],request['rank'], \
                                                    request['message'])
                                                    
                                                    
if __name__ == '__main__':
    
    import sys
    device = sys.argv[1]
    modelfile = sys.argv[2]
    modelclass = sys.argv[3]

    server = EASGD_Server(device)
    
    import importlib
    mod = importlib.import_module(modelfile)
    modcls = getattr(mod, modelclass)
    model = modcls(config)
    
    server.build(model)
    
    server.EASGD_run(model)