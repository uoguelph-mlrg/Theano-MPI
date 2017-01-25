from __future__ import absolute_import

from theanompi.lib.base import MPI_GPU_Process

from mpi4py import MPI

server_alpha = 0.5

class EASGD_Server(MPI_GPU_Process):
    
    '''
    An implementation of the server process in the Elastic Averaging SGD rule
    https://arxiv.org/abs/1412.6651
    
    implementation idea from platoon:
    https://github.com/mila-udem/platoon/tree/master/platoon/channel
    '''
    
    def __init__(self, device):
        MPI_GPU_Process.__init__(self, device) # setup ctx, comm
        
        self.worker_gpucomm = {} # gpucomm to be setup through worker registration
        self.worker_id = {}
        self.first_worker_id = None
        self.valid = {}
        self.uidx = {}
        self.adj_lr = {}
        self.last = None
        self.last_uidx = 0
        self.start_time = None
        self.uepoch = 0
        self.last_uepoch = 0
        
    def process_request(self, model, worker_id, worker_rank, message):

        
        reply = None
        
        import time
    
        # Connection related request
        
        if message in ['sync_register']:
            
            if self.first_worker_id == None:
                self.first_worker_id = worker_id
                print '[Server] recording worker is %s' % worker_id
                reply = 'first'
            # rank -> id -> gpucomm
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
            # when a new worker joins
            self.adj_lr = self.adj_lr.fromkeys(self.adj_lr, True) 
            
        # Training related requests
        
        if message == 'next':
            
            if self.start_time is None:
                self.start_time = time.time()
            # stop when finish all epochs
            if sum(self.uidx.values()) >= self.validFreq*model.n_epochs: 
                print "[Server] Total training time %.2fh" % \
                        ((time.time() - self.start_time)/3600.0)
                reply = 'stop'
        
            elif self.valid['%s' % worker_id]:
                self.valid['%s' % worker_id] = False
                reply = 'val'
        
            elif self.adj_lr['%s' % worker_id]:
                self.adj_lr['%s' % worker_id] = False
                reply = 'adjust_hyperp'
        
            else:
                reply = 'train' 
        
        elif 'done' in message:
    
            self.uidx['%s' % worker_id] += message['done']
            #print '[Server] uidx %d' % sum(self.uidx.values())

        elif message == 'uepoch':

            reply = [self.uepoch, len(self.worker_gpucomm)]
        
        if message in ['next', 'uepoch'] or 'done' in message:       
    
            now_uidx = sum(self.uidx.values())
            self.uepoch = int(now_uidx/self.validFreq)
            if self.last_uepoch != self.uepoch:
                #print "[Server] now global epoch %d" % self.uepoch
                self.last_uepoch = self.uepoch 
                # when a epoch is finished
                self.adj_lr = self.adj_lr.fromkeys(self.adj_lr, True) 
                #self.valid = self.valid.fromkeys(self.valid, True)
                # only the first worker validates
                self.valid["%s" % self.first_worker_id] = True 
                
        
        
            if self.last == None:
                
                self.last = float(time.time())
        
            if now_uidx - self.last_uidx >= 40:
        
                now = float(time.time())

                print '[Server] %d time per 40 batches: %.2f s' % \
                        (self.uepoch, (now - self.last))

                self.last_uidx = now_uidx
                self.last = now
    
        return reply
    
    def action_after(self, model, worker_id,  worker_rank, message):
        
        if message == 'disconnect':

            self.worker_gpucomm.pop(str(worker_id))
        
            print '[Server] disconnected with worker %d' % worker_id
            
        elif message == 'stop':
            
            print '[Server] stopped by %d' % worker_id
            
            import sys
            sys.exit(0)
            
            
        if message == 'sync_register':
            
            gpucomm = self.get_intranode_pair_comm(pair=(0,worker_rank))
            
            self.worker_gpucomm[str(worker_id)]= gpucomm
            
        elif message == 'exchange':
    
            self.exchanger.gpucomm = self.worker_gpucomm[str(worker_id)]
            # self.exchanger.dest = worker_rank
    
            self.exchanger.exchange()
    
        elif message == 'copy_to_local':
    
            self.exchanger.gpucomm = self.worker_gpucomm[str(worker_id)]
            # self.exchanger.dest = worker_rank
    
            self.exchanger.copy_to_local()
        
    
    def build(self, model):
        
        from theanompi.lib.helper_funcs import check_model
        
        # check model has necessary attributes
        check_model(model)
        
        # choose the type of exchanger
        from theanompi.lib.exchanger import EASGD_Exchanger
        self.exchanger = EASGD_Exchanger(alpha=server_alpha, 
                                        param_list=model.params, 
                                        etype='server')
                                        
        self.validFreq = model.data.n_batch_train
                
    def run(self, model):
        
        if self.comm == None:
            
            print 'Server communicator not initialized'
            
            return
            
        print 'server started'

        while True:
            #  Wait for next request from client
            
            request = self.comm.recv(source=MPI.ANY_SOURCE, tag=199)
                
            #  Do some process work and formulate a reply
            reply = self.process_request(model, request['id'],
                                        request['rank'],request['message'])

            #  Send reply back to client
            self.comm.send(reply, dest=request['rank'], tag=200)
            
            # Do some action work after reply
            self.action_after(model, request['id'],
                                request['rank'], request['message'])
                                                    
                                                    
if __name__ == '__main__':
    
    import sys
    device = sys.argv[1]
    modelfile = sys.argv[2]
    modelclass = sys.argv[3]

    server = EASGD_Server(device)
    
    config={}
    config['verbose'] = False #(server.rank==0)
    config['rank'] = 0
    config['size'] = 1
    config['no_paraload'] = True
    
    import importlib
    mod = importlib.import_module(modelfile)
    modcls = getattr(mod, modelclass)
    model = modcls(config)
    
    server.build(model)
    
    server.run(model)