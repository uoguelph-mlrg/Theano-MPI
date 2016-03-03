from base.PT import PTServer
import time
import numpy as np

class EASGD_PTServer(PTServer):
    
    '''
    Server class based a specific synchronization rule (EASGD)
    
    '''
    
    def __init__(self, port, config, device):
        PTServer.__init__(self, port = port, \
                                config = config, \
                                device = device)
        self.config['irank'] = 0
        
        self.max_mb = self.config['n_epochs'] * len(self.data[0])
        print 'max_mb = %d' % self.max_mb
        self.uidx = {}
        self.valid = {}
        self.adj_lr = {}
        self.start_time = None
        self.last = None
        self.last_uidx = 0
        self.validFreq = len(self.data[0])
        self.uepoch = 0
        self.last_uepoch = 0
        self.first_worker_id = None
                                
        
    def prepare_param_exchanger(self):
    
        from base.exchanger import EASGD_Exchanger

        self.exchanger = EASGD_Exchanger(self.config, \
                                    self.drv, \
                                    self.model.params, \
                                    etype='server')
                                    
    def process_request(self, worker_id, message):
        
        # override PTServer class method
        
        reply = PTServer.process_request(self, worker_id, message)
        
        if reply != None:
            return reply
        else:
            pass
            
        try:
            valid = self.valid['%s' % worker_id]
            amount = self.uidx['%s' % worker_id]
            adj_lr = self.adj_lr['%s' % worker_id]
        except KeyError:
            self.valid['%s' % worker_id] = False
            self.adj_lr['%s' % worker_id] = False
            self.uidx['%s' % worker_id] = 0
            self.adj_lr = self.adj_lr.fromkeys(self.adj_lr, True) # when a new worker joins
            
        if self.first_worker_id == None:
            self.first_worker_id = worker_id
            print 'recording worker %s' % worker_id
        
        if self.last == None:
            self.last = float(time.time())
        
        if message == 'next':
            
            if self.start_time is None:
                self.start_time = time.time()
                
            if self.adj_lr['%s' % worker_id]:
                self.adj_lr['%s' % worker_id] = False
                reply = 'adjust_lr'
                
            elif self.valid['%s' % worker_id]:
                self.valid['%s' % worker_id] = False
                reply = 'val'
                
            else:
                reply = 'train' 
                
        elif 'done' in message:
            
            self.uidx['%s' % worker_id] += message['done']

        elif message == 'uepoch':
        
            reply = [self.uepoch, len(self.worker_comm)]
                
        if message in ['next', 'uidx'] or 'done' in message:
               
            if sum(self.uidx.values()) >= self.max_mb: # stop when finish all epochs
                reply = 'stop'
                print "Training time {:.4f}s".format(time.time() - self.start_time)
                print "Number of samples:", self.uidx['%s' % worker_id]
    
            now_uidx = sum(self.uidx.values())
            self.uepoch = int(now_uidx/self.validFreq)
            if self.last_uepoch != self.uepoch:
                print "now global epoch %d" % self.uepoch
                self.last_uepoch = self.uepoch 
                self.adj_lr = self.adj_lr.fromkeys(self.adj_lr, True) # when a epoch is finished
                #self.valid = self.valid.fromkeys(self.valid, True)
                self.valid["%s" % self.first_worker_id] = True # only the first worker validates
                
            now = float(time.time())
    
            if now_uidx - self.last_uidx >= 400:
        
                print '%d time per 5120 images: %.2f s' % \
                        (self.uepoch, (now - self.last)/(now_uidx - self.last_uidx)*40.0)
        
                self.last_uidx = now_uidx
                self.last = now
        
        return reply
            
                                        
    def action_after(self, worker_id, message):
        
        # override PTServer class method
        
        PTServer.action_after(self, worker_id, message )
        
        if message == 'connect':
            
            self.prepare_param_exchanger()
            
        elif message == 'exchange':
            
            self.exchanger.comm = self.worker_comm[str(worker_id)]
            
            self.exchanger.exchange()
            
        elif message == 'copy_to_local':
            
            self.exchanger.comm = self.worker_comm[str(worker_id)]
            
            self.exchanger.copy_to_local()
            



if __name__ == '__main__':
    
    import yaml  
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        
    #device = 'gpu' + str(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    import sys
    device = sys.argv[1]
    if device == None:
        device = 'gpu7'
        
    server = EASGD_PTServer(port=5555, config=config, device=device)
    
    server.run()