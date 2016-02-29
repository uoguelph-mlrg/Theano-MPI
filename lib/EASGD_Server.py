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
        self.start_time = None
        self.last = None
        self.last_uidx = 0
        self.validFreq = len(self.data[0])
                                
        
    def prepare_param_exchanger(self):
    
        from exchanger import EASGD_Exchanger

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
        except KeyError:
            self.valid['%s' % worker_id] = False
            self.uidx['%s' % worker_id] = 0
            reply = 'train'
        
        if self.last == None:
            self.last = float(time.time())
        
        if message == 'next':
            
            if self.start_time is None:
                self.start_time = time.time()
                
            if self.valid['%s' % worker_id]:
                self.valid['%s' % worker_id] = False
                reply = 'val'
            else:
                reply = 'train' 
                
        elif 'done' in message:
            
            self.uidx['%s' % worker_id] += message['done']
            
            if np.mod(self.uidx['%s' % worker_id], self.validFreq) == 0: # val every epoch
                
                self.valid['%s' % worker_id] = True
            
        elif message == 'uidx':
        
            reply = int(sum(self.uidx.values()))
                
        if message in ['next', 'uidx'] or 'done' in message:
               
            if sum(self.uidx.values()) >= self.max_mb: # stop when finish all epochs
                reply = 'stop'
                self.worker_is_done(worker_id)
                print "Training time {:.4f}s".format(time.time() - self.start_time)
                print "Number of samples:", self.uidx['%s' % worker_id]
    
            now_uidx = sum(self.uidx.values())
            now = float(time.time())
    
            if now_uidx - self.last_uidx >= 400:
        
                print '%d time per 5120 images: %.2f s' % \
                        (int(now_uidx/self.validFreq), (now - self.last)/(now_uidx - self.last_uidx)*40.0)
        
                self.last_uidx = now_uidx
                self.last = now
        
        return reply
            
                                        
    def action_after(self, worker_id, message):
        
        # override PTServer class method
        
        PTServer.action_after(self, worker_id, message )
        if message == 'connect':
            
            self.prepare_param_exchanger()
            
        elif message == 'disconnect':
            
            intercomm = self.worker_comm[str(worker_id)]
            intercomm.Disconnect()
            self.worker_comm.pop(str(worker_id))
            
            reply = 'disconnected'
            print 'disconnected with worker', worker_id
            
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
        
    server = EASGD_PTServer(port=5555, config=config, device='gpu7')
    
    server.run()