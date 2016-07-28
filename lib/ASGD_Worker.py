from Async_Worker import Async_PTWorker

class ASGD_PTWorker(Async_PTWorker):
    
    '''
    Async Worker class based on a specific synchronization rule (ASGD)
    Executing training routine and periodically reporting results to server
    
    '''
    
    def __init__(self, port, config, device):
        Async_PTWorker.__init__(self, port=port, 
                                      config=config, 
                                      device=device)
            
        
    def prepare_param_exchanger(self):
        
        from base.exchanger import ASGD_Exchanger

        self.exchanger = ASGD_Exchanger(self.config, \
                                    self.drv, \
                                    etype='worker', \
                                    param_list=self.model.params, \
                                    delta_list=self.model.vels
                                    )
                                    
    def prepare_iterator(self):
        #override Async_PTWorker member function
        
        from base.iterator import P_iter
        
        # iterator won't make another copy of the model 
        # instead it will just call its compiled train function
        train_fn = self.model.get_vel
        self.train_iterator = P_iter(self.config, self.model, \
                                    self.data[0], self.data[1],  'train', train_fn)
        self.val_iterator = P_iter(self.config, self.model, \
                                    self.data[2], self.data[3], 'val')
                                    
if __name__ == '__main__':
    
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        
    #device = 'gpu' + str(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    import sys
    device = sys.argv[1]
    if device == None:
        device = 'gpu0'
    worker = ASGD_PTWorker(port=5555, config=config, device=device)
    
    worker.run()

