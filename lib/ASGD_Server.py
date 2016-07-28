from Async_Server import Async_PTServer

class ASGD_PTServer(Async_PTServer):
    
    '''
    Asynchronous Server class based on a specific synchronization rule (ASGD)
    
    '''
    
    def __init__(self, port, config, device):
        Async_PTServer.__init__(self, port = port, \
                                config = config, \
                                device = device)
        
    def prepare_param_exchanger(self):
    
        from base.exchanger import ASGD_Exchanger

        self.exchanger = ASGD_Exchanger(self.config, \
                                    self.drv, \
                                    etype='server', \
                                    param_list=self.model.params
                                    )

if __name__ == '__main__':
    
    import yaml  
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        
    #device = 'gpu' + str(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    import sys
    device = sys.argv[1]
    if device == None:
        device = 'gpu7'
        
    server = ASGD_PTServer(port=5555, config=config, device=device)
    
    server.run()
