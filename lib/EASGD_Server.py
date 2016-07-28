from Async_Server import Async_PTServer

class EASGD_PTServer(Async_PTServer):
    
    '''
    Asynchronous Server class based on a specific synchronization rule (EASGD)
    
    '''
    
    def __init__(self, port, config, device):
        Async_PTServer.__init__(self, port = port, \
                                config = config, \
                                device = device)
        
    def prepare_param_exchanger(self):
    
        from base.exchanger import EASGD_Exchanger

        self.exchanger = EASGD_Exchanger(self.config, \
                                    self.drv, \
                                    self.model.params, \
                                    etype='server')
                            
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
