from Async_Worker import Async_PTWorker
import numpy as np

class ASGD_PTWorker(Async_PTWorker):
    
    '''
    Async Worker class based on a specific synchronization rule (ASGD)
    Executing training routine and periodically reporting results to server
    
    '''
    
    def __init__(self, port, config, device):
        Async_PTWorker.__init__(self, port=port, 
                                      config=config, 
                                      device=device)
                                      
        self.verbose = self.config['verbose']
        self.spawn_load()

        self.init_device()
        self.get_data()
        self.build_model()
                                      
        import time
        compile_time = time.time()
        
        self.prepare_train_fn() # different in EASGD and ASGD # need to be called before prepare_para_load()
        self.prepare_val_fn()
        
        if self.verbose: print 'compile_time %.2f s' % \
                                (time.time() - compile_time)
        
        self.para_load_init() #needs to be after compile_train() and compile_val()
                                
        self.prepare_recorder()
        self.prepare_iterator() # different in EASGD and ASGD # needs to be called after prepare_pare_load()
        
        self.uepoch = None
        if self.config['resume_train'] == True:
            self.uepoch = self.config['load_epoch']
            self.load_model(self.uepoch)
                                      
        
    def prepare_param_exchanger(self):
        
        from base.exchanger import ASGD_Exchanger

        self.exchanger = ASGD_Exchanger(self.config, \
                                    self.drv, \
                                    etype='worker', \
                                    param_list=self.model.params, \
                                    delta_list=self.model.vels2
                                    )
        
                                    
        #prepare delta accumulator
        
        
                                    
    def prepare_iterator(self):
        #override Async_PTWorker member function
        
        from base.iterator import P_iter
        
        # iterator won't make another copy of the model 
        # instead it will just call its compiled train function
        
        self.train_iterator = P_iter(self.config, self.model, \
                                    self.data[0], self.data[1],  'train', self.model.train_vel_acc)
        self.val_iterator = P_iter(self.config, self.model, \
                                    self.data[2], self.data[3], 'val', self.model.val)
                                    
    def prepare_train_fn(self):
        
        # to make sure model compiles necessary functions (train() for EASGD) and allocate necessary extra param memory (vels for EASGD)
        
        # allocate supporting params for this worker type
        
        model = self.model
        
        import theano
        
        model.vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in model.params]
        model.vels2 = [theano.shared(param_i.get_value() * 0.)
            for param_i in model.params]
        
        self.prepare_update_dict()
        
        updates_w, = model.update_dict
        
        train_args = {"inputs":[model.subb_ind], "outputs": [model.cost,model.error], "updates": updates_w, \
                                                                  "givens": [(model.x,  model.shared_x_slice), 
                                                                             (model.y,  model.shared_y_slice),
                                                                             (model.lr, model.shared_lr)]}
        model.compile_train_fn_list = [train_args]
        
        model.compile_train()
        
        model.train_vel_acc , = model.compiled_train_fn_list
            
                                    
    def prepare_update_dict(self):
    
        model = self.model
        config = self.config
        
        use_momentum=config['use_momentum'], 
        use_nesterov_momentum=config['use_nesterov_momentum']
        
        if use_momentum:
            
            from base.opt import MSGD
            
            updates_w = MSGD(model, use_nesterov_momentum,worker_type)
            
        else:
            
            from base.opt import SGD
            
            updates_w = SGD(model,worker_type)
            
        self.model.update_dict = [updates_w]
        
    def prepare_val_fn(self):
        
        self.model.compile_val()
        
        
    def adjust_lr(self):
        
        # override function in Async_PTWorker class
        
        self.uepoch, self.n_workers = self.request('uepoch')
        
        #if self.verbose: print 'global epoch %d, %d workers online' % (self.uepoch, self.n_workers )
        
        self.model.adjust_lr(self.uepoch)
        
        new_lr = self.model.shared_lr.get_value()
        
        
        self.model.shared_lr.set_value(np.float32(new_lr*self.n_workers))

    
        if self.verbose: 
            print 'Learning rate now: %.10f' % \
                    np.float32(self.model.shared_lr.get_value())
                                    
                                    
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

