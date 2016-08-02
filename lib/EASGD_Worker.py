from Async_Worker import Async_PTWorker
import numpy as np

class EASGD_PTWorker(Async_PTWorker):
    
    '''
    Worker class based on a specific synchronization rule (EASGD)
    Executing training routine and periodically reporting results to server
    
    '''
    
    def __init__(self, port, config, device):
        Async_PTWorker.__init__(self, port=port, 
                                      config=config, 
                                      device=device)
                                      
        import time
        compile_time = time.time()
        
        self.prepare_train_fn() # different in EASGD and ASGD # need to be called before prepare_para_load()
        self.prepare_val_fn()
        
        if self.verbose: print 'compile_time %.2f s' % \
                                (time.time() - compile_time)
        
        self.prepare_para_load() #needs to be after compile_train() and compile_val()
        
                                
        self.prepare_recorder()
        self.prepare_iterator() # different in EASGD and ASGD # needs to be called after prepare_pare_load()
        
        self.uepoch = None
        if self.config['resume_train'] == True:
            self.uepoch = self.config['load_epoch']
            self.load_model(self.uepoch)
    
        
    def prepare_param_exchanger(self):
        
        from base.exchanger import EASGD_Exchanger

        self.exchanger = EASGD_Exchanger(self.config, \
                                    self.drv, \
                                    self.model.params, \
                                    etype='worker')
                                    
    def prepare_iterator(self):
        
        from base.iterator import P_iter
        
        self.train_iterator = P_iter(self.config, self.model, \
                                    self.data[0], self.data[1],  'train', self.model.train)
        self.val_iterator = P_iter(self.config, self.model, \
                                    self.data[2], self.data[3], 'val', self.model.val)
                                    
    def prepare_train_fn(self):
        
        # to make sure model compiles necessary functions (train() for EASGD) and allocate necessary extra param memory (vels for EASGD)
        
        # allocate supporting params for this worker type
        
        model = self.model
        
        import theano
        
        model.vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in model.params]
        
        self.prepare_update_dict()
        
        updates_w, = model.update_dict
        
        train_args = {"inputs":[model.subb_ind], "outputs": [model.cost,model.error], "updates": updates_w, \
                                                                  "givens": [(model.x,  model.shared_x_slice), 
                                                                             (model.y,  model.shared_y_slice),
                                                                             (model.lr, model.shared_lr)]}
        model.compile_train_fn_list = [train_args]
        
        model.compile_train()
        
        model.train , = model.compiled_train_fn_list
            
                                    
    def prepare_update_dict(self):
    
        model = self.model
        config = self.config
        
        use_momentum=config['use_momentum'], 
        use_nesterov_momentum=config['use_nesterov_momentum']
    
        try:
            size = config['size']
            verbose = config['rank'] == 0
        except KeyError:
            size = 1
            verbose = True
        
        params, grads, weight_types = model.params, model.grads, model.weight_types
        
        vels = model.vels
    
        lr = model.shared_lr #T.scalar('lr')  # symbolic learning rate
        mu = model.mu # def: 0.9 # momentum
        eta = model.eta  #0.0002 # weight decay

        updates_w = []

        if use_momentum:

            assert len(weight_types) == len(params)
            
            k=0

            for param_i, grad_i, weight_type in \
                    zip(params, grads, weight_types):

                if weight_type == 'W':
                    real_grad = grad_i + eta * param_i
                    real_lr = lr
                elif weight_type == 'b':
                    real_grad = grad_i
                    real_lr = 2. * lr
                else:
                    raise TypeError("Weight Type Error")

                if use_nesterov_momentum:
                    vel_i_next = mu ** 2 * vels[k] - (1 + mu) * real_lr * real_grad
                else:
                    vel_i_next = mu * vels[k] - real_lr * real_grad
                
                updates_w.append((vels[k], vel_i_next))    
                updates_w.append((param_i, param_i + vel_i_next))
                    
                k=k+1
                

        else:
            
            k=0
            
            for param_i, grad_i, weight_type in \
                    zip(params, grads, weight_types):
                    
                if weight_type == 'W':
                        
                    update =  param_i - lr * grad_i - eta * lr * param_i

                elif weight_type == 'b':
                        
                    update = param_i - 2 * lr * grad_i

                updates_w.append((param_i, update))
                    
                    
                k=k+1
            
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
    worker = EASGD_PTWorker(port=5555, config=config, device=device)
    
    worker.run()

