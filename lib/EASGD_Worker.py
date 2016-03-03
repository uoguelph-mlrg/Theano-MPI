from base.PT import PTWorker, test_intercomm
import numpy as np

class EASGD_PTWorker(PTWorker):
    
    '''
    Worker class based a specific synchronization rule (EASGD)
    
    '''
    
    def __init__(self, port, config, device):        
        PTWorker.__init__(self, port = port, \
                                config = config, \
                                device = device)
                                
        self.verbose = self.config['verbose']
        self.worker_id = self.config['worker_id']
        
        self.MPI_register() 
        print 'worker registered'                      
        self.prepare_worker()                        
        self.prepare_recorder()
        self.prepare_iterator()
        
        if self.config['resume_train'] == True:
            self.epoch = self.config['load_epoch']
            self.load_model(self.epoch)
        else:
            self.epoch = 0

        self.train_len = self.config['avg_freq']
        self.val_len = len(self.data[2])
        self.mode = None
        self.epoch = 0
        self.count = 0
        
    def prepare_param_exchanger(self):
        
        from base.exchanger import EASGD_Exchanger

        self.exchanger = EASGD_Exchanger(self.config, \
                                    self.drv, \
                                    self.model.params, \
                                    etype='worker')
                                    
    def prepare_recorder(self):
        
        from base.recorder import Recorder
        
        self.recorder = Recorder(self.config)
                                    
    def prepare_iterator(self):
        
        from base.iterator import P_iter
        
        # iterator won't make another copy of the model 
        # instead it will just call its compiled train function
        
        self.train_iterator = P_iter(self.config, self.model, \
                                    self.data[0], self.data[1],  'train')
        self.val_iterator = P_iter(self.config, self.model, \
                                    self.data[2], self.data[3], 'val')
                                    
    def load_model(self, load_epoch):
        
        layers = self.model.layers
        path = self.config['load_path']
        learning_rate = self.model.lr
        vels = self.model.vels

        if learning_rate != None: 
            # TODO needs to verify the previous lr is when training with avg, scaled by size
            import os  
            learning_rate.set_value(np.load(os.path.join(path, 
                      'lr_' + str(load_epoch) + '.npy')))
        
        from base.helper_funcs import load_weights, load_momentums
        load_weights(layers, path, load_epoch)
        if vels != None:
            load_momentums(vels, path, load_epoch)
            
        if self.verbose: 
            print '\nlearning rate loaded %f' % learning_rate.get_value()
            print 'weights and momentums loaded from epoch %d' % load_epoch
            
    def save_model(self): 
      
        layers = self.model.layers
        path = self.config['weights_dir']
        vels = self.model.vels  
        
        from base.helper_funcs import save_weights, save_momentums
        save_weights(layers, path, self.epoch)
        np.save(path + 'lr_' + str(self.epoch) + \
                        '.npy', self.model.lr.get_value())
        save_momentums(vels, 
                       self.config['weights_dir'], self.epoch)
        if self.verbose:
            print '\nweights and momentums saved at epoch %d' % self.epoch
            
    def train(self):
        
        for i in range(self.train_len):
            
            #print self.count
            self.recorder = self.train_iterator.next(self.recorder,self.count)
            self.count += 1
            self.recorder.print_train_info(self.count)
            
            
        self.recorder.start()
        reply = self.request(dict(done=self.train_len))
        
        self.exchanger.comm = self.intercomm
        self.action(message = 'exchange', \
                    action=self.exchanger.exchange)
        self.recorder.end('comm')

        
    def val(self):
        
        self.val_iterator.reset()
        
        self.model.set_dropout_off()
        
        for i in range(self.val_len):
        
            self.recorder = self.val_iterator.next(self.recorder,self.count)
            
            if self.verbose: print '.',
        
        self.recorder.print_val_info(self.count)
        
        self.model.set_dropout_on()
        
        self.train_iterator.reset()
                                    
    
    def copy_to_local(self):
        
        self.exchanger.comm = self.intercomm
        self.action(message = 'copy_to_local', \
                    action=self.exchanger.copy_to_local)
        if self.verbose: print '\nSynchronized parameter with server'
                    
    def adjust_lr(self):
        
        self.uepoch, self.n_workers = self.request('uepoch')
        
        print 'global epoch %d, %d workers online' % (self.uepoch, self.n_workers )
        
        self.model.adjust_lr(self.uepoch, size = self.n_workers)
        
        
    def run(self):
        
        # override PTWorker class method
        
        print 'worker started'
        
        self.prepare_param_exchanger()
        
        # start training with the most recent server parameter
        self.copy_to_local()
        
        self.adjust_lr()
                    
        epoch_start = False
        

        while True:

            self.mode = self.request('next')
            
            #print self.mode

            if self.mode == 'train':
                
                if epoch_start == False:
                    self.recorder.start_epoch()
                    epoch_start = True
                    if self.verbose: 
                        print '\nNow training'
                        self.epoch+=1

                self.train()
                
            if self.mode == 'adjust_lr':
                
                self.adjust_lr()
                #self.copy_to_local()

            if self.mode == 'val':

                if self.verbose: 
                    print '\nNow validating'
                
                self.copy_to_local()

                self.val()
                
                self.adjust_lr()
                    
                self.recorder.save(self.count, self.model.lr.get_value(), \
                        filepath = self.config['record_dir'] + \
                        'inforec_'+ str(self.worker_id) + '.pkl')
                
                if self.epoch % self.config['snapshot_freq'] == 0:
                    if self.config['rank'] ==0 :
                        self.save_model()
                        
                self.copy_to_local()
                
                if epoch_start == True:
                    self.recorder.end_epoch(self.count, self.uepoch)
                    epoch_start = False
                        
            if self.mode == 'stop':
                self.train_iterator.stop_load()
                if self.verbose: print '\noptimization finished'
                break
        
        self.para_load_close() # TODO some workers blocked here can't disconnect
        
        self.MPI_deregister()
        
        print 'worker deregistered'
        
   
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

