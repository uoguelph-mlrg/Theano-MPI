from base.PT import PTWorker, test_intercomm
import numpy as np

class EASGD_PTWorker(PTWorker):
    
    '''
    Worker class based on a specific synchronization rule (EASGD)
    Executing training routine and periodically reporting results to server
    
    '''
    
    def __init__(self, port, config, device):        
        PTWorker.__init__(self, port = port, \
                                config = config, \
                                device = device)
                                
        self.worker_id = self.config['worker_id']
        
        if self.config['sync_start']:
            # sync start register, 
            # use the COMM_WORLD to communicate with server
            self._MPI_register()
            self.model.verbose = self.verbose 
        else:
            # async start register, 
            # build a separate intercomm to communicate with server
            self.MPI_register()
            self.model.verbose = self.verbose
            
        #if self.verbose: print 'worker registered'
                             
        self.prepare_worker()                        
        self.prepare_recorder()
        self.prepare_iterator()
        
        self.uepoch = None
        if self.config['resume_train'] == True:
            self.uepoch = self.config['load_epoch']
            self.load_model(self.uepoch)

        self.train_len = self.config['avg_freq']
        self.val_len = len(self.data[2])
        self.mode = None
        self.lastmode = None
        self.count = 0
        
        if self.verbose:
            self.rec_name = 'inforec.pkl'
        else:
            self.rec_name = 'inforec_'+ str(self.worker_id) + '.pkl'
        
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
        s_lr = self.model.shared_lr
        vels = self.model.vels

        
        # TODO needs to verify the previous lr is when training with avg, scaled by size
        import os  
        s_lr.set_value(np.load(os.path.join(path, 
                  'lr_' + str(load_epoch) + '.npy')))
        
        from base.helper_funcs import load_weights, load_momentums
        load_weights(layers, path, load_epoch)
        #load_momentums(vels, path, load_epoch)
            
        if self.verbose: 
            print '\nlearning rate loaded %f' % s_lr.get_value()
            print 'weights and momentums loaded from epoch %d in %s' % (load_epoch,path)
            
            record_file_path = self.config['record_dir'] + 'inforec.pkl' # bug which worker's inforec should be used, use only recording worker's, if exist, put into history
            if os.path.exists(record_file_path):
                import glob
                history_folder = self.config['record_dir']+ 'history*' 
                find = glob.glob(history_folder)
                #print find
                if find != []:
                    history_folder = sorted(find)[-1]
                    #print history_folder

                    history_folder = history_folder.split('_')[0] + '_' + \
                             "%d" % (int(history_folder.split('_')[-1])+1) + '/'
                    
                else:
                    history_folder = self.config['record_dir']+ 'history_0' + '/'
                
                print 'creating inforec history folder: ' + history_folder
                    
                os.makedirs(history_folder)
                import shutil
                shutil.copy(record_file_path, history_folder+'inforec.pkl')
                self.recorder.load(filepath = record_file_path)
                # print type(self.recorder.info_dict['train_info'])
                # print len(self.recorder.info_dict['train_info'])
                #
                # print type(self.recorder.info_dict['val_info'])
                # print len(self.recorder.info_dict['val_info'])
            
            else:
                raise OSError('record fle not found at %s ' % record_file_path)
                
            
    def save_model(self): 
        
        assert self.uepoch != None
        layers = self.model.layers
        path = self.config['weights_dir']
        vels = self.model.vels  
        
        from base.helper_funcs import save_weights, save_momentums
        save_weights(layers, path, self.uepoch)
        np.save(path + 'lr_' + str(self.uepoch) + \
                        '.npy', self.model.shared_lr.get_value())
        #save_momentums(vels,self.config['weights_dir'], self.uepoch)
        if self.verbose:
            print '\nweights and momentums saved at epoch %d' % self.uepoch
            
    def train(self):
        
        for i in range(self.train_len):
            
            for subb_ind in range(self.config['n_subb']):
                #print self.count
                self.train_iterator.next(self.recorder,self.count)
            
            self.count += 1
            self.recorder.print_train_info(self.count)

            
        self.recorder.start()
        reply = self.request(dict(done=self.train_len))
        
        self.exchanger.comm = self.intercomm
        self.action(message = 'exchange', \
                    action=self.exchanger.exchange)
        self.recorder.end('comm')
        
        self.lastmode = 'train'

        
    def val(self):
        
        if self.lastmode == 'train':
            self.train_iterator.reset()
        
        self.model.set_dropout_off()
        
        for i in range(self.val_len):
        
            self.val_iterator.next(self.recorder,self.count)
            
            if self.verbose: print '.',
        
        self.recorder.print_val_info(self.count)
        
        self.model.set_dropout_on()
        
        self.val_iterator.reset()
                                    
    
    def copy_to_local(self):
        
        self.exchanger.comm = self.intercomm
        self.action(message = 'copy_to_local', \
                    action=self.exchanger.copy_to_local)
        if self.verbose: print '\nSynchronized param with server'
                    
    def adjust_lr(self):
        
        self.uepoch, self.n_workers = self.request('uepoch')
        
        #if self.verbose: print 'global epoch %d, %d workers online' % (self.uepoch, self.n_workers )
        
        self.model.adjust_lr(self.uepoch, size = self.n_workers)
        
        
    def run(self):
        
        # override PTWorker class method
        
        if self.verbose: print 'worker %s started' % self.worker_id
        
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

                self.train()
                
            if self.mode == 'adjust_lr':
                
                self.adjust_lr()
                #self.copy_to_local()

            if self.mode == 'val':

                if self.verbose: 
                    print '\nNow validating'
                
                self.copy_to_local()

                self.val()
                
                self.recorder.save(self.count, self.model.shared_lr.get_value(), \
                        filepath = self.config['record_dir'] + self.rec_name)
                        
                self.uepoch, self.n_workers = self.request('uepoch')
                
                if self.uepoch % self.config['snapshot_freq'] == 0: # TODO BUG: if too few images in training set, uepoch may skip more than 1 per check
                    self.save_model()
                
                self.copy_to_local()
                
                if epoch_start == True:
                    self.recorder.end_epoch(self.count, self.uepoch)
                    epoch_start = False
                        
            if self.mode == 'stop':
                
                self.copy_to_local()

                self.val()
                
                if epoch_start == True:
                    self.recorder.end_epoch(self.count, self.uepoch)
                    epoch_start = False
                
                if self.verbose: print '\nOptimization finished'
                
                break
        
        self.para_load_close() # TODO some workers blocked here can't disconnect
        self.ctx.pop()
        self.MPI_deregister()
        
        if self.verbose: print '\nWorker %s deregistered' % self.worker_id
        
   
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

