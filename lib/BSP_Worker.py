from base.PT import PTWorker
import numpy as np

class BSP_PTWorker(PTWorker):
    
    '''
    Worker class based a specific synchronization rule (EASGD)
    
    '''
    
    def __init__(self, port, config, device):
        PTWorker.__init__(self, port = port, \
                                config = config, \
                                device = device)
                                
        self.verbose = self.config['verbose']
        self.worker_id = self.config['worker_id']
        
        self.prepare_worker()                         
        self.prepare_recorder()
        self.prepare_iterator()
        
        self.mode = None
        self.epoch = 0
        self.count = 0
        
        if self.config['resume_train'] == True:
            self.epoch = self.config['load_epoch']
            self.load_model(self.epoch)

        self.train_len = len(self.data[0]) #self.config['avg_freq']
        self.val_len = len(self.data[2])
        
        
    def prepare_param_exchanger(self):
        
        from base.exchanger import BSP_Exchanger

        self.exchanger = BSP_Exchanger(self.config,\
                                    self.drv, \
                                    self.ctx,
                                    self.model)
                                    
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
        #l_range = set(range(16))-set([1,3])
        load_weights(layers, path, load_epoch)
        #load_momentums(vels, path, load_epoch)
            
        if self.verbose: 
            print '\nlearning rate loaded %f' % s_lr.get_value()
            print 'weights and momentums loaded from epoch %d' % load_epoch
            print 'in %s' % path
        
            record_file_path = self.config['record_dir'] + 'inforec.pkl'
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
      
        layers = self.model.layers
        path = self.config['weights_dir']
        vels = self.model.vels  
        
        from base.helper_funcs import save_weights, save_momentums
        save_weights(layers, path, self.epoch)
        np.save(path + 'lr_' + str(self.epoch) + \
                        '.npy', self.model.shared_lr.get_value())
        #save_momentums(vels, self.config['weights_dir'], self.epoch)
        
        if self.verbose:
            print '\nweights and momentums saved at epoch %d' % self.epoch
        
        with open(path+"val_info.txt", "a") as f:
            f.write("\nepoch: {} val_info {}:".format(self.epoch, \
                                                    self.model.current_info))
        
            
    def train(self):
        
        # avoiding dots evaluation
        i_next = self.train_iterator.next
        r_start = self.recorder.start
        if self.size>1: exch = self.exchanger.exchange
        r_end = self.recorder.end
        r_print = self.recorder.print_train_info
        
        for i in xrange(0,self.train_len,self.size):
            
            for subb_ind in range(self.config['n_subb']):
                
                i_next(self.recorder,self.count)
                self.comm.Barrier()
                r_start()
                #print self.model.params[0].get_value()[1][1][1][1]
                if self.size>1: exch()
                
                r_end('comm')
                
            self.count += self.size
            
            r_print(self.count)
            
        self.train_iterator.reset()
        
    def val(self):
        
        self.model.set_dropout_off()
        
        for i in xrange(0,self.val_len,self.config['size']):
            
            for subb_ind in range(self.config['n_subb']):
        
                self.val_iterator.next(self.recorder,self.count)
            
                print '.',
            
        self.recorder.gather_val_info()
        
        self.recorder.print_val_info(self.count)
        
        self.model.current_info = self.recorder.get_latest_val_info()
        
        self.model.set_dropout_on()
        
        self.val_iterator.reset()
    
    def adjust_lr(self):
        
        self.model.adjust_lr(self.epoch, size = self.size)
        
    def run(self):
        
        # override PTWorker class method
        
        print 'worker started'
        
        if self.size>1: self.prepare_param_exchanger()
        
        self.adjust_lr()
        
        if self.config['initial_val']:
            self.mode = 'val'
        else:
            self.mode = 'train'
        
        
        while True:

            if self.mode == 'train':
                
                self.comm.Barrier()
                
                self.recorder.start_epoch()
                self.epoch+=1
                if self.verbose: 
                    print '\nNow training'

                self.train()
                
                self.recorder.end_epoch(self.count, self.epoch)
                
                self.mode = 'val'

            elif self.mode == 'val':
                
                self.comm.Barrier()
                
                if self.verbose: 
                    print '\nNow validating'

                self.val()
                
                self.adjust_lr()
                    
                self.recorder.save(self.count, self.model.shared_lr.get_value(), \
                        filepath = self.config['record_dir'] + 'inforec.pkl')
                
                if self.epoch % self.config['snapshot_freq'] == 0:
                    if self.config['rank'] ==0 :
                        self.save_model()
                
                if self.epoch >= self.config['n_epochs']:
                    self.mode = 'stop'
                else:
                    self.mode = 'train'
                        
            elif self.mode == 'stop':
                if self.verbose: print '\nOptimization finished'
                break
            
            else:
                raise ValueError('wrong mode')
        
        self.para_load_close()
        
                                
if __name__ == '__main__':
    
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        
    #device = 'gpu' + str(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    import sys
    device = sys.argv[1]
    if device == None:
        raise ValueError('Need to specify a GPU device')
    worker = BSP_PTWorker(port=5555, config=config, device=device)
    
    worker.run()

