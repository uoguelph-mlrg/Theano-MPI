from base.PT import PTWorker
import numpy as np

'''
Author: He Ma, Fei Mao, Graham Taylor
School of Engineering, University of Guelph

Copyright 2016 University of Guelph. Licensed under the
Educational Community License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may
obtain a copy of the License at

http://www.osedu.org/licenses /ECL-2.0 

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS"
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing
permissions and limitations under the License.
'''

class BSP_PTWorker(PTWorker):
    
    '''
    Worker class based a specific synchronization rule (BSP)
    
    '''
    
    def __init__(self, config, device):
        PTWorker.__init__(self, config = config, \
                                device = device)
                                
        self.verbose = self.config['verbose']
        
        import time
        compile_time = time.time()
        
        self.prepare_train_fn() # 1 (local to worker type) allocate supporting params and compiling theano functions
        self.prepare_val_fn()
        
        if self.verbose: print 'compile_time %.2f s' % \
                                (time.time() - compile_time)
        
        self.prepare_para_load() #needs to be after compile_train and compile_val()  
                              
        self.prepare_recorder()
        self.prepare_iterator()
        
        self.mode = None
        self.epoch = 0
        self.count = 0
        
        self.train_len = len(self.data[0]) #self.config['avg_freq']
        self.val_len = len(self.data[2])
        
        
    def prepare_param_exchanger(self):
        
        from base.exchanger import BSP_Exchanger
        
        # 3 (local to worker type)

        self.exchanger = BSP_Exchanger(self.config,\
                                    self.drv, \
                                    self.ctx,
                                    self.model)
                                    
    def prepare_train_fn(self):
        
        # to make sure model compiles necessary functions (get_vels() and descent() for cdd, or train() for avg) and allocate necessary extra param memory (vels,vels2 for cdd, or nothing for avg)
        
        # allocate supporting params for this worker type
        
        worker_type=self.config['worker_type']
        
        model = self.model
        
        if worker_type == 'cdd':
            
            import theano
            
            model.vels = [theano.shared(param_i.get_value() * 0.)
                for param_i in model.params]
            
            model.vels2 = [theano.shared(param_i.get_value() * 0.)
                        for param_i in model.params]
                        
            self.prepare_update_dict(worker_type='cdd')
            
            updates_v, updates_dv = model.update_dict
            
            get_vel_args = {"inputs":[model.subb_ind], "outputs":[model.cost,model.error], "updates":updates_v, \
                                                           "givens":[(model.x,  model.shared_x_slice), 
                                                                     (model.y,  model.shared_y_slice),
                                                                     (model.lr, model.shared_lr)]}
                                                                     
            descent_vel_args = {"inputs":[], "outputs":[], "updates":updates_dv}
                                                                     
            model.compile_train_fn_list = [get_vel_args, descent_vel_args]
            
            model.compile_train() # needs compile model before para_load_init() # 2 (local to worker type)
            
            model.get_vel, model.descent_vel = model.compiled_train_fn_list
        

        
        elif worker_type == 'avg':
            
            import theano
            
            model.vels = [theano.shared(param_i.get_value() * 0.)
                for param_i in model.params]
            
            self.prepare_update_dict(worker_type='avg')
            
            updates_w, = model.update_dict
            
            train_args = {"inputs":[model.subb_ind], "outputs": [model.cost,model.error], "updates": updates_w, \
                                                                      "givens": [(model.x,  model.shared_x_slice), 
                                                                                 (model.y,  model.shared_y_slice),
                                                                                 (model.lr, model.shared_lr)]}
            model.compile_train_fn_list = [train_args]
            
            model.compile_train()
            
            model.train , = model.compiled_train_fn_list
                    
        
                    
    def prepare_update_dict(self, worker_type):
    
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
        
        vels, vels2 = model.vels, model.vels2
    
        lr = model.lr #shared_lr #T.scalar('lr')  # symbolic learning rate
        mu = model.mu # def: 0.9 # momentum
        eta = model.eta  #0.0002 # weight decay

        updates_w = [] # for avg
        
        updates_v = [] # for cdd
        updates_dv = [] # for cdd

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
                    
                if worker_type == 'cdd':

                    updates_v.append((vels[k], vel_i_next))
                    updates_dv.append((param_i, param_i + vels2[k]))
                    
                elif worker_type == 'avg':
                    
                    updates_w.append((vels[k], vel_i_next))
                    updates_w.append((param_i, param_i + vel_i_next))
                    
                k=k+1
                

        else:
            
            k=0
            
            for param_i, grad_i, weight_type in \
                    zip(params, grads, weight_types):
                    
            
                if weight_type == 'W':
                    
                    if worker_type == 'cdd':
                        
                        update =          - lr * grad_i - eta * lr * param_i
                        
                    elif worker_type == 'avg':
                        
                        update =  param_i - lr * grad_i - eta * lr * param_i

                elif weight_type == 'b':
                    
                    if worker_type == 'cdd':
                    
                        update =         - 2 * lr * grad_i
                        
                    elif worker_type == 'avg':
                        
                        update = param_i - 2 * lr * grad_i
                        
                if worker_type == 'cdd':
                    
                    updates_v.append((vels[k], update))
                    updates_dv.append((param_i, param_i + vels2[k]))
                    
                elif worker_type == 'avg':
                    
                    # updates_w.append((vel_i, - 2 * lr * grad_i))
                    updates_w.append((param_i, update))
                    
                    
                k=k+1
                
        if worker_type == 'cdd':
        
            self.model.update_dict = [updates_v, updates_dv]
        
        elif worker_type == 'avg':
            
            self.model.update_dict = [updates_w]
            
            
            
    def prepare_val_fn(self):
        
        self.model.compile_val()
        
                                    
    def prepare_recorder(self):
        
        from base.recorder import Recorder
        
        self.recorder = Recorder(self.config)
                                    
    def prepare_iterator(self):
        
        worker_type=self.config['worker_type']
        
        from base.iterator import P_iter
        
        if worker_type == 'cdd':
            
            def cdd_iter_fn(subb_ind):
                self.model.descent_vel()
                cost, error = self.model.get_vel(subb_ind)
                return cost, error
            
            self.train_iterator = P_iter(self.config, self.model, \
                                    self.data[0], self.data[1],  'train', cdd_iter_fn)
                                    
        elif worker_type == 'avg':
            
            self.train_iterator = P_iter(self.config, self.model, \
                                    self.data[0], self.data[1],  'train', self.model.train)
                                    
        self.val_iterator = P_iter(self.config, self.model, \
                                    self.data[2], self.data[3], 'val', self.model.val)
                                    
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
        
        self.model.adjust_lr(self.epoch)
        
        new_lr = self.model.shared_lr.get_value()
        
        if self.config['worker_type'] == 'avg':
            self.model.shared_lr.set_value(np.float32(new_lr*self.size))
        else:
            pass
    
        if self.verbose: 
            print 'Learning rate now: %.10f' % \
                    np.float32(self.model.shared_lr.get_value())
        
    def run(self):
        
        # override PTWorker class method
        
        print 'worker started'
        
        if self.config['resume_train'] == True:
            self.epoch = self.config['load_epoch']
            self.load_model(self.epoch)
        
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
                self.epoch+=1# epoch starts from 1, not 0. 0 means training has not started.
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
    try:
        device = sys.argv[1]
    except IndexError:
        # raise ValueError('Need to specify a GPU device')
        import os
        gpuid = str(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        device = 'gpu'+gpuid
        
    worker = BSP_PTWorker(config=config, device=device)
    
    worker.run()

