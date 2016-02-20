'''
Optimizer and training iterator

'''

import sys  
import os       
from mpi4py import MPI 
import socket
import time
import yaml
from lib.helper_funcs import unpack_configs, extend_data, \
                    save_weights, load_weights, save_momentums, load_momentums
import numpy as np

class P_iter(object):

    '''
    training iterator responsible for one iteration

    '''

    def __init__(self, config, model, filenames,labels,mode='train'):

        self.config = config
        self.icomm = self.config['icomm']

        self.shared_y = model.shared_y
        self.raw_filenames = filenames
        self.raw_labels = labels
        self.filenames = None
        self.labels = None
        
        self.len = len(self.raw_filenames)

        self.current = 0

        self.mode = mode
        self.verbose = self.config['rank'] == 0

        if self.mode == 'train':
            if self.config['train_mode'] == 'cdd':
                def train_function():
                    model.descent_vel()
                    cost, error = model.get_vel()
                    return cost, error

                self.function = train_function
            elif self.config['train_mode'] == 'avg':
                self.function = model.train
        elif self.mode == 'val':
        	self.function = model.val
            
        
        
    def __iter__(self):
        
        return self
        
    def shuffle_data(self):
        
        raw_filenames, raw_labels = self.raw_filenames, self.raw_labels
        
        if self.mode=='train':
            
            time_seed = int(time.time())*int(self.config['worker_id'])%1000
            np.random.seed(time_seed)
        
            filenames_arr = np.array(raw_filenames)
            indices = np.random.permutation(filenames_arr.shape[0])
            filenames= filenames_arr[indices]

            y=[]
            for index in range(len(indices)):
                batch_label = raw_labels[(index) \
                                * self.config['file_batch_size']: \
    							(index + 1) * self.config['file_batch_size']]
		
                y.append(batch_label)
    
            labels=[]
      
            for index in indices:
                labels.append(y[index])
        
            if self.verbose: print 'training data shuffled'

        else:
        
            filenames = np.array(raw_filenames)

            labels=[]
            for index in range(filenames.shape[0]):
                batch_label = raw_labels[index \
                                * self.config['file_batch_size']: \
    							(index + 1) * self.config['file_batch_size']]
			
                labels.append(batch_label)
                
        if self.config['syncrule'] == 'BSP':
            # localize data

            filenames = filenames[self.config['rank']::self.config['size']]
            labels = labels[self.config['rank']::self.config['size']]
            
            self.len = len(filenames)
            
        self.filenames = filenames
        self.labels = labels

    def next(self, recorder, count):	
        
        if self.current == 0:
            
            self.shuffle_data()
            
            # 3. send train mode signal
            self.icomm.send(self.mode,dest=0,tag=43) 
            
            # 4. send the shuffled filename list to parallel loading process
            self.icomm.send(self.filenames,dest=0,tag=40) 
                
            # 5. give preload signal to load the very first file
            
            self.icomm.isend("calc_finished",dest=0,tag=35) 
            
        if self.current == self.len - 1:
            last_one = True
        else:
            last_one = False

		
        if self.mode == 'train': recorder.start()
        # 6. wait for the batch to be loaded into shared_x
        msg = self.icomm.recv(source=MPI.ANY_SOURCE,tag=55) #
        assert msg == 'copy_finished'
             
        self.shared_y.set_value(self.labels[self.current])
        
        if self.mode == 'train': recorder.end('wait')
        

        if self.mode == 'train':
            recorder.start()
            cost,error= self.function()
            recorder.train_error(count, cost, error)
            recorder.end('calc')

        else:
            cost,error,error_top5 = self.function()
            recorder.val_error(count, cost, error, error_top5)


        if last_one == False:
            # 5. give signal to load another file unless it's the last file now
            self.icomm.isend("calc_finished",dest=0,tag=35) 
            self.current+=1

        else:
            self.current=0

            
        return recorder


class Optimizer(object):
    
    '''
    training and optimizing models
    
    '''
    
    def __init__(self, config, model, data, recorder):

        self.model = model
        self.config = config
        self.recorder = recorder

        from exchanger import Exchanger

        self.exchanger = Exchanger(config, self.model)

        self.mode = None
        self.epoch = 0
        self.count = 0

        self.train_iterator = P_iter(config, self.model, \
                                    data[0], data[1],  'train')
        self.val_iterator = P_iter(config, self.model, \
                                    data[2], data[3], 'val')

        if self.config['resume_train'] == True:
            epoch = self.config['load_epochs']
        else:
            epoch = 0

        self.train_len = len(data[0])

        self.val_len = len(data[2])
        
        self.verbose = self.config['rank'] == 0
    
    def train(self):
        
        self.config['comm'].Barrier()
        
        if self.verbose: 
            print '\nNow training'
        
        for i in range(0,self.train_len,self.config['size']):
        
            self.recorder = self.train_iterator.next(self.recorder,self.count)

            self.recorder.start()
            
            self.exchanger.exchange()
            
            self.recorder.end('comm')

            self.count += self.config['size']

            self.recorder.print_train_info(self.count)
        
        
        
    def val(self):
        
        self.config['comm'].Barrier()
        
        if self.verbose: 
            print '\nNow validating'
        
        self.model.set_dropout_off()
        
        for i in range(0,self.val_len,self.config['size']):
        
            self.recorder = self.val_iterator.next(self.recorder,self.count)
            
            print '.',
            
        if self.config['syncrule'] == 'BSP':

            self.recorder.gather_val_info()
        
        self.recorder.print_val_info(self.count)
        
        self.model.set_dropout_on()
               
    def load_model(self, load_epoch):
        
        layers = self.model.layers
        path = self.config['load_path']
        learning_rate = self.model.lr
        vels = self.model.vels
        
        load_weights(layers, path, load_epoch)
        if learning_rate != None: 
            learning_rate.set_value(np.load(os.path.join(path, 
                      'lr_' + str(epoch) + '.npy')))
        if vels != None:
            load_momentums(vels, path, load_epoch)
            
        if self.verbose: 
            print '\nweights and momentums loaded from epoch %d' % load_epoch
            
    def save_model(self): 
      
        layers = self.model.layers
        path = self.config['weights_dir']
        vels = self.model.vels  
        
        save_weights(layers, path, self.epoch)
        np.save(path + 'lr_' + str(self.epoch) + \
                        '.npy', self.model.lr.get_value())
        save_momentums(vels, 
                       self.config['weights_dir'], self.epoch)
        if self.verbose:
            print '\nweights and momentums saved at epoch %d' % self.epoch
        
    def start(self):

        if self.config['syncrule'] == 'BSP':

            while True:

                self.epoch+=1
                
                self.recorder.start_epoch()

                self.train()
                self.val()
                self.model.adjust_lr(self.epoch)

                self.recorder.end_epoch(self.count, self.epoch)

                self.recorder.save(self.count, self.model.lr.get_value())
                
                if self.epoch % self.config['snapshot_freq'] == 0:
                    if self.config['rank'] ==0 :
                        self.save_model()

                if self.epoch >= self.config['n_epochs']: 
                    print '\noptimization finished'
                    break
    		
    		
        elif self.config['syncrule'] == 'EASGD':

            while True:

                self.mode = worker.send_req('next')

                if self.mode == 'train':
                    if start_time==None:
                        start_time = time.time()

                    self.train()
	
                if self.mode == 'val':

                    if valid_sync:
                        worker.copy_to_local()

                    self.val()

                    if start_time!=None:

                        self.epoch+=1

                        start_time=None

    	
        
# TODO make sure EASGD process are not started simultaneously
                
        
        
    
        
            
        
                

        
