'''
Training iterator

'''   

import time

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
        self.verbose = self.config['verbose'] == 0

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
            
            filenames_arr = np.array(raw_filenames)
            
            if self.config['random'] and self.config['shuffle']:
                time_seed = int(time.time())*int(self.config['worker_id'])%1000
                np.random.seed(time_seed)
            
                indices = np.random.permutation(filenames_arr.shape[0])
                filenames= filenames_arr[indices]
            else:
                filenames = filenames_arr

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
                
        if self.config['sync_rule'] == 'BSP':
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
        from mpi4py import MPI
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
   
# TODO make sure EASGD process are not started simultaneously
                
        
        
    
        
            
        
                

        
