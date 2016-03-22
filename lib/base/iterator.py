'''
Training iterator

'''   


from mpi4py import MPI
import numpy as np
import time

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
        self.last_one = False
        self.n_subb = self.config['n_subb']
        self.subb = 0

        self.mode = mode
        self.verbose = self.config['verbose']
        self.monitor = self.config['monitor_grad']

        if self.mode == 'train':
            
            if self.config['train_mode'] == 'cdd':
                
                def train_function(subb_ind):
                    model.descent_vel()
                    cost, error = model.get_vel(subb_ind)
                    return cost, error

                self.function = train_function #model.get_vel
            elif self.config['train_mode'] == 'avg':
                self.function = model.train
                
            if self.config['monitor_grad']:
                self.get_norm = model.get_norm
                
        elif self.mode == 'val':
        	self.function = model.val

    def __iter__(self):
        
        return self
        
    def shuffle_data(self):
        
        raw_filenames, raw_labels = self.raw_filenames, self.raw_labels
        
        if self.mode=='train':
            
            filenames_arr = np.array(raw_filenames)
            
            if self.config['shuffle']:
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
        
    def reset(self):
        
        self.current = 0
        self.subb=0
        
        self.icomm.isend('stop',dest=0,tag=40)
        
    def stop_load(self):
        
        # to stop the paraloading process
        
        self.icomm.isend('stop',dest=0,tag=40)
        
        self.icomm.isend('stop',dest=0,tag=40)

        
    def next(self, recorder, count):	
        
        if self.subb == 0:
            
            if self.current == 0:
        
                self.shuffle_data()
        
                # 3.0 give mode signal to adjust loading mode between train and val
        
                self.icomm.isend(self.mode,dest=0,tag=40)
        
                # 3.1 give load signal to load the very first file
        
                self.icomm.isend(self.filenames[self.current],dest=0,tag=40)
        
            if self.current == self.len - 1:
                self.last_one = True
                # Only to get the last copy_finished signal from load
                self.icomm.isend(self.filenames[self.current],dest=0,tag=40) 
            else:
                self.last_one = False
                # 4. give preload signal to load next file
                self.icomm.isend(self.filenames[self.current+1],dest=0,tag=40)

            if self.mode == 'train': recorder.start()
        
            # 5. wait for the batch to be loaded into shared_x
            msg = self.icomm.recv(source=0,tag=55) #
            assert msg == 'copy_finished'
             
            self.shared_y.set_value(self.labels[self.current])
        
            if self.mode == 'train': recorder.end('wait')
        
        if self.mode == 'train':
            recorder.start()
            
            cost,error= self.function(self.subb)
            
            if self.verbose: 
                print count, cost, error
                if self.config['monitor_grad']: 
                    print np.array(self.get_norm(self.subb))
                    #print [np.int(np.log10(i)) for i in np.array(self.get_norm(self.subb))]
                
            recorder.train_error(count, cost, error)
            recorder.end('calc')

        else:
            cost,error,error_top5 = self.function(self.subb)
            recorder.val_error(count, cost, error, error_top5)
            
        if (self.subb+1)//self.n_subb == 1:
            
            if self.last_one == False:
                self.current+=1
            else:
                self.current=0
            
            self.subb=0
        else:
            self.subb=self.subb+1
    
        
            
        
                

        
