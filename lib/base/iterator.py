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
        
        self.paraload = self.config['para_load']

        self.shared_y = model.shared_y
        self.raw_filenames = filenames
        self.raw_labels = labels
        self.filenames = None # the filename list
        self.labels = None
        
        if config['para_load']:
            
            self.icomm = self.config['icomm']
            
        else:
            
            self.shared_x = model.shared_x
            self.img_mean = model.img_mean
            
            from helper_funcs import get_rand3d
            from proc_load_mpi import crop_and_mirror
            import hickle as hkl
            self.hkl = hkl
            self.get_rand3d = get_rand3d
            self.crop_and_mirror = crop_and_mirror
        
        self.len = len(self.raw_filenames)

        self.current = 0 # current filename pointer in the filename list
        self.last_one = False # if pointer is pointing to the last filename in the list
        self.n_subb = self.config['n_subb']
        self.subb = 0 # sub-batch index

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
        
        # this function need to be called at the begining of an epoch
        
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
                indices = np.arange(filenames_arr.shape[0])

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
        
        if self.subb == 0: # load the whole file into shared_x when loading sub-batch 0 of each file.
            
            if self.paraload:
            
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
                
            else:
                
                if self.current == 0:
        
                    self.shuffle_data()
                
                if self.mode == 'train': recorder.start()
                
                data = self.hkl.load(str(self.filenames[self.current])) - self.img_mean
        
                rand_arr = self.get_rand3d(self.config, self.mode)

                data = self.crop_and_mirror(data, rand_arr, \
                                        flag_batch=self.config['batch_crop_mirror'], \
                                        cropsize = self.config['input_width'])
                self.shared_x.set_value(data)
                self.shared_y.set_value(self.labels[self.current])
                
                if self.mode == 'train': recorder.end('wait')
                
        
        if self.mode == 'train':
            recorder.start()
            
            cost,error= self.function(self.subb)
            
            if self.verbose: 
                #print count+self.config['rank'], cost, error
                #if count+self.config['rank']>45: exit(0)
                if self.config['monitor_grad']: 
                    print np.array(self.get_norm(self.subb))
                    #print [np.int(np.log10(i)) for i in np.array(self.get_norm(self.subb))]
                
            recorder.train_error(count, cost, error)
            recorder.end('calc')

        else:
            cost,error,error_top5 = self.function(self.subb)
            recorder.val_error(count, cost, error, error_top5)
            
        if (self.subb+1)//self.n_subb == 1: # test if next sub-batch is in another file
            
            if self.last_one == False:
                self.current+=1
            else:
                self.current=0
            
            self.subb=0
        else:
            self.subb=self.subb+1
    
        
            
        
                

        
