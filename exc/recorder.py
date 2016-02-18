'''
Recorder class for recording training, validation, lr and time info during training

'''

import time
import numpy as np

class Recorder(object):
    
    '''
    training time and info recorder
    
    '''
    
    def __init__(self, config):

        self.config = config
        self.t_start = None
        self.t_end = None

        self.info_dict = {}

        self.info_dict['train_info'] = []
        self.train_info = {}
        self.train_info['cost'] = []
        self.train_info['error'] = []

        self.info_dict['val_info'] = []
        self.val_info = {}
        self.val_info['cost'] = []
        self.val_info['error'] = []
        self.val_info['error_top5'] = []

        self.info_dict['epoch_time'] = []

        self.info_dict['all_time'] = []
        self.all_time = {}
        self.all_time['calc'] = []
        self.all_time['comm'] = []
        self.all_time['wait'] = []        

        self.info_dict['lr'] = []

        self.epoch_time = None
        self.verbose = self.config['rank'] == 0
	
    def start(self):
	
    	self.t_start = time.time()
	
    def end(self,mode):
	
    	self.all_time[mode].append( time.time() - self.t_start )
	
    	self.t_start = None
		
    def start_epoch(self):
	
    	self.epoch_time = time.time()
	
    def end_epoch(self, count, epoch):

        duration = time.time() - self.epoch_time

        self.info_dict['epoch_time'].append([count, duration])

        if self.verbose: 
            print 'epoch %d took %.2f h' % (epoch, duration/3600.0)
            print ''

        self.epoch_time = None

    def train_error(self,count, cost, error):

        self.train_info['cost'].append(cost)
        self.train_info['error'].append(error)

    def val_error(self,count, cost, error, error_top5):

        self.val_info['cost'].append(cost)
        self.val_info['error'].append(error)
        self.val_info['error_top5'].append(error_top5)

    def print_train_info(self, count):

        printFreq = self.config['print_info_every']   \
                                /self.config['file_batch_size']
        if (count) % printFreq ==0:

            #print train info
            if self.verbose: print ''
            cost = sum(self.train_info['cost'])/len(self.train_info['cost'])
            error = sum(self.train_info['error'])/len(self.train_info['error'])

            self.info_dict['train_info'].append([count, cost, error])

            if self.verbose: print '%d %.4f %.4f'% (count, cost, error)

            self.train_info['cost'] = []
            self.train_info['error'] = []

            # print time info

            calc = sum(self.all_time['calc'])
            comm = sum(self.all_time['comm'])
            wait = sum(self.all_time['wait'])
            t_all = calc + comm + wait

            self.info_dict['all_time'].append([count, t_all, calc, comm, wait])

            if self.verbose:
                print 'time per %d images: %.2f (train %.2f comm %.2f wait %.2f)' % (self.config['print_info_every'], t_all, calc, comm, wait)
             
            self.all_time['calc'] = []
            self.all_time['comm'] = []
            self.all_time['wait'] = []
            
    def gather_val_info(self):
        
        self.val_info['cost'] = np.array(
        self.config['comm'].allgather(
                                     self.val_info['cost'])
                                     ).flatten().tolist()
        self.val_info['error'] = np.array(
        self.config['comm'].allgather(
                                     self.val_info['error'])
                                     ).flatten().tolist()
        self.val_info['error_top5'] = np.array(
        self.config['comm'].allgather(
                                     self.val_info['error_top5'])
                                     ).flatten().tolist()

    def print_val_info(self, count):
    
        cost = sum(self.val_info['cost'])/len(self.val_info['cost'])
        error = sum(self.val_info['error'])/len(self.val_info['error'])
        error_top5 = sum(self.val_info['error_top5']) \
                                        /len(self.val_info['error_top5'])
    
        self.info_dict['val_info'].append([count, cost, error, error_top5])
    
        if self.verbose:
            print 'validation cost:%.4f' % cost
            print 'validation error:%.4f' % error
            print 'validation top_5_error:%.4f' % error_top5
    
        self.val_info['cost'] = []
        self.val_info['error'] = []
        self.val_info['error_top5'] = []
    
    def save(self, count,lr, filepath = '../run/inforec/inforec.pkl'):

        self.info_dict['lr'].append([count,lr])
        
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.info_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
          

        #np.save('../run/inforec/inforec.npy' , self.info_dict.items())
        
    def load(self, filepath = '../run/inforec/inforec.pkl'):
        
        import pickle
        
        with open(filepath, 'rb') as f:

            self.info_dict = pickle.load(f)
            

    
        
        
        
        
    
    
    
	
												 
		
		
    
        
        
    
        
            
        
                

        
