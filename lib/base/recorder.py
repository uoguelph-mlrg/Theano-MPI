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
        self.verbose = self.config['verbose']
	
    def start(self):
	
    	self.t_start = time.time()
	
    def end(self,mode):
	
    	self.all_time[mode].append( time.time() - self.t_start )
	
    	self.t_start = None
		
    def start_epoch(self):
	
    	self.epoch_time = time.time()
	
    def end_epoch(self, count, uepoch):

        duration = time.time() - self.epoch_time

        self.info_dict['epoch_time'].append([count, duration])

        if self.verbose: 
            print 'global epoch %d took %.2f h' % (uepoch, duration/3600.0)
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
            cost =sum(self.train_info['cost'])/len(self.train_info['cost'])
            error =sum(self.train_info['error'])/len(self.train_info['error'])

            self.info_dict['train_info'].append([count, cost, error])

            if self.verbose: print '%d %f %f'% (count, cost, error)

            self.train_info['cost'][:] = []
            self.train_info['error'][:] = []

            # print time info

            calc = sum(self.all_time['calc'])
            comm = sum(self.all_time['comm'])
            wait = sum(self.all_time['wait'])
            t_all = calc + comm + wait

            self.info_dict['all_time'].append([count, t_all, calc, comm, wait])

            if self.verbose:
                print 'time per %d images: %.2f (train %.2f comm %.2f wait %.2f)' % \
                            (self.config['print_info_every'], t_all, calc, comm, wait)
             
            self.all_time['calc'][:] = []
            self.all_time['comm'][:] = []
            self.all_time['wait'][:] = []
            
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
            print '\nvalidation cost:%.4f' % cost
            print 'validation error:%.4f' % error
            print 'validation top_5_error:%.4f' % error_top5
    
        self.val_info['cost'][:] = []
        self.val_info['error'][:] = []
        self.val_info['error_top5'][:] = []
    
    def get_latest_val_info(self):
        
        try:
            latest = self.info_dict['val_info'][-1]
        except:
            latest = None
            
        return latest
    
    def save(self, count,lr, filepath = 'inforec/inforec.pkl'):

        self.info_dict['lr'].append([count,lr])
        
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.info_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
          

        #np.save('../run/inforec/inforec.npy' , self.info_dict.items())
        
    def load(self, filepath = '../run/inforec/inforec.pkl'):
        
        import pickle
        
        with open(filepath, 'rb') as f:

            self.info_dict = pickle.load(f)
            
    def show(self, label='', color_id = 0, show=True):
        
        import matplotlib.pyplot as plt

        from matplotlib.font_manager import FontProperties

        fontP = FontProperties()
        fontP.set_size('small')
        color = ['-r','-b','-m','-g', '--r','--b','--m']
        
        count_train, train_loss, train_error  = np.transpose(self.info_dict['train_info'])
        train_loss = train_loss[:int(train_loss.shape[0]/250) * 250]
        train_error = train_error[:int(train_error.shape[0]/250) * 250]
        train_loss = np.mean(train_loss.reshape(-1, 250), axis=1)
        train_error = np.mean(train_error.reshape(-1, 250), axis=1)
        
        count_val, val_loss, val_error, val_error_top5  = np.transpose(self.info_dict['val_info'])
        count_t, t_all, t_calc, t_comm, t_wait = np.transpose(self.info_dict['all_time'])
        t_all = t_all[:int(t_all.shape[0]/250) * 250]
        t_calc = t_calc[:int(t_calc.shape[0]/250) * 250]
        t_comm = t_comm[:int(t_comm.shape[0]/250) * 250]
        t_wait = t_wait[:int(t_wait.shape[0]/250) * 250]
        
        t_all = np.mean(t_all.reshape(-1, 250), axis=1)
        t_calc = np.mean(t_calc.reshape(-1, 250), axis=1)
        t_comm = np.mean(t_comm.reshape(-1, 250), axis=1)
        t_wait = np.mean(t_wait.reshape(-1, 250), axis=1)
        
        count_epoch, t_epoch = np.transpose(self.info_dict['epoch_time'])
        count_lr , lr = np.transpose(self.info_dict['lr'])

        # train error
        fig = plt.figure(1, figsize=(5,8))
        fig.subplots_adjust(left = 0.15, bottom = 0.07,
                            right = 0.94, top = 0.94,
                            hspace = 0.14)

        ax = plt.subplot(211) # one record per 40 iterations , total 250 recordings for 10008 iterations in an epoch
        ax.plot(train_loss, color[0+color_id], label=self.config['name']+ label)
        ax.legend(loc='upper right',prop=fontP)
        #ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        

        ax = plt.subplot(212)
        ax.plot(train_error, color[0+color_id], label=self.config['name']+ label)
        ax.legend(loc='upper right',prop=fontP)
        ax.set_xlabel('epoch')
        ax.set_ylabel('error')
        
        
        plt.suptitle('training info')
        fig.savefig('train.png',format='png')

        # val error
        fig = plt.figure(2, figsize=(5,8))
        fig.subplots_adjust(left = 0.15, bottom = 0.07,
                            right = 0.94, top = 0.94,
                            hspace = 0.14)

        ax = plt.subplot(311) # one record per epoch
        ax.plot(val_loss, color[0+color_id], label=self.config['name']+ label)
        ax.legend(loc='upper right',prop=fontP)
        #ax.set_xlabel('epoch')
        ax.set_ylabel('loss')

        ax = plt.subplot(312)
        ax.plot(val_error, color[0+color_id], label=self.config['name']+ label)
        ax.legend(loc='upper right',prop=fontP)
        #ax.set_xlabel('epoch')
        ax.set_ylabel('error')

        ax = plt.subplot(313)
        ax.plot(val_error_top5, color[0+color_id], label=self.config['name']+ label)
        ax.legend(loc='upper right',prop=fontP)
        ax.set_xlabel('epoch')
        ax.set_ylabel('top5 error')
        
        plt.suptitle('validation info')
        fig.savefig('val.png',format='png')

        # time
        fig = plt.figure(3)

        ax = plt.subplot(411) # one record per 40 iterations ,
        ax.plot(t_all, color[0+color_id], label='all_time')
        #ax.set_xlabel('epoch')
        ax.set_ylabel('overall')

        ax = plt.subplot(412)
        ax.plot(t_calc, color[0+color_id], label='train_time')
        #ax.set_xlabel('epoch')
        ax.set_ylabel('calc')

        ax = plt.subplot(413)
        ax.plot(t_comm, color[0+color_id], label='comm_time')
        #ax.set_xlabel('epoch')
        ax.set_ylabel('comm')

        ax = plt.subplot(414)
        ax.plot(t_wait, color[0+color_id], label='wait_time')
        ax.set_xlabel('epoch')
        ax.set_ylabel('wait')
        
        plt.suptitle('time per 5120 images')


        # epoch time
        fig = plt.figure(4) # in hour

        plt.plot(t_epoch/3600.0, color[0+color_id], label='epoch_time')
        plt.xlabel('epoch')
        plt.ylabel('time per epoch')

        plt.ylim([0,3])
        
        plt.suptitle('epoch time')
        
        
        # learning rate
        
        fig = plt.figure(5)

        plt.plot(lr, color[0+color_id], label='lr')
        plt.xlabel('epoch')
        plt.ylabel('learning rate')
        
        plt.ylim([0,0.1])
        
        plt.suptitle('learning rate')
        

        if show: plt.show()
            

    
        
        
        
        
    
    
    
	
												 
		
		
    
        
        
    
        
            
        
                

        
