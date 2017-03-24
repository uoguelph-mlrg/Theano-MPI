from __future__ import absolute_import

import numpy as np

def iterate_minibatches(inputs, targets, shuffle=False,
                        forever=False):
                        
    '''
    source:
         https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066#file-wgan_mnist-py                    
    '''
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            yield inputs[idx], targets[idx]
        if not forever:
            break
            
class Cifar10_data():
    
    def __init__(self, verbose):
        
        # data hyperparams
        
        self.data_path  = '/mnt/data/hma02/data/cifar10/cifar-10-batches-py/'
        
        self.channels = 3
        self.width =32
        self.height =32

        self.n_class = 10
        
        self.get_data()
        
        self.verbose = verbose
        
        self.batched=False
        self.extended=False

        
    def get_data(self):

        path = self.data_path
        '''processes the raw downloaded cifar10 dataset, and returns test/val/train set'''
        
        from theanompi.models.data.utils import unpickle

        d1 = unpickle(path+'data_batch_1')
        d2 = unpickle(path+'data_batch_2')
        d3 = unpickle(path+'data_batch_3')
        d4 = unpickle(path+'data_batch_4')
        d5 = unpickle(path+'data_batch_5')
        dt  = unpickle(path+'test_batch')

        
        d = np.concatenate((d1['data'], d2['data']), axis=0)
        d = np.concatenate((d,  d3['data']), axis=0)
        d = np.concatenate((d,  d4['data']), axis=0)
        img = np.concatenate((d,  d5['data']), axis=0)
                    
        img=img.reshape([img.shape[0], 3, 32, 32]) # needs to be in c01b
        
        img_mean = img.mean(axis=0)[:,:,:,np.newaxis]
        
        
        l = np.concatenate((d1['labels'], d2['labels']), axis=0)
        l = np.concatenate((l, d3['labels']), axis=0)
        l = np.concatenate((l, d4['labels']), axis=0)
        labels = np.concatenate((l, d5['labels']), axis=0)
        
        test_set = [dt['data'], np.asarray(dt['labels'], dtype='float32')]

        
        N = img.shape[0]
        perms = np.random.permutation(N)
        img   = img[perms,:]
        labels = labels[perms]
        
        train_data, train_labels = img[0:int(N*0.8)] , labels[0:int(N*0.8)]
        val_data, val_labels = img[int(N*0.8):] , labels[int(N*0.8):]
        
        self.rawdata=[train_data, train_labels, val_data, val_labels, img_mean]
        
        
        
    def batch_data(self, file_batch_size):
    
        if self.batched==False:
            
            self.n_batch_train = self.rawdata[0].shape[0]/file_batch_size
        
            self.train_img, self.train_labels=[],[]

            raw_img = self.rawdata[0]
            raw_labels = self.rawdata[1]
    
            for index in range(self.n_batch_train):
                batch_img =  raw_img[(index) \
                                * file_batch_size: \
                				(index + 1) * file_batch_size]
                batch_label = raw_labels[(index) \
                                * file_batch_size: \
                				(index + 1) * file_batch_size]

                self.train_img.append(batch_img)
                self.train_labels.append(batch_label)
            
            
            
            self.n_batch_val = self.rawdata[2].shape[0]/file_batch_size
        
            self.val_img, self.val_labels=[],[]

            raw_img = self.rawdata[2]
            raw_labels = self.rawdata[3]
    
            for index in range(self.n_batch_val):
                batch_img =  raw_img[(index) \
                                * file_batch_size: \
                				(index + 1) * file_batch_size]
                batch_label = raw_labels[(index) \
                                * file_batch_size: \
                				(index + 1) * file_batch_size]

                self.val_img.append(batch_img)
                self.val_labels.append(batch_label)
                
            self.batched=True
            
            self.batches_train = iterate_minibatches(self.train_img, self.train_labels, shuffle=True,
                                                  forever=True)  
            
            self.batches_val = iterate_minibatches(self.val_img, self.val_labels, shuffle=True,
                                                  forever=True)
    
    def extend_data(self, rank, size):

        if self.extended == False:
            if self.batched == False:
                raise RuntimError('extend_data needs to be after batch_data')

            # make divisible
            from theanompi.models.data.utils import extend_data
            self.train_img_ext, self.train_labels_ext = extend_data(rank, size, self.train_img, self.train_labels)
            self.val_img_ext, self.val_labels_ext = extend_data(rank, size, self.val_img, self.val_labels)
    
            self.n_batch_train = len(self.train_img_ext)
            self.n_batch_val = len(self.val_img_ext)
    
            self.extended=True
 
 
    def shuffle_data(self, mode, common_seed=1234):
        
        if self.extended == False:
            raise RuntimError('shuffle_data needs to be after extend_data')
    
        # To be called at the begining of an epoch for shuffling the order of training data

        # 312 = 40000 / 128 
        if mode=='train':
        # 1. generate random indices 
            
            import time, os
            time_seed = int(time.time())*int(os.getpid())%1000
            np.random.seed(time_seed)
            
            self.n_batch_train = len(self.train_img_ext)
            
            indices = np.random.permutation(self.n_batch_train)

            # 2. shuffle batches based on indices
            img = []
            labels=[]

            for index in indices:
                img.append(self.train_img_ext[index])
                labels.append(self.train_labels_ext[index])
            
            self.train_img_shuffle = img
            self.train_labels_shuffle = labels
        
            if self.verbose: print 'training data shuffled', indices
            
        elif mode=='val':
            
            self.val_img_shuffle = self.val_img_ext
            self.val_labels_shuffle = self.val_labels_ext

    
    def shard_data(self, mode, rank, size):
        
        if mode=='train':
            
            # sharding
            self.train_img_shard, self.train_labels_shard = \
                    self.train_img_shuffle[rank::size], self.train_labels_shuffle[rank::size]
            self.n_batch_train = len(self.train_img_shard)
            
            if self.verbose: print 'training data sharded', self.n_batch_train
        
        elif mode=='val':
            self.val_img_shard, self.val_labels_shard = \
                    self.val_img_shuffle[rank::size], self.val_labels_shuffle[rank::size]
            self.n_batch_val = len(self.val_img_shard)
            
            if self.verbose: print 'validation data sharded', self.n_batch_val
            
            
        
        
        