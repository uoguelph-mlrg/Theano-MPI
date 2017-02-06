from __future__ import absolute_import

import numpy as np

from keras import backend as K
nb_classes = 10

class Cifar10_data():
    
    def __init__(self, verbose):
        
        # data hyperparams
        
        # self.data_path  = '/scratch/hma02/data/cifar10/cifar-10-batches-py/'
        
        self.channels = 3
        self.width =32
        self.height =32

        self.n_class = 10
        
        self.get_data()
        
        self.verbose = verbose
        
        self.batched=False
        self.shuffled=False
        self.sharded=False
        
    def get_data(self):
        
        from keras.datasets import cifar10
        
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        from keras.utils import np_utils
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
    
        
        img_mean = X_train.mean(axis=0)[np.newaxis,:,:,:]

        N = X_train.shape[0]
        perms = np.random.permutation(N)
        self.X_train   = X_train[perms,:]
        self.Y_train = Y_train[perms,:]
        self.X_test = X_test
        self.Y_test = Y_test
        
        self.rawdata=[self.X_train, self.Y_train, self.X_test, self.Y_test, img_mean]
        
        
        
    def batch_data(self, model, batch_size):
    
        if self.batched==False:

            x, y, sample_weights = model._standardize_user_data(
                        self.X_train, self.Y_train,
                        sample_weight=None,
                        class_weight=None,
                        check_batch_axis=False,
                        batch_size=batch_size)
        
            val_x, val_y, val_sample_weights = model._standardize_user_data(
                            self.X_test, self.Y_test,
                            sample_weight=None,
                            check_batch_axis=False,
                            batch_size=batch_size)
        
               
        
            ins = x + y + sample_weights
            val_ins = val_x + val_y + val_sample_weights
        
            if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
                ins+=[1.]
                val_ins+=[0.]
                
            from keras.engine.training import slice_X
            
            self.n_batch_train = ins[0].shape[0]/batch_size
            self.train_batches = []
            index_arr = range(ins[0].shape[0])
            for batch_index in range(self.n_batch_train):
            
                batch_ids = index_arr[batch_index * batch_size:
                                    (batch_index+1)*batch_size]
                                        
                if isinstance(ins[-1], float):
                    # do not slice the training phase flag
                    ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
                else:
                    ins_batch = slice_X(ins, batch_ids)
                                        
                self.train_batches.append(ins_batch)
            
            
            self.n_batch_val = val_ins[0].shape[0]/batch_size
            self.val_batches = []
            index_arr = range(val_ins[0].shape[0])
            for batch_index in range(self.n_batch_val):
            
                batch_ids = index_arr[batch_index * batch_size:
                                    (batch_index+1)*batch_size]
                                        
                if isinstance(val_ins[-1], float):
                    # do not slice the training phase flag
                    ins_batch = slice_X(val_ins[:-1], batch_ids) + [val_ins[-1]]
                else:
                    ins_batch = slice_X(val_ins, batch_ids)
                                        
                self.val_batches.append(ins_batch)
                
            self.batched=True

    def shuffle_data(self):
        '''
        shuffle training data
        '''
        
        if self.shuffled == False:
            
            import time, os
            time_seed = int(time.time())*int(os.getpid())%1000
            np.random.seed(time_seed)

            indices = np.random.permutation(self.n_batch_train)

            # 2. shuffle batches based on indices
            batches = []

            for index in indices:
                batches.append(self.train_batches[index])

            self.train_batches = batches

            if self.verbose: print 'training data shuffled'

            self.shuffled=True
        
        
        
    def shard_data(self, file_batch_size, rank, size):
        
        '''
        shard validation data
        '''
        
        val_batches = self.val_batches
        # make divisible
        
        if self.sharded == False:
            
            # sharding
            
            self.val_batches_shard = val_batches[rank::size]
        
            self.n_batch_val = len(self.val_batches_shard)
            
            if self.verbose: print 'validation data sharded'
            
            self.sharded=True
            
            
        