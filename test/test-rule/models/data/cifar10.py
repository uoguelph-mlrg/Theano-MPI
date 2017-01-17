import numpy as np

class Cifar10_data():
    
    def __init__(self, verbose):
        
        # data hyperparams
        
        self.data_path  = '/scratch/hma02/data/cifar10/cifar-10-batches-py/'
        
        self.channels = 3
        self.width =32
        self.height =32

        self.n_class = 10
        
        self.get_data()
        
        self.verbose = verbose
        
        self.batched=False
        self.shuffled=False
        self.extended=False
        self.sharded=False
        
    def get_data(self):

        path = self.data_path
        '''processes the raw downloaded cifar10 dataset, and returns test/val/train set'''
        
        from helper_funcs import unpickle

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
    
    
    def shuffle_data(self):
    
        # To be called at the begining of an epoch for shuffling the order of training data

        # 312 = 40000 / 128 
    
        # 1. generate random indices 
        if self.shuffled == False:
            
            import time, os
            time_seed = int(time.time())*int(os.getpid())%1000
            np.random.seed(time_seed)

            indices = np.random.permutation(self.n_batch_train)

            # 2. shuffle batches based on indices
            img = []
            labels=[]

            for index in indices:
                img.append(self.train_img[index])
                labels.append(self.train_labels[index])
            
            self.train_img = img
            self.train_labels = labels
        
            if self.verbose: print 'training data shuffled'
            
            self.shuffled=True

    
    def shard_data(self, file_batch_size, rank, size):
        
        # usually after batch_data and each shuffle_data call
        
        img_t, labels_t = self.train_img, self.train_labels
        img_v, labels_v = self.val_img, self.val_labels
        # make divisible
        
        if self.extended==False:
        
            from helper_funcs import extend_data
            
            if len(img_t) % size != 0: img_t, labels_t = extend_data(rank, size, img_t, labels_t)
            if len(img_v) % size != 0: img_v, labels_v = extend_data(rank, size, img_v, labels_v)
            
            self.extended = True
        
        if self.sharded == False:
            # sharding
            img_t = img_t[rank::size]
            img_v = img_v[rank::size]
            labels_t = labels_t[rank::size]
            labels_v = labels_v[rank::size]
            
            self.sharded=True
    
        self.train_img_shard, self.train_labels_shard = img_t, labels_t
        self.val_img_shard, self.val_labels_shard = img_v, labels_v
        self.n_batch_train = len(self.train_img_shard)
        self.n_batch_val = len(self.val_img_shard)
        
        
        