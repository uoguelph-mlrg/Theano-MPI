import numpy as np

class Cifar10_data():
    
    def __init__(self):
        
        # data hyperparams
        
        self.data_path  = '/scratch/hma02/data/cifar10/cifar-10-batches-py/'
        
        self.channels = 3
        self.width =32
        self.height =32

        self.n_class = 10
        
        self.get_data()
        
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
        
        img_mean = img.mean(axis=0) 
        
        
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
        
        
        
        