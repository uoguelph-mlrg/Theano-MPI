import numpy as np

class Cifar10_data():
    
    def __init__(self, config):
        
        self.data_path='/scratch/hma02/data/cifar10/cifar-10-batches-py/'
        
        self.channels = 3
        self.input_width =28
        self.input_height =28
        self.batch_size = 128
        self.n_class = 10
        
        self.data=None
        self.config=config
        self.verbose=self.config['verbose']
    
    def gen_train_valid_test(raw_data, raw_target, r_train, r_valid):
        
       

        tot = float(r_train + r_valid )  #Denominator
        p_train = r_train / tot  #train data ratio
        p_valid = r_valid / tot  #valid data ratio
        
    
        n_raw = raw_data.shape[0] #total number of data		
        n_train =int( math.floor(n_raw * p_train)) # number of train
        n_valid =int( math.floor(n_raw * p_valid)) # number of valid


    
        train = raw_data[0:n_train, :]
        valid = raw_data[n_train:n_train+n_valid, :]
        test  = raw_data[n_train+n_valid: n_train+n_valid+n_test,:]
    
        train_target = raw_target[0:n_train]
        valid_target = raw_target[n_train:n_train+n_valid]
        test_target  = raw_target[n_train+n_valid: n_train+n_valid+n_test]
    
        print 'Among ', n_raw, 'raw data, we generated: '
        print train.shape[0], ' training data'
        print valid.shape[0], ' validation data'
        print test.shape[0],  ' test data\n'
    
        train_set = [train, train_target]
        valid_set = [valid, valid_target]
        test_set  = [test, test_target]
        return [train_set, valid_set, test_set]
        
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
        
        self.data=[train_data, train_labels, val_data, val_labels, img_mean]
        
        
        
        