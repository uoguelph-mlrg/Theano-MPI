import numpy as np


dir_head = './prepdata_1000cat_128b/'   # base dir where hkl training data is kept
label_folder = '/labels/'  # 
mean_file = '/misc/img_mean.npy'   

train_folder = '/train_hkl_128b/'   #/hkl_data/  
val_folder = '/val_hkl_128b/' 

RGB_mean = False

class ImageNet_data():
    
    def __init__(self, verbose):
        
        # data hyperparams
        
        self.data_path  = '/scratch/ilsvrc12/'
        self.train_folder = 'train_hkl_b256_b_128'
        self.val_folder = 'val_hkl_b256_b_128'
        
        self.channels = 3
        self.width =256
        self.height =256

        self.n_class = 1000
        
        self.get_data()
        
        self.verbose = verbose
        
    def get_data(self):

        dir_head = self.data_path
        ext_data='.hkl'
        ext_label='.npy'
        
         
        train_folder_path = dir_head + self.train_folder
        val_folder_path = dir_head + self.val_folder
        label_folder_path = dir_head + label_folder
        import glob
        import numpy as np
        train_filenames = sorted(glob.glob(train_folder_path + '/*' + ext_data))
        val_filenames = sorted(glob.glob(val_folder_path + '/*' + ext_data))
        train_labels = np.load(label_folder_path + 'train_labels' + ext_label)
        val_labels = np.load(label_folder_path + 'val_labels' + ext_label)
        img_mean = np.load(dir_head + mean_file)
        img_mean = img_mean[:, :, :, np.newaxis].astype('float32')
            
        if RGB_mean == True:
            
            image_mean = img_mean.mean(axis=-1).mean(axis=-1).mean(axis=-1) 
            #c01b to # c 
            #print 'BGR_mean %s' % image_mean #[ 122.22585297  116.20915222  103.56548309]
            import numpy as np
            image_mean = image_mean[:,np.newaxis,np.newaxis,np.newaxis]

    

        self.rawdata = [train_filenames,train_labels,\
                    val_filenames,val_labels,img_mean] # 5 items
                
                
        
        #self.rawdata=[train_data, train_labels, val_data, val_labels, img_mean]
        
        
        
    def batch_data(self, file_batch_size):
        
        
        self.n_batch_train = len(self.rawdata[0])
        self.n_batch_val = len(self.rawdata[2])
        
        if self.verbose: print 'train on %d files' % n_train_files  
        if self.verbose: print 'val on %d files' % n_val_files
                
                
        
        self.train_img, self.train_labels = self.rawdata[0],[]

        raw_labels = self.rawdata[1]

        for index in range(self.n_batch_train):

            batch_label = raw_labels[(index) \
                            * file_batch_size: \
                            (index + 1) * file_batch_size]

            self.train_labels.append(batch_label)
        



        self.val_img, self.val_labels=self.rawdata[2],[]

        raw_labels = self.rawdata[3]

        for index in range(self.n_batch_train):
           
            batch_label = raw_labels[(index) \
                            * file_batch_size: \
                            (index + 1) * file_batch_size]

            self.val_labels.append(batch_label)
    
    
    def shuffle_data(self):
    
        # To be called at the begining of an epoch for shuffling the order of training data
    
        # 1. generate random indices 

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

    
    def shard_data(self, filenames, labels, rank, size):
        
        # usually after each shuffle_data call
        
        # make divisible
        n_files = len(filenames)
        labels = labels[:n_files*file_batch_size]  # cut unused labels 
   
        # get a list of training filenames that cannot be allocated to any rank
        bad_left_list = get_bad_list(n_files, size)
        if rank == 0: print 'bad list is '+str(bad_left_list)
        need = (size - len(bad_left_list))  % size  
        if need !=0: 
            filenames.extend(filenames[-1*need:])
            labels=labels.tolist()
            labels.extend(labels[-1*need*file_batch_size:])
        n_files = len(filenames)
        
        # sharding
        filenames = filenames[rank::size]
        labels = labels[rank::size]
    
        return img,labels
        
        
        