class ImageNet():
    
    def __init__(self,config):
        
        self.data_path=''
        
        self.channels = 3
        self.input_width =224
        self.input_height =224
        self.batch_size = 128
        self.n_class = 1000
        
        self.data=None
        self.config=config
        self.verbose=self.config['verbose']
    
    def get_data(self):
        
        from helper_funcs import unpack_configs, extend_data
        (flag_para_load, flag_top_5, train_filenames, val_filenames, \
        train_labels, val_labels, img_mean) = unpack_configs(self.config)
        
        if self.config['image_mean'] == 'RGB_mean':
            
            image_mean = img_mean.mean(axis=-1).mean(axis=-1).mean(axis=-1) 
            #c01b to # c 
            #print 'BGR_mean %s' % image_mean #[ 122.22585297  116.20915222  103.56548309]
            import numpy as np
            image_mean = image_mean[:,np.newaxis,np.newaxis,np.newaxis]

        if self.config['debug']:
            train_filenames = train_filenames[:40]
            val_filenames = val_filenames[:8]

        env_train=None
        env_val = None
            
        train_filenames,train_labels,train_lmdb_cur_list,n_train_files=\
            extend_data(self.config,train_filenames,train_labels,env_train)
        val_filenames,val_labels,val_lmdb_cur_list,n_val_files \
    		    = extend_data(self.config,val_filenames,val_labels,env_val)  

        self.data = [train_filenames,train_labels,\
                    val_filenames,val_labels,img_mean] # 5 items
    
        if self.verbose: print 'train on %d files' % n_train_files  
        if self.verbose: print 'val on %d files' % n_val_files