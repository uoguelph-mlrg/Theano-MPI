from __future__ import absolute_import
import numpy as np


dir_head = './prepdata_1000cat_128b/'   # base dir where hkl training data is kept
label_folder = '/labels/'  # 
mean_file = '/misc/img_mean.npy'   

train_folder = '/train_hkl_128b/'   #/hkl_data/  
val_folder = '/val_hkl_128b/' 

RGB_mean = False

# parallel loading
para_load = True
sock_data = 5020

sc=False # if the architecture already includes subtract mean and cropping
        
debug=False

class ImageNet_data(object):
    
    def __init__(self, verbose):
        
        # data hyperparams
        
        self.data_path  = dir_head #'/scratch/ilsvrc12/'
        self.train_folder = train_folder #'train_hkl_b256_b_128/'
        self.val_folder = val_folder #'val_hkl_b256_b_128/'
        
        self.channels = 3
        self.width =256
        self.height =256

        self.n_class = 1000
        
        self.get_data(file_batch_size=128)
        
        self.verbose = verbose
        
        self.batched=False
        self.extended=False
        
        # parallel loading
        self.para_load = para_load
        
    def get_data(self, file_batch_size=128):

        dir_head = self.data_path
        ext_data='.hkl'
        ext_label='.npy'
        
        # if file_batch_size==128:
        #
        #     self.train_folder = 'train_hkl_b256_b_128'
        #     self.val_folder = 'val_hkl_b256_b_128'
        #
        # elif file_batch_size==256:
        #
        #     self.train_folder = 'train_hkl_b256_b_256'
        #     self.val_folder = 'val_hkl_b256_b_256'
        #
        # else:
        #
        #     raise ValueError('Wrong file_batch_size')
        
         
        train_folder_path = dir_head + self.train_folder
        val_folder_path = dir_head + self.val_folder
        label_folder_path = dir_head + label_folder
        import glob
        import numpy as np
        train_filenames = sorted(glob.glob(train_folder_path + '/*' + ext_data))
        val_filenames = sorted(glob.glob(val_folder_path + '/*' + ext_data))
        
        
        if debug:
            train_filenames = train_filenames[:40]
            val_filenames = val_filenames[:20]
            
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
        
        img_std = np.array([0.229, 0.224, 0.225]).astype(np.float32)
        img_std = img_std[:,np.newaxis,np.newaxis,np.newaxis]

    

        self.rawdata = [train_filenames,train_labels,\
                    val_filenames,val_labels,img_mean,img_std] # 6 items
                
                
        
        #self.rawdata=[train_data, train_labels, val_data, val_labels, img_mean]
        
        
        
    def batch_data(self, file_batch_size):
        
        if self.batched==False:
            
            self.n_batch_train = len(self.rawdata[0])
            self.n_batch_val = len(self.rawdata[2])
        
            if self.verbose: print('train on %d files' % self.n_batch_train)
            if self.verbose: print('val on %d files' % self.n_batch_val)
                
                
        
            self.train_img, self.train_labels = self.rawdata[0],[]

            raw_labels = self.rawdata[1]

            for index in range(self.n_batch_train):

                batch_label = raw_labels[(index) \
                                * file_batch_size: \
                                (index + 1) * file_batch_size]

                self.train_labels.append(batch_label)
        



            self.val_img, self.val_labels=self.rawdata[2],[]

            raw_labels = self.rawdata[3]

            for index in range(self.n_batch_val):
           
                batch_label = raw_labels[(index) \
                                * file_batch_size: \
                                (index + 1) * file_batch_size]

                self.val_labels.append(batch_label)
                
            self.batched=True
    
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
    
        # To be called at the begining of an epoch for shuffling the order of training data

            
        if self.extended == False:
            raise RuntimError('shuffle_data needs to be after extend_data')
        
        if mode=='train':
            # 1. generate random indices 
            np.random.seed(common_seed)
            
            self.n_batch_train=len(self.train_img_ext)
            
            indices = np.random.permutation(self.n_batch_train)

            # 2. shuffle batches based on indices
            img = []
            labels=[]

            for index in indices:
                img.append(self.train_img_ext[index])
                labels.append(self.train_labels_ext[index])
        
            self.train_img_shuffle = img
            self.train_labels_shuffle = labels
            
            if self.verbose: print('training data shuffled', indices)
        
        elif mode=='val':
            
            self.val_img_shuffle = self.val_img_ext
            self.val_labels_shuffle = self.val_labels_ext
            

                

    
    def shard_data(self, mode, rank, size):
        
            
        if mode=='train':
        
            # sharding
            self.train_img_shard, self.train_labels_shard = \
                    self.train_img_shuffle[rank::size], self.train_labels_shuffle[rank::size]
            self.n_batch_train = len(self.train_img_shard)
            
            if self.verbose: print('training data sharded', self.n_batch_train)
            
        elif mode=='val':
            self.val_img_shard, self.val_labels_shard = \
                    self.val_img_shuffle[rank::size], self.val_labels_shuffle[rank::size]
            self.n_batch_val = len(self.val_img_shard)
        
            if self.verbose: print('validation data sharded', self.n_batch_val)
        
        
        
    def spawn_load(self):
    
        '''spwan a parallel loading process using MPI'''

        if not para_load:
            return

        from mpi4py import MPI
        import os
        import sys
        
        hostname = MPI.Get_processor_name()
        mpiinfo = MPI.Info.Create()
        
        # will give all nodes filled issue if use key=host because need an additional slot
        # also the hostname should be exactly the same in the output list of --display-allocation
        if hostname != hostname.split('.')[0]:
            hostname = hostname.split('.')[0]
        mpiinfo.Set(key = 'add-host',value = hostname)
            
        num_spawn = 1

        if "CPULIST_train" in os.environ:
        # see https://gist.github.com/lebedov/eadce02a320d10f0e81c
            # print os.environ['CPULIST_train']
            envstr=""
            # for key, value in dict(os.environ).iteritems():
            #     envstr+= '%s=%s\n' % (key,value)
            envstr+='CPULIST_train=%s\n' %  os.environ['CPULIST_train']
            mpiinfo.Set(key ='env', value = envstr)

        
        ninfo = mpiinfo.Get_nkeys()
        # print ninfo
        
        mpicommand = sys.executable

        file_dir = os.path.dirname(os.path.realpath(__file__))# get the dir of imagenet.py
    
        self.icomm= MPI.COMM_SELF.Spawn(mpicommand, 
                args=[file_dir+'/proc_load_mpi.py'],
                info = mpiinfo, maxprocs = num_spawn)
                
                
    def para_load_init(self, shared_x, input_width, input_height, 
                                    rand_crop, batch_crop_mirror):
        
        # 0. send config dict (can't carry any special objects) to loading process
        if not self.para_load:
            return
            
        assert self.icomm != None
        
        # get the context running on 
        import theano.gpuarray
        # This is a bit of black magic that may stop working in future
        # theano releases
        ctx = theano.gpuarray.type.get_context(None)
        
        config={}

        config['gpuid'] = ctx.dev
        config['verbose'] = self.verbose
        config['input_width'] = input_width
        config['input_height'] = input_height
        config['rand_crop'] = rand_crop
        config['batch_crop_mirror'] = batch_crop_mirror
        config['img_mean'] = self.rawdata[4]
        config['img_std'] = self.rawdata[5]
        
        import os
        _sock_data = ((sock_data + int(os.getpid())) % 64511)+1024
        config['sock_data'] = _sock_data
        
        self.icomm.send(config,dest=0,tag=99)

        import zmq
        sock = zmq.Context().socket(zmq.PAIR)
        
        sock.connect('tcp://localhost:{0}'.format(_sock_data))
    
        gpuarray_batch = shared_x.container.value
        # pass ipc handle and related information
        h = gpuarray_batch.get_ipc_handle()
        # 1. send ipc handle of shared_x
        sock.send_pyobj((gpuarray_batch.shape, gpuarray_batch.dtype, h))

        # # 2. send img_mean
        # self.icomm.send(img_mean, dest=0, tag=66)
        
    def para_load_close(self):
        
        # to stop the paraloading process
        
        self.icomm.send('stop',dest=0,tag=40)
        self.icomm.send('stop',dest=0,tag=40)
        
                
        
        
        
        