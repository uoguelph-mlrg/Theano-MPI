'''
Worker class

'''

import numpy as np
import sys  
import os       
from mpi4py import MPI 
import socket
import time
import yaml
from lib.helper_funcs import unpack_configs, extend_data, bufint_cn, bufint, dtype_to_mpi


# Due to MPI default screen output buffering, 
# this is for doing a flush of buffer after every call of 'print'
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)



class Worker(object):
    
    '''
    General Worker class that defines the essential communicators and async APIs for a worker process
    
    '''
    
    def __init__(self, syncrule, paraload = True, device = None):
        
        self.worker_id = os.getpid()
        self.syncrule = syncrule
        self.device = None
        self.paraload = paraload
        

        self.comm = None # for communicating with master (in EASGD) or other workers (in BSP)
        self.rank = None
        self.size = None
        self.icomm = None # for communicating with parallel loading process 
        
        
        if self.syncrule == 'BSP':
            self.device = 'gpu' + str(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            self.BSP_init()
        elif self.syncrule == 'EASGD':
            self.device = device
            self.EASGD_init()
        else:
            raise NotImplementedError('Wrong Synchronization Rule')
        
        if self.paraload == True:
            self.spawn_load()
        
    def BSP_init(self):
    	
    	self.comm = MPI.COMM_WORLD
    	self.rank = self.comm.rank
    	self.size = self.comm.size
        
    def EASGD_init(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        '''
        other initialization
        '''
        
    def spawn_load(self):
    
        num_spawn = 1
        hostname = MPI.Get_processor_name()
        mpiinfo = MPI.Info.Create()
        mpiinfo.Set(key = 'host',value = hostname)
        ninfo = mpiinfo.Get_nkeys()
        print ninfo
        mpicommand = sys.executable

        gpuid = self.device[-1] #str(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        print gpuid
        socketnum = 0
        
        # adjust numactl according to the layout of copper nodes [1-8]
        if int(gpuid) > 3:
            socketnum=1
        printstr = "rank" + str(self.rank) +":numa"+ str(socketnum)
        print printstr

        # spawn loading process
        self.icomm= MPI.COMM_SELF.Spawn('numactl', \
                args=['-N',str(socketnum),mpicommand,'../exc/lib/proc_load_mpi.py',gpuid],\
                info = mpiinfo, maxprocs = num_spawn)
        
    
    def control_apis(self):
        '''
        define other control apis for coordinating asynchronous comm 
        
        '''
        pass
        
        
class ConvNet_Worker(Worker):
    
    '''
    Inherit communicators and API definition from General Worker class
    
    '''
    def __init__(self, config, syncrule, paraload = True, device = None ):
        
        Worker.__init__(self, syncrule, paraload, device)
        
        self.drv = None
        
        self.config = config
        
        self.data = None
        self.model = None
        self.model_name = None
        self.optimizer = None
        self.recorder = None
        self.verbose = self.rank == 0
        
        self.process_config()
        
        # send config dict to loading process
        self.icomm.isend(self.config,dest=0,tag=99)
        
        self.get_data()
        self.init_device()
        self.config['drv'] = self.drv
        self.build_model()
        self.para_load_init()
        
        
        
    def process_config(self):
        
        '''
        modify some config items based on run time info
        
        '''
        self.config['icomm'] = self.icomm
        self.config['comm'] = self.comm
        self.config['rank'] = self.rank
        self.config['size'] = self.size
        self.config['syncrule'] = self.syncrule
        self.config['worker_id'] = self.worker_id
        self.config['device'] = self.device
        
        self.config['sock_data'] += int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        self.model_name=self.config['name']
        with open(self.model_name+'.yaml', 'r') as f:
            model_config = yaml.load(f)
        self.config = dict(self.config.items()+model_config.items())
        
        date = '-%d-%d' % (time.gmtime()[1],time.gmtime()[2])    
        self.config['weights_dir']+= '-'+self.config['name'] \
                                     + '-'+str(self.config['size'])+'gpu-' \
                                     + str(self.config['batch_size'])+'b-' \
                                     + socket.gethostname() + date + '/'
                    
        if self.config['rank'] == 0:
            if not os.path.exists(self.config['weights_dir']):
                os.makedirs(self.config['weights_dir'])
                if self.verbose: print "Creat folder: " + \
                                 self.config['weights_dir']
            else:
                if self.verbose: print "folder exists: " + \
                                 self.config['weights_dir']
            if not os.path.exists(self.config['record_dir']):
                os.makedirs(self.config['record_dir'])
                if self.verbose: print "Creat folder: " + \
                                 self.config['record_dir'] 
            else:
                if self.verbose: print "folder exists: " + \
                                 self.config['record_dir']
            
    def get_data(self):

        '''
        prepare filename and label list 

        '''
        (flag_para_load, flag_top_5, train_filenames, val_filenames, \
        train_labels, val_labels, img_mean) = unpack_configs(self.config)

        if self.config['debug']:
            train_filenames = train_filenames[:16]
            val_filenames = val_filenames[:8]

        if self.config['data_source'] in ['lmdb', 'both']:
            import lmdb
            env_train = lmdb.open(config['lmdb_head']+'/train', readonly=True, lock=False)
            env_val = lmdb.open(config['lmdb_head']+'/val', readonly=True, lock=False)
        else:
            env_train=None
            env_val = None
            
        train_filenames,train_labels,train_lmdb_cur_list,n_train_files=\
            extend_data(self.config,train_filenames,train_labels,env_train)
        val_filenames,val_labels,val_lmdb_cur_list,n_val_files \
    		    = extend_data(self.config,val_filenames,val_labels,env_val)  
        if self.config['data_source'] == 'hkl':
            self.data = [train_filenames,train_labels,\
                        val_filenames,val_labels,img_mean]
        elif self.config['data_source'] == 'lmdb':
            self.data = [train_lmdb_cur_list,train_labels,\
                        val_lmdb_cur_list,val_labels,img_mean]
        else:
            raise NotImplementedError('wrong data source')
    
        if self.verbose: print 'train on %d files' % n_train_files  
        if self.verbose: print 'val on %d files' % n_val_files


    def init_device(self):
        
        gpuid = int(self.device[-1])

        # pycuda and zmq set up
        import pycuda.driver as drv

        drv.init()
        dev = drv.Device(gpuid)
        ctx = dev.make_context()
        self.config['dev'] = dev
        self.config['ctx'] = ctx
        
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(self.config['device'])
        self.drv = drv
        
    def build_model(self):

        import theano
        theano.config.on_unused_input = 'warn'

        if self.model_name=='googlenet':
        	from models.googlenet import GoogLeNet
        	#from lib.googlenet import Dropout as drp
        	self.model = GoogLeNet(self.config)

        elif self.model_name=='alexnet':
        	from models.alex_net import AlexNet
        	#from lib.layers import DropoutLayer as drp
        	self.model = AlexNet(self.config)
        else:
            raise NotImplementedError("wrong model name")
    
    
        from models.googlenet import updates_dict

        compile_time = time.time()
        self.model.compile_train(self.config,updates_dict)
        self.model.compile_val()
        if self.verbose: print 'compile_time %.2f s' % \
                                (time.time() - compile_time)
            
    def para_load_init(self):
    	
        drv = self.drv
        shared_x = self.model.shared_x
        img_mean = self.data[4]

        sock_data = self.config['sock_data']

        import zmq
        sock = zmq.Context().socket(zmq.PAIR)
        sock.connect('tcp://localhost:{0}'.format(sock_data))

        #import theano.sandbox.cuda
        #theano.sandbox.cuda.use(config.device)
        import theano.misc.pycuda_init
        import theano.misc.pycuda_utils
        # pass ipc handle and related information
        gpuarray_batch = theano.misc.pycuda_utils.to_gpuarray(
            shared_x.container.value)
        h = drv.mem_get_ipc_handle(gpuarray_batch.ptr)
        # 1. send ipc handle of shared_x
        sock.send_pyobj((gpuarray_batch.shape, gpuarray_batch.dtype, h))

        # 2. send img_mean
        self.icomm.send(img_mean, dest=0, tag=66)
            
    def run(self):
    
        from optimizer import Optimizer

        from recorder import Recorder

        self.recorder = Recorder(self.config)

        opt = Optimizer(self.config, self.model, self.data, self.recorder)
        
        if self.config['resume_train'] == True:
        	opt.load_model(self.config['load_epoch'])

        opt.start()
        
    


if __name__ == '__main__':
    
     
    
    with open('config.yaml', 'r') as f:
        worker_config = yaml.load(f)
    if sys.argv[1] == 'debug':
        worker_config['debug'] = True
    
    # For running a BSP worker
    worker = ConvNet_Worker(worker_config, 
                            syncrule='BSP',
                            paraload=True)
                             
    worker.run()
        
        
    
        
            
        
                

        
