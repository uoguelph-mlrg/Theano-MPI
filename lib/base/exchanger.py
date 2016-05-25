from mpi4py import MPI
import numpy as np
import pycuda.gpuarray as gpuarray
import theano
import theano.misc.pycuda_init
import theano.misc.pycuda_utils

from helper_funcs import bufint, dtype_to_mpi

class BSP_Exchanger(object):
    '''
    model parameter exchanger during BSP weight exchanging
    '''
    def __init__(self, config, drv, ctx, model):

        self.drv = drv
        self.ctx = ctx
        self.comm = config['comm']

        self.size = config['size']
        
        self.exch_strategy = config['exch_strategy']

        self.train_mode = config['train_mode']
        self.cuda_aware = config['cuda_aware']
        
        # TODO make sure exchanger class doesn't keep a self copy of model, only the reference to its param list
        self.param_list = model.params
        self.vels = model.vels
        self.vels2 = model.vels2

        self.avg_func_list = []
        
        if self.train_mode == 'cdd' and self.cuda_aware == False:
            
            self.cdd()
            
        elif self.train_mode == 'cdd' and self.cuda_aware == True:
            
            self.cdd_ca_fp32()
            self.compile_fp32_kernels()
            
            self.d_f32_sumfloats, self.ranksize = self.fp32_kernels
            
        elif self.train_mode == 'avg' and self.exch_strategy == 'ar':
            
            from exchanger_strategy import Exch_allreduce
            self.exch = Exch_allreduce(self.comm)
            self.exch.prepare(self.param_list)
            
        elif self.exch_strategy == 'copper':
            
            from exchanger_strategy import Exch_copper
            self.exch = Exch_copper(self.comm)
            self.exch.prepare(self.param_list, self.ctx, self.drv)
            
        elif self.exch_strategy == 'asa32':
            
            from exchanger_strategy import Exch_asa32
            self.exch = Exch_asa32(self.comm)
            self.exch.prepare(self.param_list, self.ctx, self.drv)
            
        elif self.exch_strategy == 'asa16': 
            
            from exchanger_strategy import Exch_asa16
            self.exch = Exch_asa16(self.comm)
            self.exch.prepare(self.param_list, self.ctx, self.drv)
                

    def exchange(self):
        
        # average w
        if self.train_mode == 'avg' and self.size > 1:
            
            if self.exch_strategy == 'ar':
                
                self.exch.exchange()

            elif self.exch_strategy == 'copper':
                
                self.exch.exchange()
                    
            elif self.exch_strategy == 'asa32': 

                self.exch.exchange()          

            elif self.exch_strategy == 'asa16':
                
                self.exch.exchange()

        # sum delta w
        elif self.train_mode == 'cdd' and self.size > 1:
            
            if self.cuda_aware == False:
	    
                self.comm.Barrier()
                for vel,vel2, param_update in \
                                    zip(self.vels,self.vels2, self.param_update_list):
                    self.comm.Allreduce(vel.get_value(), param_update)
                    vel2.set_value(param_update)
                    
            else:
                
                # copy weight from param_ga to param_update_ga
                for vel, vel_update_ga in \
                                zip(self.vels, self.vel_update_ga_list):

                    vel_ga = \
                     theano.misc.pycuda_utils.to_gpuarray(vel.container.value)

                    self.drv.memcpy_dtod(vel_update_ga.ptr,
                                          vel_ga.ptr,
                                          vel_ga.dtype.itemsize *
                                          vel_ga.size)
                    self.ctx.synchronize()
                    del vel_ga
                                          
                # allreduce weight from param_update_ga to itself
                                          
                wcount=0
                for vel_update_ga in self.vel_update_ga_list:
			        
                    self.comm.Alltoall(
			                  [bufint(vel_update_ga), self.mpitp],
			                  [bufint(self.d_param_32_tmps[wcount]),\
                                                           self.mpitp])
			    
			        # sumfloats(float* f1, float* f2, int numElements,int ranksize,int reducesize)
                    self.d_f32_sumfloats(self.d_param_32_tmps[wcount], \
                                    self.d_param_32_sums[wcount],\
			                        self.reducesizes[wcount],self.ranksize,\
                                    self.reducesizes[wcount],\
                                    block=(256,1,1),\
                                    grid=self.grid_sum_sizes[wcount])
                                    
                    self.ctx.synchronize()
                    self.comm.Allgather(\
                                    [bufint(self.d_param_32_sums[wcount]),\
                                        self.mpitp], [bufint(vel_update_ga),self.mpitp])
                    #param.container.value.release_buffer(param_buf)
                    
                    wcount = wcount +1
                    
                # copy weight from param_reduce_ga back to param_ga
                for vel2, vel_update_ga in \
                                zip(self.vels2, self.vel_update_ga_list):
    
                    vel2_ga = \
                     theano.misc.pycuda_utils.to_gpuarray(vel2.container.value)

                    self.drv.memcpy_dtod(vel2_ga.ptr,
                                          vel_update_ga.ptr,
                                          vel_update_ga.dtype.itemsize *
                                          vel2_ga.size)
                    self.ctx.synchronize()
                      
                    del vel2_ga
                
    def compile_fp32_kernels(self):

        from pycuda.compiler import SourceModule
        mod = SourceModule("""
        __global__ void sumfloats(float* f1, float* f2, int numElements,int ranksize,int reducesize)
        {
                int i =  blockDim.x * blockIdx.x + threadIdx.x;
                //unsigned short t1,t2;
                float t1,t2;
                if (i < numElements)
                {
                        t2 = f1[i];
                        //tf2 = __half2float(t2);

                        for (int j=1;j<ranksize;j++)
                        {
                                t1 = f1[i + reducesize*j];
                                //tf1 = __half2float(t1);
                                //tf2 += tf1;
                                t2 += t1;
                        }

                        //t2 = __float2half_rn(tf2);
                        f2[i] = t2;
                }

        }
        """)
        
        d_f32_sumfloats = mod.get_function("sumfloats")
        ranksize = np.int32(self.size)

        self.fp32_kernels =[ d_f32_sumfloats,ranksize ]

    def cdd(self):

        param_update_list = []
        for param in self.param_list:
            param_update = np.zeros_like(param.get_value())
            param_update_list.append(param_update)

        self.param_update_list=param_update_list
        
    def cdd_ca_fp32(self):
        
        vel_update_ga_list=[]

        for vel in self.vels:
            
            vel_update = vel.get_value()
            
            size_tmp=self.size
            if vel_update.size % size_tmp != 0 and len(vel_update.shape)==1:

                vel_update_shape = (vel_update.shape[0]+ size_tmp - \
                                         vel_update.shape[0]%size_tmp,)

                assert vel_update_shape[0] % size_tmp == 0
                print 'weight shape changed from %s to %s' % \
                             (vel_update.shape, vel_update_shape)
                             
            elif vel_update.size % size_tmp == 0:
                vel_update_shape = vel_update.shape
                
            elif vel_update.size % size_tmp != 0 and len(vel_update.shape)!=1:
                raise NotImplementedError
                
            vel_update_ga = gpuarray.GPUArray(vel_update_shape,vel_update.dtype)
            vel_update_ga_list.append(vel_update_ga)

        # fp32 related parameters
        block_size = np.int32(256)

        param_32_sums=[]
        param_32_tmps=[]

        grid_sum_sizes=[]

        numElements=[]
        reducesizes=[]

        for vel_update_ga in vel_update_ga_list:
	
            numElement = np.int32(vel_update_ga.size)
            
            if numElement%size_tmp!=0:
                print numElement,'x',vel_update_ga.shape
                raise
                
            numElements.append(numElement)
            reducesize = np.int32(numElement/self.size)
            reducesizes.append(reducesize)

            grid_sum_size = (reducesize / block_size + 1, 1)
            grid_sum_sizes.append(grid_sum_size)

            param_32_tmp = np.zeros(numElement, dtype=np.float32)
            param_32_tmps.append(param_32_tmp)
            param_32_sum = np.zeros(reducesize, dtype=np.float32)
            param_32_sums.append(param_32_sum)
            
        # fp32 gpu device related parameters
        
        d_param_32_tmps=[]
        d_param_32_sums=[]

        wcount=0
        for param in self.vels:

        	d_param_32_tmp = gpuarray.to_gpu(param_32_tmps[wcount])
        	d_param_32_tmps.append(d_param_32_tmp)
        	d_param_32_sum = gpuarray.to_gpu(param_32_sums[wcount])
        	d_param_32_sums.append(d_param_32_sum)

        	wcount+=1

        mpitp = dtype_to_mpi(d_param_32_tmps[0].dtype)

        self.vel_update_ga_list=vel_update_ga_list
        self.d_param_32_tmps=d_param_32_tmps
        self.d_param_32_sums=d_param_32_sums
        self.grid_sum_sizes=grid_sum_sizes
        self.numElements=numElements
        self.reducesizes=reducesizes
        self.mpitp=mpitp
        
        
        
class EASGD_Exchanger(object):
    '''
    model parameter exchanger during EASGD weight exchanging (with sync rule intergrated)
    '''
    def __init__(self, config, drv, param_list, etype):
        
        self.etype = etype
        self.drv = drv
        self.param_list = param_list

        self.dest = 0 # size is 1 on both side of this intercomm, so server_rank=0, worker_rank=0
        # TODO not sure if alpha = 1.0/config['size'] is better
        

        if self.etype == 'server':
            self.prepare_server()
            self.alpha = config['server_alpha']
        elif self.etype == 'worker':
            self.prepare_worker()
            self.alpha = config['worker_alpha']
            
        self.update_func = self.mk_update_func()
        self.comm = None
            
        
    def prepare_server(self):
        
        self.g_param_list = self.param_list
        self.g_param_ga_list = []
        self.w_param_ga_list = []
        self.w_param_list = []

        for param in self.param_list:
            np_param = param.get_value()
            w_param = theano.shared(np_param)
            self.w_param_list.append(w_param)
            w_param_ga = gpuarray.GPUArray(np_param.shape,np_param.dtype)
            self.w_param_ga_list.append(w_param_ga)
            g_param_ga = gpuarray.GPUArray(np_param.shape,np_param.dtype)
            self.g_param_ga_list.append(g_param_ga)
            
    def prepare_worker(self):
        
        self.w_param_list = self.param_list
        self.w_param_ga_list = []
        self.g_param_ga_list = []
        self.g_param_list = []

        for param in self.param_list:
            
            np_param = param.get_value()
            g_param = theano.shared(np_param)
            self.g_param_list.append(g_param)
            g_param_ga = gpuarray.GPUArray(np_param.shape,np_param.dtype)
            self.g_param_ga_list.append(g_param_ga)
            w_param_ga = gpuarray.GPUArray(np_param.shape,np_param.dtype)
            self.w_param_ga_list.append(w_param_ga)
            
    def mk_update_func(self):
        
        w_updates = []
        g_updates = []
        for w_param, g_param in zip(self.w_param_list, self.g_param_list):
             w_updates.append(w_param - self.alpha * (w_param - g_param))
             g_updates.append(g_param + self.alpha * (w_param - g_param))
            
        updates = zip(self.w_param_list, w_updates) + \
                        zip(self.g_param_list, g_updates)
        
        return theano.function([], updates=updates)
        
        
    def exchange(self):
        
        # server and worker send param to each other
        
        # this function needs the worker to send an 'exchange' message 
        # to the server through REQ-REP socket first.
        
        assert self.comm != None
        
        if self.etype == 'server':
            
            # copy weight from g_param to g_param_ga
            for g_param, g_param_ga in \
                            zip(self.g_param_list, self.g_param_ga_list):

                param_ga = \
                 theano.misc.pycuda_utils.to_gpuarray(g_param.container.value)

                self.drv.memcpy_dtod(g_param_ga.ptr,
                                      param_ga.ptr,
                                      param_ga.dtype.itemsize *
                                      param_ga.size)
                                      
            # Sendrecv(self, sendbuf, int dest=0, int sendtag=0, recvbuf=None, int source=0, int recvtag=0, Status status=None)
            
            for g_param_ga, w_param_ga in zip(self.g_param_ga_list, self.w_param_ga_list):
                self.comm.Sendrecv(sendbuf = [bufint(g_param_ga), MPI.FLOAT], dest = self.dest,
                                   recvbuf = [bufint(w_param_ga), MPI.FLOAT], source = self.dest,
                                   )
                                   
            # copy weight from w_param_ga to w_param
            for w_param, w_param_ga in \
                            zip(self.w_param_list, self.w_param_ga_list):

            	param_ga = \
                 theano.misc.pycuda_utils.to_gpuarray(w_param.container.value)

            	self.drv.memcpy_dtod(param_ga.ptr,
                                      w_param_ga.ptr,
                                      w_param_ga.dtype.itemsize *
                                      w_param_ga.size)
                               
                               
        elif self.etype == 'worker':
            
            # copy weight from w_param to w_param_ga
            for w_param, w_param_ga in \
                            zip(self.w_param_list, self.w_param_ga_list):

                param_ga = \
                 theano.misc.pycuda_utils.to_gpuarray(w_param.container.value)

                self.drv.memcpy_dtod(w_param_ga.ptr,
                                      param_ga.ptr,
                                      param_ga.dtype.itemsize *
                                      param_ga.size)
                                      
            # Sendrecv(self, sendbuf, int dest=0, int sendtag=0, recvbuf=None, int source=0, int recvtag=0, Status status=None)
            
            for w_param_ga, g_param_ga in zip(self.w_param_ga_list, self.g_param_ga_list):
                self.comm.Sendrecv(sendbuf = [bufint(w_param_ga), MPI.FLOAT], dest = self.dest,
                                   recvbuf = [bufint(g_param_ga), MPI.FLOAT], source = self.dest,
                                   )
                                   
            # copy weight from w_param_ga to w_param
            for g_param, g_param_ga in \
                            zip(self.g_param_list, self.g_param_ga_list):

            	param_ga = \
                 theano.misc.pycuda_utils.to_gpuarray(g_param.container.value)

            	self.drv.memcpy_dtod(param_ga.ptr,
                                      g_param_ga.ptr,
                                      g_param_ga.dtype.itemsize *
                                      g_param_ga.size)
                                      
        self.update_func()
            
        self.comm = None
        
    def copy_to_local(self):
        
        assert self.comm != None
        
        if self.etype == 'server':
            # copy weight from g_param to g_param_ga
            for g_param, g_param_ga in \
                            zip(self.g_param_list, self.g_param_ga_list):

                param_ga = \
                 theano.misc.pycuda_utils.to_gpuarray(g_param.container.value)

                self.drv.memcpy_dtod(g_param_ga.ptr,
                                      param_ga.ptr,
                                      param_ga.dtype.itemsize *
                                      param_ga.size)
                                      
            # Send(self, buf, int dest=0, int tag=0)
            
            
            mpitp = dtype_to_mpi(self.g_param_ga_list[0].dtype)
            
            for g_param_ga in self.g_param_ga_list:
                
                self.comm.Send(buf = [bufint(g_param_ga), mpitp], dest = self.dest)
            
        elif self.etype == 'worker':
            
            mpitp = dtype_to_mpi(self.w_param_ga_list[0].dtype)
            
            for w_param_ga in self.w_param_ga_list:
                
                self.comm.Recv(buf = [bufint(w_param_ga), mpitp], source = self.dest)
                
            # copy weight from w_param_ga to w_param
            for w_param_ga, w_param in \
                            zip(self.w_param_ga_list, self.w_param_list):

                param_ga = \
                 theano.misc.pycuda_utils.to_gpuarray(w_param.container.value)

                self.drv.memcpy_dtod(w_param_ga.ptr,
                                      param_ga.ptr,
                                      param_ga.dtype.itemsize *
                                      param_ga.size)
        self.comm = None
                
                
        
    

 
        
        


