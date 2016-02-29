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

        self.train_mode = config['train_mode']
        self.cuda_aware = config['cuda_aware']
        self.fp = config['fp']
        self.avg_freq = config['avg_freq']
        
        # TODO make sure exchanger class doesn't keep a self copy of model, only the reference to its param list
        self.param_list = model.params
        self.vels = model.vels
        self.vels2 = model.vels2

        self.avg_func_list = []
        
        if self.train_mode == 'cdd':
            
            self.cdd()
            
        elif self.train_mode == 'avg' and self.cuda_aware == False:
            
            self.avg()
            
        elif self.train_mode == 'avg' and self.cuda_aware == True \
                                                and self.fp == 32:
            self.avg_ca_fp32()
            self.compile_fp32_kernels()

            self.d_f32_sumfloats, self.ranksize = self.fp32_kernels
            
        elif self.train_mode == 'avg' and self.cuda_aware == True \
                                                and self.fp == 16:
            self.avg_ca_fp16()
            self.compile_fp16_kernels()
            
            self.float2half, self.half2float, \
                    self.sumhalfs, self.ranksize =  \
                                    self.fp16_kernels
                

    def exchange(self):
        
        # average w
        if self.train_mode == 'avg' and self.size > 1:

            for avg_func in self.avg_func_list:
                avg_func()
            
            if self.cuda_aware == False:
                
                self.comm.Barrier()
                for param, param_update in \
                            zip(self.param_list, self.param_updates):
                    self.comm.Allreduce(param.get_value(), param_update)
                    param.set_value(param_update)
                    
            elif self.fp == 32: 
                
                # copy weight from param_ga to param_update_ga
                for param, param_update_ga in \
                                zip(self.param_list, self.param_update_ga_list):

                    param_ga = \
                     theano.misc.pycuda_utils.to_gpuarray(param.container.value)

                    self.drv.memcpy_dtod(param_update_ga.ptr,
                                          param_ga.ptr,
                                          param_ga.dtype.itemsize *
                                          param_ga.size)
                                          
                # allreduce weight from param_update_ga to itself
                                          
                wcount=0
                for param_update_ga in self.param_update_ga_list:
			        
                    self.comm.Alltoall(
			                  [bufint(param_update_ga), self.mpitp],
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
                                        self.mpitp], [bufint(param_update_ga),self.mpitp])
                    #param.container.value.release_buffer(param_buf)
                    
                    wcount = wcount +1
                    
                # copy weight from param_reduce_ga back to param_ga
                for param, param_update_ga in \
                                zip(self.param_list, self.param_update_ga_list):
    
                	param_ga = \
                     theano.misc.pycuda_utils.to_gpuarray(param.container.value)

                	self.drv.memcpy_dtod(param_ga.ptr,
                                          param_update_ga.ptr,
                                          param_update_ga.dtype.itemsize *
                                          param_update_ga.size)

            elif self.fp == 16:
			
                # copy weight from param_ga to param_update_ga
                for param, param_update_ga in \
                                zip(self.param_list, self.param_update_ga_list):

                    param_ga = \
                     theano.misc.pycuda_utils.to_gpuarray(param.container.value)

                    self.drv.memcpy_dtod(param_update_ga.ptr,
                                          param_ga.ptr,
                                          param_ga.dtype.itemsize *
                                          param_ga.size)
						  
                # allreduce weight from param_update_ga to itself

                wcount=0
                for param_update_ga in self.param_update_ga_list:

                    self.float2half(param_update_ga, self.d_param_halfs[wcount], \
                                        self.numElements[wcount], self.offsets[wcount], \
                                        block=(256,1,1),grid=self.grid_sizes[wcount])
    
                    self.comm.Alltoall(
                                    [bufint(self.d_param_halfs[wcount]), self.mpitp],\
                                    [bufint(self.d_param_half_tmps[wcount]),self.mpitp])
                    self.sumhalfs(self.d_param_half_tmps[wcount], \
                             self.d_param_half_sums[wcount], \
                             self.reducesizes[wcount],self.ranksize,\
                             self.reducesizes[wcount], \
                             block=(256,1,1),grid=self.grid_sum_sizes[wcount])

                    self.comm.Allgather(
                                [bufint(self.d_param_half_sums[wcount]),self.mpitp],\
                                [bufint(self.d_param_half_updates[wcount]),self.mpitp])
    
                    self.half2float(self.d_param_half_updates[wcount], param_update_ga, \
                                        self.numElements[wcount],self.offsets[wcount], \
                                        block=(256,1,1),grid=self.grid_sizes[wcount])
    
                    wcount+=1

                # copy weight from param_reduce_ga back to param_ga
                for param, param_update_ga in \
                                zip(self.param_list, self.param_update_ga_list):
    
                	param_ga = \
                     theano.misc.pycuda_utils.to_gpuarray(param.container.value)

                	self.drv.memcpy_dtod(param_ga.ptr,
                                          param_update_ga.ptr,
                                          param_update_ga.dtype.itemsize *
                                          param_update_ga.size)
    

        # sum delta w
        elif self.train_mode == 'cdd' and self.size > 1:
	    
            self.comm.Barrier()
            for vel,vel2, param_update in \
                                zip(self.vels,self.vels2, self.param_update_list):
                self.comm.Allreduce(vel.get_value(), param_update)
                vel2.set_value(param_update)
		
    def avg(self):

    	param_update_list = []

    	division_factor = 1.0 / self.size
    	for param in self.param_list:
    	    average_fun = theano.function([], \
                            updates=[(param, param * division_factor)])
    	    self.avg_func_list.append(average_fun)
    	    param_update = param.get_value()
    	    param_update_list.append(param_update)
	
    	self.param_update_list=param_update_list
		
	
    def avg_ca_fp32(self):
        
        param_ga_list = []
        param_update_ga_list=[]

        division_factor = 1.0 / self.size
        for param in self.param_list:
            average_fun = theano.function([], \
                            updates=[(param, param * division_factor)])
            self.avg_func_list.append(average_fun)
            
            param_ga = \
                theano.misc.pycuda_utils.to_gpuarray(param.container.value)
            param_ga_list.append(param_ga)
            param_update = param.get_value()
            param_update_ga = gpuarray.GPUArray(param_update.shape,param_update.dtype)
            param_update_ga_list.append(param_update_ga)

        # fp32 related parameters
        block_size = np.int32(256)

        param_32_sums=[]
        param_32_tmps=[]

        grid_sum_sizes=[]

        numElements=[]
        reducesizes=[]

        for param in self.param_list:
	
            param_update = np.zeros_like(param.get_value())
            numElement = np.int32(param_update.size)
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
        for param in self.param_list:

        	d_param_32_tmp = gpuarray.to_gpu(param_32_tmps[wcount])
        	d_param_32_tmps.append(d_param_32_tmp)
        	d_param_32_sum = gpuarray.to_gpu(param_32_sums[wcount])
        	d_param_32_sums.append(d_param_32_sum)

        	wcount+=1

        mpitp = dtype_to_mpi(d_param_32_tmps[0].dtype)

        self.param_update_ga_list=param_update_ga_list
        self.d_param_32_tmps=d_param_32_tmps
        self.d_param_32_sums=d_param_32_sums
        self.grid_sum_sizes=grid_sum_sizes
        self.numElements=numElements
        self.reducesizes=reducesizes
        self.mpitp=mpitp


    def avg_ca_fp16(self):

        param_ga_list = []
        param_update_ga_list=[]

        division_factor = 1.0 / self.size
        for param in self.param_list:
            average_fun = theano.function([], updates=[(param, param * division_factor)])
            self.avg_func_list.append(average_fun)

            param_ga = \
                theano.misc.pycuda_utils.to_gpuarray(param.container.value)
            param_ga_list.append(param_ga)
            param_update = param.get_value()
            param_update_ga = gpuarray.GPUArray(param_update.shape,param_update.dtype)
            param_update_ga_list.append(param_update_ga)

        # fp16 related parameters
        block_size = np.int32(256)

        offsets=[]
        param_half_updates=[]
        param_half_sums=[]
        param_half_tmps=[]
        param_halfs=[]
        grid_sum_sizes=[]
        grid_sizes=[]
        numElements=[]
        reducesizes=[]

        for param in self.param_list:

            param_update = np.zeros_like(param.get_value())
            numElement = np.int32(param_update.size)
            numElements.append(numElement)
            reducesize = np.int32(numElement/self.size)
            reducesizes.append(reducesize)
            grid_size = (numElement/(block_size*8) + 1,1)
            grid_sizes.append(grid_size)
            grid_sum_size = (reducesize / block_size + 1, 1)
            grid_sum_sizes.append(grid_sum_size)
            param_half = np.zeros(numElement, dtype=np.ushort)
            param_halfs.append(param_half)
            param_half_tmp = np.zeros(numElement, dtype=np.ushort)
            param_half_tmps.append(param_half_tmp)
            param_half_sum = np.zeros(reducesize, dtype=np.ushort)
            param_half_sums.append(param_half_sum)
            param_half_update = np.zeros(numElement, dtype=np.ushort)
            param_half_updates.append(param_half_update)
            offset = np.int32(numElement/8)
            offsets.append(offset)

        d_param_halfs = []
        d_param_half_tmps=[]
        d_param_half_sums=[]
        d_param_half_updates=[]

        wcount=0
        for param in self.param_list:
        
            d_param_half = gpuarray.to_gpu(param_halfs[wcount])
            d_param_halfs.append(d_param_half)
            d_param_half_tmp = gpuarray.to_gpu(param_half_tmps[wcount])
            d_param_half_tmps.append(d_param_half_tmp)
            d_param_half_sum = gpuarray.to_gpu(param_half_sums[wcount])
            d_param_half_sums.append(d_param_half_sum)
            d_param_half_update = gpuarray.to_gpu(param_half_updates[wcount])
            d_param_half_updates.append(d_param_half_update)
            wcount+=1

        mpitp = dtype_to_mpi(d_param_halfs[0].dtype)

        self.param_update_ga_list=param_update_ga_list
        self.d_param_halfs=d_param_halfs
        self.d_param_half_tmps=d_param_half_tmps
        self.d_param_half_sums=d_param_half_sums
        self.d_param_half_updates=d_param_half_updates
                    
        self.grid_sum_sizes=grid_sum_sizes
        self.numElements=numElements
        self.reducesizes=reducesizes
        self.mpitp=mpitp
        self.offsets=offsets
        self.grid_sizes=grid_sizes


    def compile_fp16_kernels(self):

        from pycuda.compiler import SourceModule
        mod = SourceModule("""
__global__ void float2half(float* f, unsigned short* h, int numElements,int offset)
{
	    int i = blockDim.x * blockIdx.x + threadIdx.x;
	    if (i+ offset*7 < numElements)
	    {
	            float t0 = f[i];
	            float t1 = f[i+ offset];
	            float t2 = f[i+ offset*2];
	            float t3 = f[i+ offset*3];
	            float t4 = f[i+ offset*4];
	            float t5 = f[i+ offset*5];
	            float t6 = f[i+ offset*6];
	            float t7 = f[i+ offset*7];
	            unsigned short t01 = __float2half_rn(t0);
	            unsigned short t11 = __float2half_rn(t1);
	            unsigned short t21 = __float2half_rn(t2);
	            unsigned short t31 = __float2half_rn(t3);
	            unsigned short t41 = __float2half_rn(t4);
	            unsigned short t51 = __float2half_rn(t5);
	            unsigned short t61 = __float2half_rn(t6);
	            unsigned short t71 = __float2half_rn(t7);
	            h[i] = t01;
	            h[i+ offset] = t11;
	            h[i+ offset*2] = t21;
	            h[i+ offset*3] = t31;
	            h[i+ offset*4] = t41;
	            h[i+ offset*5] = t51;
	            h[i+ offset*6] = t61;
	            h[i+ offset*7] = t71;
	    }
}
__global__ void half2float(unsigned short* h, float* f, int numElements,int offset)
{
	    int i = blockDim.x * blockIdx.x + threadIdx.x;
	    if (i+ offset*7 < numElements)
	    {
	            unsigned short t0 = h[i];
	            unsigned short t1 = h[i+ offset];
	            unsigned short t2 = h[i+ offset*2];
	            unsigned short t3 = h[i+ offset*3];
	            unsigned short t4 = h[i+ offset*4];
	            unsigned short t5 = h[i+ offset*5];
	            unsigned short t6 = h[i+ offset*6];
	            unsigned short t7 = h[i+ offset*7];
	            float t10 = __half2float(t0);
	            float t11 = __half2float(t1);
	            float t12 = __half2float(t2);
	            float t13 = __half2float(t3);
	            float t14 = __half2float(t4);
	            float t15 = __half2float(t5);
	            float t16 = __half2float(t6);
	            float t17 = __half2float(t7);
	            f[i] = t10;
	            f[i+ offset] = t11;
	            f[i+ offset*2] = t12;
	            f[i+ offset*3] = t13;
	            f[i+ offset*4] = t14;
	            f[i+ offset*5] = t15;
	            f[i+ offset*6] = t16;
	            f[i+ offset*7] = t17;
	    }
}
__global__ void sumhalfs(unsigned short* h1, unsigned short* h2, int numElements,int ranksize,int reducesize)
{
	    int i =  blockDim.x * blockIdx.x + threadIdx.x;
	    unsigned short t1,t2;
	    float tf1,tf2;
	    if (i < numElements)
	    {
	            t2 = h1[i];
	            tf2 = __half2float(t2);

	            for (int j=1;j<ranksize;j++)
	            {
	                    t1 = h1[i + reducesize*j];
	                    tf1 = __half2float(t1);
	                    tf2 += tf1;
	            }

	            t2 = __float2half_rn(tf2);
	            h2[i] = t2;
	    }

}
        """)
        float2half = mod.get_function("float2half")
        half2float = mod.get_function("half2float")
        sumhalfs = mod.get_function("sumhalfs")
        ranksize = np.int32(self.size)

        self.fp16_kernels = [ float2half,half2float,sumhalfs,ranksize ]

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
        
        
        
class EASGD_Exchanger(object):
    '''
    model parameter exchanger during EASGD weight exchanging (with sync rule intergrated)
    '''
    def __init__(self, config, drv, param_list, etype):
        
        self.etype = etype
        self.drv = drv
        self.param_list = param_list
        
        self.rank = config['irank'] # TODO not this rank should be the rank from intercomm,
                                    # may not need rank because its constant to 0 or 1
        self.dest = 0 # stands for the server's irank
        self.alpha = config['alpha'] # 1.0/config['size']

        if self.etype == 'server':
            self.prepare_server()
        elif self.etype == 'worker':
            self.prepare_worker()
            
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
                                   recvbuf = [bufint(g_param_ga), MPI.FLOAT], source = self.rank,
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
                
                
        
    

 
        
        


