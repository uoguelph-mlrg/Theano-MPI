import pycuda.gpuarray as gpuarray
import theano
import theano.misc.pycuda_init
import theano.misc.pycuda_utils
from helper_funcs import bufint, dtype_to_mpi
from pycuda.compiler import SourceModule
import numpy as np

class Exch_strategy(object):
    
    '''
    base class of different BSP parameter exchanging strategies
    '''
    
    def __init__(self):
        
        pass
        
    def prepare(self):
        
        pass
    
    def exchange(self):
        
        pass

class Exch_allreduce(Exch_strategy):
    
    '''
    Basic MPI Allreduce, Non-CUDA-aware
    paramter transfer passing host memory
    
    '''
    def __init__(self, comm, avg=True):
        Exch_strategy.__init__(self)
        
        self.comm = comm
        self.size = self.comm.size
        self.avg = avg
        
        
    def prepare(self, param_list):
        
    	self.param_list = param_list
        self.param_update_list = []

    	
    	for param in self.param_list:
    	    param_update = param.get_value()
    	    self.param_update_list.append(param_update)
            
        if self.avg:
        
            division_factor = 1.0 / self.size
            self.avg_func = theano.function([], \
                            updates=[(param, param * division_factor) \
                            for param in self.param_list])
        
    def exchange(self):
        
        if self.avg:
            self.avg_func()
            
        self.comm.Barrier()
        
        for param, param_update in \
                        zip(self.param_list, self.param_update_list):
            self.comm.Allreduce(param.get_value(), param_update)
            param.set_value(param_update)
        
        
        

class Exch_asa32(Exch_strategy):
    '''
    designed by Fei, He
    alltoall-sum-allgather strategy in float32
    
    '''
    def __init__(self, comm, avg=True):
        Exch_strategy.__init__(self)
        
        self.comm = comm
        self.size = self.comm.size
        self.rank = self.comm.rank
        self.avg = avg
    
    def verify_shape(self, param_update):
        
        size_tmp=self.size
        if param_update.size % size_tmp != 0 and len(param_update.shape)==1:

            param_update_shape = (param_update.shape[0]+ size_tmp - \
                                     param_update.shape[0]%size_tmp,)

            assert param_update_shape[0] % size_tmp == 0
            print 'weight shape changed from %s to %s' % \
                         (param_update.shape, param_update_shape)
                         
        elif param_update.size % size_tmp == 0:
            param_update_shape = param_update.shape
            
        elif param_update.size % size_tmp != 0 and len(param_update.shape)!=1:
            raise NotImplementedError
            
        return param_update_shape
        
    def verify_numElements(self, numElements,param_update_shape,param_update):
        
        size_tmp=self.size
        assert numElements>=param_update.size
        if numElements%size_tmp!=0:
            print numElements,'x',param_update_shape
            raise
            
    def prepare(self, param_list, ctx, drv):
        
        self.param_list = param_list
        self.ctx = ctx
        self.drv = drv
    
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


        self.d_f32_sumfloats = mod.get_function("sumfloats")
        
        self.param_update_ga_list=[]
        self.d_param_32_tmp_list=[]
        self.d_param_32_sum_list=[]
        
        self.numElements_list=[]
        self.reduce_size_list = []
        self.grid_sum_size_list=[]
        
        self.ranksize = np.int32(self.size)
        
        block_size = np.int32(256)

        for param in self.param_list:
            
            # Prepare data in host (CPU) memory
            param_update = param.get_value()
            
            param_update_shape = self.verify_shape(param_update)
            
            numElements = np.int32(np.prod(param_update_shape))
            self.verify_numElements(numElements,param_update_shape,param_update)
            self.numElements_list.append(numElements)
            reducesize = np.int32(numElements/self.size)
            self.reduce_size_list.append(reducesize)
            grid_sum_size = (reducesize / block_size + 1, 1)
            self.grid_sum_size_list.append(grid_sum_size)
            
            param_32_tmp = np.zeros(numElements, dtype=np.float32)
            param_32_sum = np.zeros(reducesize, dtype=np.float32)

            
            #Prepare data in decive (GPU) memory
            param_update_ga = gpuarray.GPUArray(param_update_shape,param_update.dtype)
            self.param_update_ga_list.append(param_update_ga)

            d_param_32_tmp = gpuarray.to_gpu(param_32_tmp)
            self.d_param_32_tmp_list.append(d_param_32_tmp)

            d_param_32_sum = gpuarray.to_gpu(param_32_sum)
            self.d_param_32_sum_list.append(d_param_32_sum)

        self.mpidtype = dtype_to_mpi(self.d_param_32_tmp_list[0].dtype)

        if self.avg:
            division_factor = 1.0 / self.size
            self.avg_func = theano.function([], \
                            updates=[(param, param * division_factor) \
                            for param in self.param_list])
    
    def exchange(self):
        
        mpidtype = self.mpidtype
        if self.avg:
            self.avg_func()
        
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
	                  [bufint(param_update_ga), mpidtype],
	                  [bufint(self.d_param_32_tmp_list[wcount]),\
                                                   mpidtype])
	    
	        # sumfloats(float* f1, float* f2, int numElements,int ranksize,int reducesize)
            self.d_f32_sumfloats(self.d_param_32_tmp_list[wcount], \
                            self.d_param_32_sum_list[wcount],\
	                        self.reduce_size_list[wcount],self.ranksize,\
                            self.reduce_size_list[wcount],\
                            block=(256,1,1),\
                            grid=self.grid_sum_size_list[wcount])
                            
            self.ctx.synchronize()
            self.comm.Allgather(\
                            [bufint(self.d_param_32_sum_list[wcount]),mpidtype], \
                            [bufint(param_update_ga),mpidtype])
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
                                  param_ga.size)

class Exch_asa16(Exch_strategy):
    '''
    designed by Fei, He
    alltoall-sum-allgather strategy in float32
    
    TODO: debug indeterministic behaviour
    
    '''
    def __init__(self, comm, avg=True):
        Exch_strategy.__init__(self)
        
        self.comm = comm
        self.size = self.comm.size
        self.rank = self.comm.rank
        self.avg = avg
    
    def verify_shape(self, param_update):
        
        size_tmp=self.size
        if param_update.size % size_tmp != 0 and len(param_update.shape)==1:

            param_update_shape = (param_update.shape[0]+ size_tmp - \
                                     param_update.shape[0]%size_tmp,)

            assert param_update_shape[0] % size_tmp == 0
            print 'weight shape changed from %s to %s' % \
                         (param_update.shape, param_update_shape)
                         
        elif param_update.size % size_tmp == 0:
            param_update_shape = param_update.shape
            
        elif param_update.size % size_tmp != 0 and len(param_update.shape)!=1:
            raise NotImplementedError
            
        return param_update_shape
        
    def verify_numElements(self, numElements,param_update_shape,param_update):
        
        size_tmp=self.size
        assert numElements>=param_update.size
        if numElements%size_tmp!=0:
            print numElements,'x',param_update_shape
            raise
            
    def prepare(self, param_list, ctx, drv):
        
        self.param_list = param_list
        self.ctx = ctx
        self.drv = drv
    
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
        self.float2half = mod.get_function("float2half")
        self.half2float = mod.get_function("half2float")
        self.sumhalfs = mod.get_function("sumhalfs")
        
        self.param_update_ga_list=[]
        self.d_param_16_list = []
        self.d_param_16_update_list = []
        self.d_param_16_tmp_list=[]
        self.d_param_16_sum_list=[]
        
        self.numElements_list=[]
        self.reduce_size_list = []
        self.grid_sum_size_list=[]
        self.grid_size_list=[]
        self.offset_list=[]
        
        self.ranksize = np.int32(self.size)
        
        block_size = np.int32(256)

        for param in self.param_list:
            
            #Prepare data in host (CPU) memory
            
            param_update =  param.get_value()

            param_update_shape = self.verify_shape(param_update)

            numElements = np.int32(np.prod(param_update_shape))
            self.verify_numElements(numElements,param_update_shape, param_update)
            self.numElements_list.append(numElements)
            reducesize = np.int32(numElements/self.size)
            self.reduce_size_list.append(reducesize)
            grid_size = (numElements/(block_size*8) + 1,1)
            self.grid_size_list.append(grid_size)
            grid_sum_size = (reducesize / block_size + 1, 1)
            self.grid_sum_size_list.append(grid_sum_size)
            offset = np.int32(numElements/8)
            self.offset_list.append(offset)
            
            param_16 = np.zeros(numElements, dtype=np.ushort)
            param_16_tmp = np.zeros(numElements, dtype=np.ushort)
            param_16_sum = np.zeros(reducesize, dtype=np.ushort)
            param_16_update = np.zeros(numElements, dtype=np.ushort)
            

            #Prepare data in decive (GPU) memory
            param_update_ga = gpuarray.GPUArray(param_update_shape,param_update.dtype)
            self.param_update_ga_list.append(param_update_ga)

            d_param_16_tmp = gpuarray.to_gpu(param_16_tmp)
            self.d_param_16_tmp_list.append(d_param_16_tmp)

            d_param_16_sum = gpuarray.to_gpu(param_16_sum)
            self.d_param_16_sum_list.append(d_param_16_sum)
            
            d_param_16 =gpuarray.to_gpu(param_16)
            self.d_param_16_list.append(d_param_16)
            
            d_param_16_update =gpuarray.to_gpu(param_16_update)
            self.d_param_16_update_list.append(d_param_16_update)

        self.mpidtype = dtype_to_mpi(self.d_param_16_tmp_list[0].dtype)

        if self.avg:
            division_factor = 1.0 / self.size
            self.avg_func = theano.function([], \
                            updates=[(param, param * division_factor) \
                            for param in self.param_list])
    
    def exchange(self):
        
        mpidtype = self.mpidtype
        if self.avg:
            self.avg_func()
        
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

            self.float2half(param_update_ga, self.d_param_16_list[wcount], \
                                self.numElements_list[wcount], self.offset_list[wcount], \
                                block=(256,1,1),grid=self.grid_size_list[wcount])

            self.comm.Alltoall(
                            [bufint(self.d_param_16_list[wcount]), mpidtype],\
                            [bufint(self.d_param_16_tmp_list[wcount]),mpidtype])
            self.sumhalfs(self.d_param_16_tmp_list[wcount], \
                     self.d_param_16_sum_list[wcount], \
                     self.reduce_size_list[wcount],self.ranksize,\
                     self.reduce_size_list[wcount], \
                     block=(256,1,1),grid=self.grid_sum_size_list[wcount])

            self.comm.Allgather(
                        [bufint(self.d_param_16_sum_list[wcount]),mpidtype],\
                        [bufint(self.d_param_16_update_list[wcount]),mpidtype]) # d_param_16_update_list redundant

            self.half2float(self.d_param_16_update_list[wcount], param_update_ga, \
                                self.numElements_list[wcount],self.offset_list[wcount], \
                                block=(256,1,1),grid=self.grid_size_list[wcount]) # d_param_16_update_list redundant

            wcount+=1
            
        # copy weight from param_reduce_ga back to param_ga
        for param, param_update_ga in \
                        zip(self.param_list, self.param_update_ga_list):

        	param_ga = \
             theano.misc.pycuda_utils.to_gpuarray(param.container.value)

        	self.drv.memcpy_dtod(param_ga.ptr,
                                  param_update_ga.ptr,
                                  param_update_ga.dtype.itemsize *
                                  param_ga.size)
        
    
class Exch_copper(Exch_strategy):
    '''
    designed by Nikhil and Fei, 
    specific to copper GPU-CPU topology,
    gives lower communication overhead than alltoall-sum-allgather
    except when self.ranksize=8 and parameter size > 16MB
    
    '''
    def __init__(self, comm, avg=True):
        Exch_strategy.__init__(self)
        
        self.comm = comm
        self.size = self.comm.size
        self.rank = self.comm.rank
        self.avg = avg
    
    def prepare(self, param_list, ctx, drv):
        
        self.param_list = param_list
        self.ctx = ctx
        self.drv = drv
    
        mod = SourceModule("""
        __global__ void vecadd(float* current, float* temp, int numElements)
        {
        	int i =  blockDim.x * blockIdx.x + threadIdx.x;
	
        	if (i < numElements)
        	current[i] += temp[i];
        }
        """)
        self.vecadd = mod.get_function("vecadd")
        
        self.param_update_ga_list=[]
        self.d_param_32_tmp_list=[]
        self.numElements_list=[]
        self.grid_size_list=[]
        
        block_size = np.int32(256)

        for param in self.param_list:
            
            # Prepare data in host (CPU) memory
            param_update = param.get_value()
            
            numElements = np.int32(param_update.size)
            self.numElements_list.append(numElements)
            # reducesize = np.int32(numElement/self.size)
            # reducesizes.append(reducesize)
            grid_size = (numElements / block_size + 1, 1)
            self.grid_size_list.append(grid_size)
            
            param_32_tmp = np.zeros(numElements, dtype=np.float32)

            # param_32_sum = np.zeros(reducesize, dtype=np.float32)

            
            #Prepare data in decive (GPU) memory
            param_update_ga = gpuarray.to_gpu(param_update)
            self.param_update_ga_list.append(param_update_ga)

            d_param_32_tmp = gpuarray.to_gpu(param_32_tmp)
            self.d_param_32_tmp_list.append(d_param_32_tmp)

            # d_param_32_sum = gpuarray.to_gpu(param_32_sum)
            # self.d_param_32_sum_list.append(d_param_32_sum)

        self.mpidtype = dtype_to_mpi(self.d_param_32_tmp_list[0].dtype)
        if self.avg:
            
            division_factor = 1.0 / self.size
            self.avg_func = theano.function([], \
                            updates=[(param, param * division_factor) \
                            for param in self.param_list])
    
    def exchange(self):
        
        mpidtype = self.mpidtype
        
        if self.avg: self.avg_func()
        
        # copy weight from param_ga to param_update_ga
        for param, param_update_ga in \
                        zip(self.param_list, self.param_update_ga_list):

            param_ga = \
             theano.misc.pycuda_utils.to_gpuarray(param.container.value)

            self.drv.memcpy_dtod(param_update_ga.ptr,
                                  param_ga.ptr,
                                  param_ga.dtype.itemsize *
                                  param_ga.size)
                                  
        
        if (self.size == 2):
            
            for param_update_ga,d_param_tmp,numElements,grid_size in \
                    zip(self.param_update_ga_list, \
                        self.d_param_32_tmp_list, \
                        self.numElements_list, \
                        self.grid_size_list):

                '''
                Summing and Sharing GPU Data
                Sendrecv Pairing: 0 and 1
                '''
    
                if (self.rank == 1):
                    self.comm.Sendrecv([bufint(param_update_ga), mpidtype], \
                        dest=0, recvbuf=[bufint(d_param_tmp), mpidtype], source=0)
                    self.vecadd(param_update_ga, d_param_tmp, numElements, \
                        block=(256, 1, 1), grid=grid_size)
                    self.ctx.synchronize() 
                    #should synchronize context after a kernel call 
                    # to make sure the kernel has been finished
   	
                elif (self.rank == 0):
                    self.comm.Sendrecv([bufint(param_update_ga), mpidtype], \
                        dest=1, recvbuf=[bufint(d_param_tmp), mpidtype], source=1)
                    self.vecadd(param_update_ga, d_param_tmp, numElements, \
                        block=(256, 1, 1), grid=grid_size)
                    self.ctx.synchronize() 
                    #should synchronize context after a kernel call 
                    # to make sure the kernel has been finished
   	
                self.comm.Barrier()



        elif (self.size == 4):
            
            for param_update_ga,d_param_tmp,numElements,grid_size in \
                    zip(self.param_update_ga_list, \
                        self.d_param_32_tmp_list, \
                        self.numElements_list, \
                        self.grid_size_list):
    
                '''
                Summing GPU Data
                Step 1
                Source GPU -> Destination GPU
                1 -> 0, 3 -> 2
                '''
    
                if (self.rank %2 == 1):
                   	self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank-1)
   	
                elif (self.rank %2 == 0):
                   	self.comm.Recv([bufint(d_param_tmp), mpidtype], source=self.rank+1)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize()
    
                '''
                Step 2
                Sendrecv Pairing: 0 and 2
                '''
                if (self.rank == 2):
                   	self.comm.Sendrecv([bufint(param_update_ga), mpidtype], \
                            dest=0, recvbuf=[bufint(d_param_tmp), mpidtype], source=0)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                            block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize() 
   	
                elif (self.rank == 0):
                   	self.comm.Sendrecv([bufint(param_update_ga), mpidtype], \
                            dest=2, recvbuf=[bufint(d_param_tmp), mpidtype], source=2)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                            block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize() 
    
    
                '''
                Broadcasting Result
                Source GPU -> Destination GPU
                0 -> 1, 2 -> 3
                '''
    
                if (self.rank %2 == 0):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank+1)
   	
                elif (self.rank %2 == 1):
               	        self.comm.Recv([bufint(param_update_ga), mpidtype], source=self.rank-1)
   	
                self.comm.Barrier()



        elif (self.size == 8):
    
            # Use this for parameter size < 16MB
            # Use Fei's implementation for parameter size > 16MB
            
            for param_update_ga,d_param_tmp,numElements,grid_size in \
                    zip(self.param_update_ga_list, \
                        self.d_param_32_tmp_list, \
                        self.numElements_list, \
                        self.grid_size_list):
    
                '''
                Summing GPU Data
                Step 1
                Source GPU -> Destination GPU
                1 -> 0, 3 -> 2, 5 -> 4, 7 -> 6
                '''
    
                if (self.rank %2 == 1):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank-1)
   	
                elif (self.rank %2 == 0):
                   	self.comm.Recv([bufint(d_param_tmp), mpidtype], source=self.rank+1)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize() 
    
    
                '''
                Step 2
                Source GPU -> Destination GPU
                0 -> 2, 4 -> 6
                '''
                if (self.rank %4 == 0):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank+2)
   	
                elif (self.rank == 2) or (self.rank == 6):
                   	self.comm.Recv([bufint(d_param_tmp), mpidtype], source=self.rank-2)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize() 
    
    
                '''
                Step 3
                Sendrecv Pairing: 2 and 6
                '''
                if (self.rank == 2):
                   	self.comm.Sendrecv([bufint(param_update_ga), mpidtype], \
                            dest=6, recvbuf=[bufint(d_param_tmp), mpidtype], source=6)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                    block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize() 
   	
                elif (self.rank == 6):
                   	self.comm.Sendrecv([bufint(param_update_ga), mpidtype], \
                            dest=2, recvbuf=[bufint(d_param_tmp), mpidtype], source=2)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                    block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize()
    
    
                '''
                Broadcasting Results
                Step 1
                Source GPU -> Destination GPU
                2 -> 0, 6 -> 4
                '''
                if  (self.rank == 2) or (self.rank == 6):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank-2)
   	
                elif (self.rank %4 == 0):
               	        self.comm.Recv([bufint(param_update_ga), mpidtype], source=self.rank+2)
    
    
                '''
                Step 2
                Source GPU -> Destination GPU
                0 -> 1, 2 -> 3, 4 -> 5, 6 -> 7
                '''
    
                if (self.rank %2 == 0):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank+1)
   	
                elif (self.rank %2 == 1):
               	        self.comm.Recv([bufint(param_update_ga), mpidtype], source=self.rank-1)
   	
    
                self.comm.Barrier()



        elif (self.size == 16):
            
            for param_update_ga,d_param_tmp,numElements,grid_size in \
                    zip(self.param_update_ga_list, \
                        self.d_param_32_tmp_list, \
                        self.numElements_list, \
                        self.grid_size_list):
    
                '''
                Summing GPU Data
                Step 1
                Source GPU -> Destination GPU
                1 -> 0, 3 -> 2, 5 -> 4, 7 -> 6, 9 -> 8, 11 -> 10, 13 -> 12, 15 -> 14
                '''
    
                if (self.rank %2 == 1):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank-1)
   	
                elif (self.rank %2 == 0):
                   	self.comm.Recv([bufint(d_param_tmp), mpidtype], source=self.rank+1)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                    block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize() 
    
    
                '''
                Step 2
                Source GPU -> Destination GPU
                0 -> 2, 4 -> 6, 8 -> 10, 12 -> 14
                '''
                if (self.rank %4 == 0):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank+2)
   	
                elif (self.rank == 2) or (self.rank == 6) or (self.rank == 10) or (self.rank == 14):
                   	self.comm.Recv([bufint(d_param_tmp), mpidtype], source=self.rank-2)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                        block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize()
    
    
                '''
                Step 3
                Source GPU -> Destination GPU
                2 -> 6, 10 -> 14
                '''
                if (self.rank == 2) or (self.rank == 10):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank+4)
   	
                elif (self.rank == 6) or (self.rank == 14):
                   	self.comm.Recv([bufint(d_param_tmp), mpidtype], source=self.rank-4)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                        block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize()
    
    
                '''
                Step 4
                Sendrecv Pairing: 6 and 14
                '''
                if (self.rank == 6):
                   	self.comm.Sendrecv([bufint(param_update_ga), mpidtype], \
                            dest=14, recvbuf=[bufint(d_param_tmp), mpidtype], source=14)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                        block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize() 
   	
                elif (self.rank == 14):
                   	self.comm.Sendrecv([bufint(param_update_ga), mpidtype], \
                                dest=6, recvbuf=[bufint(d_param_tmp), mpidtype], source=6)
                   	self.vecadd(param_update_ga, d_param_tmp, numElements, \
                                                        block=(256, 1, 1), grid=grid_size)
                   	self.ctx.synchronize() 

    
                '''
                Broadcasting Result
                Step 1
                Source GPU -> Destination GPU
                6 -> 2, 14 -> 10
                '''
                if (self.rank == 6) or (self.rank == 14):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank-4)
   	
                elif (self.rank == 2) or (self.rank == 10):
               	        self.comm.Recv([bufint(param_update_ga), mpidtype], source=self.rank+4)
    
    
                '''
                Step 2
                Source GPU -> Destination GPU
                2 -> 0, 6 -> 4, 10 -> 8, 14 -> 12
                '''
                if  (self.rank == 2) or (self.rank == 6) or (self.rank == 10) or (self.rank == 14):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank-2)
   	
                elif (self.rank %4 == 0):
               	        self.comm.Recv([bufint(param_update_ga), mpidtype], source=self.rank+2)
    
    
                '''
                Step 3
                Source GPU -> Destination GPU
                0 -> 1, 2 -> 3, 4 -> 5, 6 -> 7, 8 -> 9, 10 -> 11, 12 -> 13, 14 -> 15
                '''
    
                if (self.rank %2 == 0):
               	        self.comm.Send([bufint(param_update_ga), mpidtype], dest=self.rank+1)
   	
                elif (self.rank %2 == 1):
               	        self.comm.Recv([bufint(param_update_ga), mpidtype], source=self.rank-1)
   	
                self.comm.Barrier()
                
                
                
        # copy weight from param_update_ga back to param_ga
        for param, param_update_ga in \
                        zip(self.param_list, self.param_update_ga_list):

        	param_ga = \
             theano.misc.pycuda_utils.to_gpuarray(param.container.value)

        	self.drv.memcpy_dtod(param_ga.ptr,
                                  param_update_ga.ptr,
                                  param_update_ga.dtype.itemsize *
                                  param_ga.size)
    
    

    
    
    