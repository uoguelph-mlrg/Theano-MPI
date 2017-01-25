from __future__ import absolute_import
from mpi4py import MPI
import numpy as np
import theano
import pygpu

from theanompi.lib.helper_funcs import bufint, dtype_to_mpi


def do_sendrecv(comm, glist, wlist, dest):
    for gs, ws in zip(glist, wlist):
        g = gs.container.value
        w = ws.container.value
        g.sync()
        w.sync()
        comm.Sendrecv(sendbuf=[bufint(g), MPI.FLOAT], dest=dest,
                      recvbuf=[bufint(w), MPI.FLOAT], source=dest)


def do_send(comm, params, dest):
    for ps in params:
        p = ps.container.value
        mpitp = dtype_to_mpi(p.dtype)
        p.sync()
        comm.Send(buf=[bufint(p), mpitp], dest=dest)


def do_recv(comm, params, dest):
    for ps in params:
        p = ps.container.value
        mpitp = dtype_to_mpi(p.dtype)
        p.sync()
        comm.Recv(buf = [bufint(p), mpitp], source=dest)


class BSP_Exchanger(object):
    '''
    model parameter exchanger during BSP weight exchanging
    '''
    def __init__(self, comm, gpucomm, exch_strategy, sync_type, ctx, model):

        self.ctx = ctx
        self.comm = comm
        self.gpucomm = gpucomm

        self.size = comm.size
        
        self.exch_strategy = exch_strategy

        self.sync_type = sync_type

        
        # TODO make sure exchanger class doesn't keep a self copy of model, only the reference to its param list
        self.param_list = model.params
        self.vels = model.vels
        self.vels2 = model.vels2

        self.avg_func_list = []
        
        if self.sync_type == 'cdd' and self.exch_strategy == 'ar':
            
            from theanompi.lib.exchanger_strategy import Exch_allreduce
            self.exch = Exch_allreduce(self.comm, avg=False)
            self.exch.prepare(self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'copper':
            
            from theanompi.lib.exchanger_strategy import Exch_copper
            self.exch = Exch_copper(self.comm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'copper16':
            
            from theanompi.lib.exchanger_strategy import Exch_copper16
            self.exch = Exch_copper16(self.comm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'asa32':
            
            from theanompi.lib.exchanger_strategy import Exch_asa32
            self.exch = Exch_asa32(self.comm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'asa16':
            
            from theanompi.lib.exchanger_strategy import Exch_asa16
            self.exch = Exch_asa16(self.comm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'nccl32':
            
            from theanompi.lib.exchanger_strategy import Exch_nccl32
            self.exch = Exch_nccl32(intercomm=self.comm, intracomm=self.gpucomm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'nccl16':
            
            from theanompi.lib.exchanger_strategy import Exch_nccl16
            self.exch = Exch_nccl16(intercomm=self.comm, intracomm=self.gpucomm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
         
        elif self.sync_type == 'avg' and self.exch_strategy == 'ar':
            
            from theanompi.lib.exchanger_strategy import Exch_allreduce
            self.exch = Exch_allreduce(self.comm)
            self.exch.prepare(self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'copper':
            
            from theanompi.lib.exchanger_strategy import Exch_copper
            self.exch = Exch_copper(self.comm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'copper16':
            
            from theanompi.lib.exchanger_strategy import Exch_copper16
            self.exch = Exch_copper16(self.comm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'asa32':
            
            from theanompi.lib.exchanger_strategy import Exch_asa32
            self.exch = Exch_asa32(self.comm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'asa16':
            
            from theanompi.lib.exchanger_strategy import Exch_asa16
            self.exch = Exch_asa16(self.comm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'nccl32':
            
            from theanompi.lib.exchanger_strategy import Exch_nccl32
            self.exch = Exch_nccl32(intercomm=self.comm, intracomm=self.gpucomm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'nccl16':
            
            from theanompi.lib.exchanger_strategy import Exch_nccl16
            self.exch = Exch_nccl16(intercomm=self.comm, intracomm=self.gpucomm)
            self.exch.prepare(self.ctx, self.param_list)
                

    def exchange(self, recorder):
        
        recorder.start()
        
        # average w
        if self.sync_type == 'avg' and self.size > 1:
            
            if self.exch_strategy == 'ar':
                
                self.exch.exchange()

            elif self.exch_strategy == 'copper':
                
                self.exch.exchange()
                    
            elif self.exch_strategy == 'asa32': 

                self.exch.exchange()          

            elif self.exch_strategy == 'asa16':
                
                self.exch.exchange()
                
            elif self.exch_strategy == 'copper16':
            
                self.exch.exchange()
                
            elif self.exch_strategy == 'nccl32':
                
                self.exch.exchange()
                
            elif self.exch_strategy == 'nccl16':
                
                self.exch.exchange()

        # sum delta w
        elif self.sync_type == 'cdd' and self.size > 1:
            
            if self.exch_strategy == 'ar':
                
                self.exch.exchange()
                
            elif self.exch_strategy == 'copper':
                
                self.exch.exchange()
                
            elif self.exch_strategy == 'asa32':

                self.exch.exchange()
                
            elif self.exch_strategy == 'asa16':
                
                self.exch.exchange()
                
            elif self.exch_strategy == 'copper16':
            
                self.exch.exchange()
                
            elif self.exch_strategy == 'nccl32':
                
                self.exch.exchange()
            
            elif self.exch_strategy == 'nccl16':
                
                self.exch.exchange()
                
        elif self.size>1:
            
            raise NotImplementedError('wrong sync type')
            
        recorder.end('comm')
                
        
class EASGD_Exchanger(object):
    '''
    model parameter exchanger during EASGD weight exchanging
    
    See: 
        https://arxiv.org/abs/1412.6651
    
    '''
    def __init__(self, alpha, param_list, etype, test=False):
        
        self.etype = etype
        self.param_list = param_list

        self.server_gpurank=0
        self.worker_gpurank=1
        
        # TODO not sure if alpha = 1.0/config['size'] is better

        if self.etype == 'server':
            self.prepare_server()
            self.alpha = alpha
        elif self.etype == 'worker':
            self.prepare_worker()
            self.alpha = alpha
            
        self.update_func = self.mk_update_func()
        self.gpucomm = None
        
        self.test=test
            
        
    def prepare_server(self):
        
        self.g_param_list = self.param_list
        self.w_param_list = []

        for param in self.param_list:
            np_param = param.get_value()
            w_param = theano.shared(np_param)
            self.w_param_list.append(w_param)
            
    def prepare_worker(self):
        
        self.w_param_list = self.param_list
        self.g_param_list = []

        for param in self.param_list:
            np_param = param.get_value()
            g_param = theano.shared(np_param)
            self.g_param_list.append(g_param)
            
    def mk_update_func(self):
        
        if self.etype == 'server':
            
            g_updates = [] #update on the server side
            for w_param, g_param in zip(self.w_param_list, self.g_param_list):
                 g_updates.append(g_param + self.alpha * (w_param - g_param))
            
            updates =  zip(self.g_param_list, g_updates)
                 
        elif self.etype == 'worker':
            
            w_updates = [] # update on the worker side
        
            for w_param, g_param in zip(self.w_param_list, self.g_param_list):
                 w_updates.append(w_param - self.alpha * (w_param - g_param))
                             
            updates = zip(self.w_param_list, w_updates)
            
        else:
            raise NotImplementedError('wrong etype')
        
        
        return theano.function([], updates=updates)
        
        
    def exchange(self, recorder=None):
        
        # server and worker send param to each other

        # this function needs the worker to send an 'exchange' message to server first
        
        assert self.gpucomm != None
        
        if self.test==True: 

            print 'before exchange rank%d : %s' % (self.gpucomm.rank,
                            get_1d_value(self.w_param_list[0].get_value()))
            

        
                    
        
        if recorder: recorder.start()
        
        
        # if self.etype == 'server':
            # do_sendrecv(self.comm, self.g_param_list, self.w_param_list, self.dest)
            
        # elif self.etype == 'worker':
            # do_sendrecv(self.comm, self.w_param_list, self.g_param_list, self.dest)
            
        # server to worker,   g to g 
        for g in self.g_param_list:
            g.container.value.sync()
            self.gpucomm.broadcast(g.container.value, root=self.server_gpurank)
            
        # worker to server,  w to w 
        for w in self.w_param_list:
            w.container.value.sync()
            self.gpucomm.broadcast(w.container.value, root=self.worker_gpurank)
        
        if self.test==True:
            
            print 'after exchange rank%d : %s' % (self.gpucomm.rank,
                            get_1d_value(self.w_param_list[0].get_value()))
                            
        
        self.update_func()
            
        if recorder: recorder.end('comm')
                            
                            
        self.gpucomm = None
        
        
    def copy_to_local(self):
        
        assert self.gpucomm != None
        
        # if self.etype == 'server':
#             do_send(self.comm, self.g_param_list, self.dest)
#
#         elif self.etype == 'worker':
#             do_recv(self.comm, self.w_param_list, self.dest)

        # server to worker,  g to w
        
        if self.etype=='server':
            for g in self.g_param_list:
                g.container.value.sync()
                self.gpucomm.broadcast(g.container.value, root=self.server_gpurank)
            
        else:
            for w in self.w_param_list:
                w.container.value.sync()
                self.gpucomm.broadcast(w.container.value, root=self.server_gpurank)

        self.gpucomm = None
        
        
class ASGD_Exchanger(object):
    '''
    model parameter exchanger during ASGD weight exchanging (with sync rule intergrated)

    '''
    def __init__(self, config, etype, param_list, delta_list=None):
        
        self.etype = etype
        self.param_list = param_list

        self.dest = 0 # size is 1 on both side of this intercomm, so server_rank=0, worker_rank=0
        

        if self.etype == 'server':
            self.prepare_server()
            self.alpha = 1 #config['server_alpha']
        elif self.etype == 'worker':
            self.delta_list = delta_list
            self.prepare_worker()
            self.alpha = 1 #config['worker_alpha']
            
        self.update_func = self.mk_update_func()
        self.comm = None
            
        
    def prepare_server(self):
        
        self.g_param_list = self.param_list
        self.w_delta_list = []

        for param in self.param_list:
            np_param = param.get_value()
            w_delta = theano.shared(np_param)
            self.w_delta_list.append(w_delta)
            
    def prepare_worker(self):
        
        self.w_param_list = self.param_list # vels in this case and sync_freq = 1
        self.w_delta_list = self.delta_list
        self.g_param_list = []

        for param in self.param_list:
            np_param = param.get_value()
            g_param = theano.shared(np_param)
            self.g_param_list.append(g_param)
            
    def mk_update_func(self):
        
        if self.etype == 'server':
            
            g_updates = [] #update on the server side: p_server + delta
            
            for w_delta, g_param in zip(self.w_delta_list, self.g_param_list):
                 g_updates.append(g_param + w_delta) # w_param is the received delta
            
            updates =  zip(self.g_param_list, g_updates)
                 
        elif self.etype == 'worker':
            
            w_updates = [] # update on the worker side: p_server + delta
        
            for g_param, w_delta in zip(self.g_param_list, self.w_delta_list):
                 w_updates.append(g_param + w_delta) # g_param is the received server param
                             
            updates = zip(self.w_param_list, w_updates) \
                        +[(delta,delta*0) for delta in self.w_delta_list] # clear accumulated delta after each push to server
            
        else:
            raise NotImplementedError('wrong etype')
        
        
        return theano.function([], updates=updates)
        
        
    def exchange(self):
        
        # server and worker send param to each other
        
        # this function needs the worker to send an 'exchange' message after a call to its get_vel()
        # to the server through REQ-REP socket first.
        
        assert self.comm != None
        
        if self.etype == 'server':
            
            do_sendrecv(self.comm, self.g_param_list, self.w_delta_list, self.dest)
        elif self.etype == 'worker':
            
            do_sendrecv(self.comm, self.w_delta_list, self.g_param_list, self.dest)
        self.update_func()
            
        self.comm = None # clear comm for server in order to select a different one when another worker requests
        
    def copy_to_local(self):
        
        assert self.comm != None
        
        if self.etype == 'server':
            do_recv(self.comm, self.g_param_list, self.dest)
            
        elif self.etype == 'worker':
            do_recv(self.comm, self.w_param_list, self.dest)

        self.comm = None
                
        
    
def get_1d_value(ndarray):
        
    array = ndarray
    dim_left =  array.ndim 
    
    while dim_left!=1:
        
        array = array[0]
        
        dim_left = array.ndim
        
        # print dim_left
        
    return array 
 
 
class GOSGD_Exchanger(object):
    '''
    model parameter exchanger in GOSGD
    
    See: 
        https://arxiv.org/abs/1611.09726
    
    '''
    def __init__(self, comm, Dict_gpucomm, model, p, test=False):
        
        self.comm = comm
        self.rank=comm.rank
        self.size=comm.size
        
        self.Dict_gpucomm = Dict_gpucomm
        self.gpucomm=None # to be choosen from Dict_gpucomm when exchanging

        self.param_list = model.params
        self.alpha=theano.shared(np.asarray(1.0/self.size, dtype=theano.config.floatX))
        self.src_alpha=theano.shared(np.asarray(0., dtype=theano.config.floatX))
        self.p=p
        
        self.prepare()
        
        self.test=test
            
    def prepare(self):

        self.b_param_list = [] # for receiving params from other process

        for param in self.param_list:
            np_param = param.get_value()
            b_param = theano.shared(np_param)
            self.b_param_list.append(b_param)
            
        self.merge_fn = self.mk_merge_func()
            
            
    def mk_merge_func(self):
                 
        updates = [] # update on the merging side
    
        for param, b_param in zip(self.param_list, self.b_param_list):
             updates.append((self.alpha * param + 
                             self.src_alpha * b_param)/
                             (self.alpha+self.src_alpha)
                             )
                         
        updates = zip(self.param_list, updates)
            
        return theano.function([], updates=updates)
        
        
    def get_gpucomm_with(self, rank_other):
        
        if rank_other<self.rank:
            key = '%d%d' % (rank_other, self.rank)
            other_gpurank = 0
            self_gpurank = 1
            
        elif rank_other>self.rank:
            key = '%d%d' % (self.rank, rank_other)
            other_gpurank = 1
            self_gpurank = 0
        else:
            raise RuntimeError("self.rank should be different from rank_other")
            
        gpucomm=self.Dict_gpucomm[key]
        
        return gpucomm, other_gpurank, self_gpurank
        
        
    def process_messages(self, recorder=None):
        
        if recorder: recorder.start()
        
        status = MPI.Status()
        
        s = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=700, status=status)
        
        # if self.test: print '%d probed, got %s' % (self.rank,s)
        
        while s:
            
            src_rank=status.source
            
            print '%d receiving from %d' % (self.rank, src_rank)
            
            request=self.comm.recv(source=src_rank, tag=700, status=status)
            
            print '%d getting gpucomm pair with %d' % (self.rank, src_rank)
            
            self.gpucomm, src_gpurank, self_gpurank = self.get_gpucomm_with(src_rank)
            
            print '%d merging with %d' % (self.rank, src_rank)
            
            self._merge_params_from(src_gpurank, src_rank)
            
            print '%d probing again' % self.rank
            
            s = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=700, status=status)
            
            if self.test: print '%d probed again, got %s' % (self.rank,s)
            
        if recorder: recorder.end('comm')

            
    def _merge_params_from(self, src_gpurank, src_rank):
        
        assert self.gpucomm!=None
        
        for b in self.b_param_list:
            b.container.value.sync()
            self.gpucomm.broadcast(b.container.value, root=src_gpurank)
        
        src_alpha = self.comm.recv(source=src_rank, tag=701)
        
        self.src_alpha.set_value(src_alpha)
        
        self.merge_fn() # merge self.b_param_list with self.param_list
        
        self.alpha.set_value(src_alpha+self.alpha.get_value())
            
        self.gpucomm=None
        
        
    def push_message(self, dest_rank, recorder=None):
        
        '''
        push message:
        push params_i and alpha_i to the choosen rank
        '''
        if recorder: recorder.start()
        
        # 0. blocking request
        
        if self.test: print '%d pushing msg to %d'  % (self.rank,dest_rank)
        
        self.comm.send(obj='request' ,dest=dest_rank, tag=700)  
        
        # 1. push
        
        self.gpucomm, dest_gpurank, self_gpurank = self.get_gpucomm_with(dest_rank)
        
        self._push_params(self_gpurank, dest_rank)
        
        if self.test: print '%d msg pushed'  % self.rank
        
        if recorder: recorder.end('comm')
        

    def _push_params(self, self_gpurank, dest_rank):
        
        assert self.gpucomm!=None
        
        for p in self.param_list:
            p.container.value.sync()
            self.gpucomm.broadcast(p.container.value, root=self_gpurank)
        
        alpha_new = self.alpha.get_value() / np.asarray(2.0, dtype=theano.config.floatX)
        self.alpha.set_value(alpha_new)
        
        self.comm.send(obj=alpha_new, dest=dest_rank,tag=701)
        
        self.gpucomm=None
        
    def draw(self):
        
        '''
        draw from Bernoulli distribution
        
        '''
        # Bernoulli distribution is a special case of binomial distribution with n=1
        a_draw = np.random.binomial(n=1, p=self.p, size=None)
        
        success = (a_draw==1)
        
        # if self.test: print '%d draw=%s' % (self.rank, success)
        
        return success
        
    
    def choose(self):
        
        '''
        choose a dest_rank from range(size) to push to
        
        '''
        
        dest_rank = self.rank
        
        while dest_rank==self.rank:
            
            dest_rank = np.random.randint(low=0,high=self.size)
            
        # if self.test: print '%d choose=%d' % (self.rank, dest_rank)
        
        return dest_rank
        


