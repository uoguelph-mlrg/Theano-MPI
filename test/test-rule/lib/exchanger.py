from mpi4py import MPI
import numpy as np
import theano
import pygpu

from helper_funcs import bufint, dtype_to_mpi


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
            
            from exchanger_strategy import Exch_allreduce
            self.exch = Exch_allreduce(self.comm, avg=False)
            self.exch.prepare(self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'copper':
            
            from exchanger_strategy import Exch_copper
            self.exch = Exch_copper(self.comm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'copper16':
            
            from exchanger_strategy import Exch_copper16
            self.exch = Exch_copper16(self.comm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'asa32':
            
            from exchanger_strategy import Exch_asa32
            self.exch = Exch_asa32(self.comm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'asa16':
            
            from exchanger_strategy import Exch_asa16
            self.exch = Exch_asa16(self.comm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'nccl32':
            
            from exchanger_strategy import Exch_nccl32
            self.exch = Exch_nccl32(intercomm=self.comm, intracomm=self.gpucomm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
            
        elif self.sync_type == 'cdd' and self.exch_strategy == 'nccl16':
            
            from exchanger_strategy import Exch_nccl16
            self.exch = Exch_nccl16(intercomm=self.comm, intracomm=self.gpucomm, avg=False)
            self.exch.prepare(self.ctx, self.vels, self.vels2)
         
        elif self.sync_type == 'avg' and self.exch_strategy == 'ar':
            
            from exchanger_strategy import Exch_allreduce
            self.exch = Exch_allreduce(self.comm)
            self.exch.prepare(self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'copper':
            
            from exchanger_strategy import Exch_copper
            self.exch = Exch_copper(self.comm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'copper16':
            
            from exchanger_strategy import Exch_copper16
            self.exch = Exch_copper16(self.comm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'asa32':
            
            from exchanger_strategy import Exch_asa32
            self.exch = Exch_asa32(self.comm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'asa16':
            
            from exchanger_strategy import Exch_asa16
            self.exch = Exch_asa16(self.comm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'nccl32':
            
            from exchanger_strategy import Exch_nccl32
            self.exch = Exch_nccl32(intercomm=self.comm, intracomm=self.gpucomm)
            self.exch.prepare(self.ctx, self.param_list)
            
        elif self.sync_type == 'avg' and self.exch_strategy == 'nccl16':
            
            from exchanger_strategy import Exch_nccl16
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
                
        else:
            
            raise NotImplementedError('wrong sync type')
            
        recorder.end('comm')
                
        
class EASGD_Exchanger(object):
    '''
    model parameter exchanger during EASGD weight exchanging (with sync rule intergrated)
    
    '''
    def __init__(self, config, param_list, etype):
        
        self.etype = etype
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
        
        
    def exchange(self):
        # server and worker send param to each other

        # this function needs the worker to send an 'exchange' message after a call to its train()
        # to the server through REQ-REP socket first.

        assert self.comm != None
        
        if self.etype == 'server':
            
            do_sendrecv(self.comm, self.g_param_list, self.w_param_list, self.dest)
        elif self.etype == 'worker':
            
            do_sendrecv(self.comm, self.w_param_list, self.g_param_list, self.dest)
        self.update_func()
            
        self.comm = None
        
    def copy_to_local(self):
        
        assert self.comm != None
        
        if self.etype == 'server':
            do_send(self.comm, self.g_param_list, self.dest)

        elif self.etype == 'worker':
            do_recv(self.comm, self.w_param_list, self.dest)

        self.comm = None
        
        
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
                
        
    

 
        
        


