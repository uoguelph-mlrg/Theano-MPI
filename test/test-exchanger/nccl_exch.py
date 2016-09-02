import theano
from helper_funcs import bufint, dtype_to_mpi
import numpy as np
import pygpu

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
        

class Exch_nccl32(Exch_strategy):
    def __init__(self, intercomm, intracomm, avg=True):
        Exch_strategy.__init__(self)
        
        self.intercomm = intercomm
        self.intersize = intercomm.size
        self.interrank = intercomm.rank
        
        self.intracomm = intracomm
        self.intrasize = intracomm.count
        self.intrarank = intracomm.rank
        
        self.avg = avg

    def verify_shape(self, param_update):
        return param_update.shape

    def verify_numElements(self, *args):
        pass

    def prepare(self, ctx, source_param_list, dest_param_list=None):
        self.source_param_list = source_param_list
        if dest_param_list!=None:
            self.dest_param_list = dest_param_list
        else:
            self.dest_param_list = self.source_param_list

        self.ctx = ctx

        if self.avg:
            division_factor = 1.0 / self.size
            self.avg_func = theano.function(
                [],
                updates=[(param, param * division_factor)
                         for param in self.source_param_list])

    def exchange(self):
        # divding source param first before exchanging
        if self.avg:
            self.avg_func()

        for source_s, dest_s in zip(self.source_param_list,
                                    self.dest_param_list):
            source = source_s.container.value
            source.sync()
            dest = dest_s.container.value
            dest.sync()
            self.intracomm.all_reduce(source, '+', dest)
