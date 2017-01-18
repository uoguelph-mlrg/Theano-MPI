from __future__ import absolute_import

from theanompi.lib.base import MPI_GPU_Process

from mpi4py import MPI

class EASGD_Server(MPI_GPU_Process):
    
    def __init__(self, device):
        MPI_GPU_Process.__init__(self, device)
        
        # self.worker_comm = {}
        self.worker_id = {}
        self.first_worker_id = None
        
    def process_request(self, worker_id, worker_rank, message):

        # override Server class method, for connection related request
        reply = None
    
        if message in ['sync_register']:
            if self.first_worker_id == None:
                self.first_worker_id = worker_id
                print '[Server] recording worker is %s' % worker_id
                reply = 'first'
    
        return reply
    
    def action_after(self, worker_id,  worker_rank, message):
        
        if 'sync_register' in message: # Connecting synchronously started workers
        
            worker_rank = self.comm.recv(source = MPI.ANY_SOURCE, tag=int(worker_id))
        
            self.worker_rank[str(worker_id)] = int(worker_rank)
        
            print '[Server] registered worker %d' % worker_id
        
        elif message == 'disconnect':

            self.worker_comm.pop(str(worker_id))
        
            print '[Server] disconnected with worker %d' % worker_id
            
        elif message == 'stop':
            
            print '[Server] stopped by %d' % worker_id
            
            import sys
            sys.exit(0)
        
                
    def test_run(self):
        
        if self.comm == None:
            
            print 'Server communicator not initialized'
            
            return
            
        print 'server started'
        
        # after the barrier, run asynchronously
        self.comm.Barrier()

        while True:
            #  Wait for next request from client
            
            request = self.comm.recv(source=MPI.ANY_SOURCE, tag=199)
                
            #  Do some process work and formulate a reply
            reply = self.process_request(request['id'],request['rank'],\
                                                    request['message'])

            #  Send reply back to client
            self.comm.send(reply, dest=request['rank'], tag=200)
            
            # Do some action work after reply
            self.action_after(request['id'],request['rank'], \
                                                    request['message'])
                                                    
if __name__ == '__main__':
    
    import sys
    device = sys.argv[1]
    modelfile = sys.argv[2]
    modelclass = sys.argv[3]

    server = EASGD_Server(device)
    
    server.test_run()