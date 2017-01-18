
class MPIClient(object):
    
    def __init__(self, port=5555):
        
        self.port = port
        import os
        self.worker_id = os.getpid()
        
        # MPI comm to be defined in child class in get_intranode_comm()
        self.comm = None
        self.server_rank=0
        # self.init_socket()
        
    def comm_request(self, message):
        
        if self.comm == None:
            
            print 'MPIClient communicator not initialized'
            
            return
            
            
        request = {'id': self.worker_id, 'rank': self.rank, 'message':message }
        
        self.comm.send(request, dest=self.server_rank, tag=199)
        
        reply = self.comm.recv(source=self.server_rank, tag=200)
        
        return reply

        
    def comm_action(self, message, action=None):
        
        if self.comm == None:
            
            print 'MPIClient not initialized'
            
            return
        
        request = {'id': self.worker_id, 'rank': self.rank, 'message': message }
        
        self.comm.send(request, dest=self.server_rank, tag=199)
        
        reply = self.comm.recv(source=self.server_rank, tag=200)
        
        if action: action()

    
    # def init_socket(self):
    #
    #     with open('ompi-server.txt', 'r') as f:
    #         line = f.readline()
    #         self.server_address = line.split('tcp://',1)[-1].split(',')[0]
    #
    #     # REQ-REP ZMQ SOCKET
    #     import zmq
    #     context = zmq.Context()
    #     self.socket = context.socket(zmq.REQ)
    #     self.socket.connect("tcp://{}:{}".format(self.server_address, self.port))
    #
    # def sock_request(self, message):
    #
    #     request = {'id': self.worker_id, 'message':message }
    #     self.socket.send_json(request)
    #     reply = self.socket.recv_json()
    #
    #     return reply
    #
    # def sock_action(self, message, action=None):
    #
    #     request = {'id': self.worker_id, 'message': message }
    #     self.socket.send_json(request)
    #     reply = self.socket.recv_json()
    #
    #     if action: action()
        
    def run(self):
        
        # TO BE overriden by inheritance
        
        # reply = self.sock_request('a message')
        
        reply = self.comm_request('a message')

        print 'success'

            
if __name__ == '__main__' :
    
    client = MPIClient()
    client.run()
    