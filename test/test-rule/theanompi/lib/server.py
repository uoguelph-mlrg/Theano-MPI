
class MPIServer(object):
    
    def __init__(self, port=5555):
        
        self.port = port
        
        self.comm=None
        # self.init_socket()
    
    # def init_socket(self):
#
#         # REQ-REP ZMQ SOCKET
#         import zmq
#         context = zmq.Context()
#         self.socket = context.socket(zmq.REP)
#         self.socket.bind("tcp://*:{}".format(self.port)) # one to many
    
    def process_request(self, worker_id, message):
        
        # process a request from a client process 
        #TO BE overriden by inheritance
    
        reply = None
        
        return reply
    
    def action_after(self, worker_id, message):
        
        #TO BE overriden by inheritance
        
        pass
        
    def run(self):
        
        if self.comm == None:
            
            print 'MPIServer communicator not initialized'
            
            return
            
        print 'server started'

        while True:
            #  Wait for next request from client
            request = self.comm.recv(MPI.ANY_SOURCE, tag=199)

            #  Do some 'work'
            reply = self.process_request(request['id'],request['rank'],\
                                                    request['message'])

            #  Send reply back to client
            self.comm.send(reply, dest=request['rank'], tag=200)
            
            # Do some 'work'
            self.action_after(request['id'],request['rank'], \
                                                    request['message'])
        
        
    # def sock_run(self):
    #
    #     print 'server started'
    #
    #     while True:
    #         #  Wait for next request from client
    #         request = self.socket.recv_json()
    #
    #         #  Do some 'work'
    #         reply = self.process_request(request['id'],\
    #                                     request['message'])
    #
    #         #  Send reply back to client
    #         self.socket.send_json(reply)
    #
    #         # Do some 'work'
    #         self.action_after(request['id'],\
    #                                     request['message'])

            
if __name__ == '__main__' :
    
    server = MPIServer()
    server.run()