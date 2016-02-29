#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#
#TODO warning other process can interupt this process with the same port number


class Client(object):
    
    def __init__(self, port=5555):
        
        self.port = port
        import os
        self.worker_id = os.getpid()
        self.init_socket()
        
    
    def init_socket(self):
        
        # REQ-REP ZMQ SOCKET
        import zmq
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:{}".format(self.port))
        
        # basic socket
        import socket as sock
        self.client = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
    
    def request(self, message):
        
        request = {'id': self.worker_id, 'message':message }
        self.socket.send_json(request)
        reply = self.socket.recv_json()
        
        return reply
    
    def action(self, message, action=None):
        
        request = {'id': self.worker_id, 'message': message }
        self.socket.send_json(request)
        reply = self.socket.recv_json()
        
        if action: action()
        
    def run(self):
        
        # TO BE overriden by inheritance
        
        reply = self.request('a message')

        print 'success'

            
if __name__ == '__main__' :
    
    client = Client()
    client.run()
    