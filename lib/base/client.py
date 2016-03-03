
class Client(object):
    
    def __init__(self, port=5555):
        
        self.port = port
        import os
        self.worker_id = os.getpid()
        self.init_socket()
        
    
    def init_socket(self):
        
        with open('ompi-server.txt', 'r') as f:
            line = f.readline()
            self.server_address = line.split('tcp://',1)[-1].split(',')[0]
        
        # REQ-REP ZMQ SOCKET
        import zmq
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://{}:{}".format(self.server_address, self.port))
    
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
    