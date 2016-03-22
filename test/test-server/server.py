from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

info = MPI.INFO_NULL

port = MPI.Open_port(info)
print "Server port: '%s'", port

service = 'cpi'
MPI.Publish_name(service, info, port)
print 'Service %s published', service

root = 0
print 'Waiting for connection request'
comm = MPI.COMM_WORLD.Accept(port, info, root)
print 'Connected to one client'

while True:

    message = comm.recv(source=0, tag=0)
    if message == 'quit':
        break
    else:
        print 'Receive one message from client:%s' % message

comm.Disconnect()
print 'Connected with one client'

MPI.Unpublish_name(service, info, port)
print 'Service unpublished'

MPI.Close_port(port)
print 'Server port closed'