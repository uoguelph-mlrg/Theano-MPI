from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

info = MPI.INFO_NULL
service = 'cpi'
print 'Looking up server by service name %s', service
port = MPI.Lookup_name(service, info)
print 'Server found at %s', port

root = 0
print 'Connecting to server'
comm = MPI.COMM_WORLD.Connect(port, info, root)
print 'Connected'

while True:
    print '\nType in your request to server, type *quit* to disconnect:'
    message = raw_input('')
    if message == 'quit':
        break
    else:
        comm.send(message, dest=0, tag=0)

comm.Disconnect()
print 'Disconnected from server'