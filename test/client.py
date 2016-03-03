#! /usr/bin/env python

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

def log(msg, *args):
    if rank == 0:
        print msg % args

info = MPI.INFO_NULL
service = 'pyeval'
log("looking-up service '%s'", service)
port = MPI.Lookup_name(service, info)
log("service located  at port '%s'", port)

root = 0
log('waiting for server connection...')
comm = MPI.COMM_WORLD.Connect(port, info, root)
log('server connected...')

while True:
    done = False
    if rank == root:
        try:
            message = raw_input('pyeval>>> ')
            if message == 'quit':
                message = None
                done = True
        except EOFError:
            message = None
            done = True
        comm.send(message, dest=0, tag=0)
    else:
        message = None
    done = MPI.COMM_WORLD.bcast(done, root)
    if done:
        break

log('disconnecting server...')
comm.Disconnect()