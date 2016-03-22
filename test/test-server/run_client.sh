source set4theano.sh

# need to use mpirun and ompi-server, otherwise comm.Lookup_names() doesn't work
# See https://www.open-mpi.org/doc/v1.5/man1/ompi-server.1.php

# $ mpirun --report-uri /path/to/urifile server
# ...
# $ mpirun --ompi-server file:/path/to/urifile client

mpirun --ompi-server file:./ompi-server.txt -n 1 python client.py