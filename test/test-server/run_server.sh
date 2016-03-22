source set4theano.sh

# need to use mpirun and ompi-server, otherwise comm.publish() doesn't work
# See https://www.open-mpi.org/doc/v1.5/man1/ompi-server.1.php

# $ mpirun --report-uri /path/to/urifile server
# ...
# $ mpirun --ompi-server file:/path/to/urifile client

rm ./ompi-server.txt

mpirun --report-uri ./ompi-server.txt -n 1 python server.py