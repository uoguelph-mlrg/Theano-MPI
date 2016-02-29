
# server device default to gpu7, so numactl = 1
echo 'numa:' 1 'device:' gpu7

# need to use mpirun and ompi-server, otherwise comm.publish() doesn't work
# See https://www.open-mpi.org/doc/v1.5/man1/ompi-server.1.php

mpirun --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -n 1 numactl -N 1 python ../lib/EASGD_Server.py