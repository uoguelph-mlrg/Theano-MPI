pref="--mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# server device default to gpu7, so numactl = 1

# need to use mpirun and ompi-server, otherwise comm.Lookup_names() doesn't work
# See https://www.open-mpi.org/doc/v1.5/man1/ompi-server.1.php
rm ./ompi-server.txt

mpirun  $pref --report-uri ./ompi-server.txt       -n 1 python -u ../lib/ASGD_Server.py 'cuda0' : \
	    $pref --ompi-server file:./ompi-server.txt -n 1 python -u ../lib/ASGD_Worker.py 'cuda1' : \
	    $pref --ompi-server file:./ompi-server.txt -n 1 python -u ../lib/ASGD_Worker.py 'cuda2'
	
