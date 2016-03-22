source set4theano.sh


# worker
host1='cop6'
device1='gpu4'
if [[ $device1 -ge '4' ]]; then
	numa1=1
else
	numa1=0
fi

# worker
host2='cop6'
device2='gpu5'
if [[ $device2 -ge '4' ]]; then
	numa2=1
else
	numa2=0
fi

# worker
host3='cop6'
device3='gpu6'
if [[ $device3 -ge '4' ]]; then
	numa3=1
else
	numa3=0
fi

# worker
host4='cop6'
device4='gpu7'
if [[ $device4 -ge '4' ]]; then
	numa4=1
else
	numa4=0
fi
	
# server device default to gpu7, so numactl = 1

# need to use mpirun and ompi-server, otherwise comm.Lookup_names() doesn't work
# See https://www.open-mpi.org/doc/v1.5/man1/ompi-server.1.php

mpirun --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host1 numactl -N $numa1 python -u ../lib/EASGD_Worker.py $device1 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host2 numactl -N $numa2 python -u ../lib/EASGD_Worker.py $device2 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host3 numactl -N $numa3 python -u ../lib/EASGD_Worker.py $device3 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host4 numactl -N $numa4 python -u ../lib/EASGD_Worker.py $device4
	
