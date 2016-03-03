if [[ -z $1 ]]; then
	echo 'need a device as argument $1'
	exit 1
else
	device=$1
fi

if [[ ${device:0:3} == "gpu" ]]; then
	
	dev=${device#gpu}
else
	echo 'device starts with *gpu* '
	exit 1
fi

if [[ $dev -ge '4' ]]; then
	numa=1
else
	numa=0
fi

echo 'numa:' $numa 'device:' $1


if [[ -z $2 ]]; then
	echo 'need a host as argument $2'
	exit 1
else
	host=$2
fi
	
# server device default to gpu7, so numactl = 1

# need to use mpirun and ompi-server, otherwise comm.Lookup_names() doesn't work
# See https://www.open-mpi.org/doc/v1.5/man1/ompi-server.1.php

mpirun --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host numactl -N $numa python -u ../lib/EASGD_Worker.py $device