
#### device 1
if [[ -z $1 ]]; then
	echo 'need a device1 as argument $1'
	exit 1
else
	device1=$1
fi

if [[ ${device1:0:3} == "gpu" ]]; then
	
	dev1=${device1#gpu}
else
	echo 'device1 starts with *gpu* '
	exit 1
fi

if [[ $dev1 -ge '4' ]]; then
	numa1=1
else
	numa1=0
fi

echo 'numa1:' $numa1 'device1:' $1

######### device 2

if [[ -z $2 ]]; then
	echo 'need a device2 as argument $1'
	exit 1
else
	device2=$2
fi

if [[ ${device2:0:3} == "gpu" ]]; then
	
	dev2=${device2#gpu}
else
	echo 'device2 starts with *gpu* '
	exit 1
fi

if [[ $dev2 -ge '4' ]]; then
	numa2=1
else
	numa2=0
fi

echo 'numa2:' $numa2 'device2:' $2

###### device 3

if [[ -z $3 ]]; then
	echo 'need a device3 as argument $1'
	exit 1
else
	device3=$3
fi

if [[ ${device3:0:3} == "gpu" ]]; then
	
	dev3=${device3#gpu}
else
	echo 'device3 starts with *gpu* '
	exit 1
fi

if [[ $dev3 -ge '4' ]]; then
	numa3=1
else
	numa3=0
fi

echo 'numa3:' $numa3 'device3:' $3


mpirun --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -n 1 numactl -N $numa1 python ../lib/EASGD_Worker.py $device1 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -n 1 numactl -N $numa2 python ../lib/EASGD_Worker.py $device2 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -n 1 numactl -N $numa3 python ../lib/EASGD_Worker.py $device3 : \
	





