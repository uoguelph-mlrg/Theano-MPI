source set4theano.sh
# server
host0='cop2' 
device0='gpu7'
if [[ $device0 -ge '4' ]]; then
	numa0=1
else
	numa0=0
fi

# worker
host1='cop2'
device1='gpu0'
if [[ $device1 -ge '4' ]]; then
	numa1=1
else
	numa1=0
fi

# worker
host2='cop2'
device2='gpu1'
if [[ $device2 -ge '4' ]]; then
	numa2=1
else
	numa2=0
fi

# worker
host3='cop2'
device3='gpu2'
if [[ $device3 -ge '4' ]]; then
	numa3=1
else
	numa3=0
fi

# worker
host4='cop2'
device4='gpu3'
if [[ $device4 -ge '4' ]]; then
	numa4=1
else
	numa4=0
fi

# worker
host5='cop2'
device5='gpu4'
if [[ $device5 -ge '4' ]]; then
	numa5=1
else
	numa5=0
fi

# worker
host6='cop2'
device6='gpu5'
if [[ $device6 -ge '4' ]]; then
	numa6=1
else
	numa6=0
fi

# worker
host7='cop2'
device7='gpu6'
if [[ $device7 -ge '4' ]]; then
	numa7=1
else
	numa7=0
fi

# worker
host8='cop2'
device8='gpu7'
if [[ $device8 -ge '4' ]]; then
	numa8=1
else
	numa8=0
fi

# server device default to gpu7, so numactl = 1

# need to use mpirun and ompi-server, otherwise comm.Lookup_names() doesn't work
# See https://www.open-mpi.org/doc/v1.5/man1/ompi-server.1.php
rm ./ompi-server.txt

mpirun --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --report-uri ./ompi-server.txt -n 1 -host $host0 numactl -N $numa0 python -u ../lib/EASGD_Server.py $device0 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host1 numactl -N $numa1 python -u ../lib/EASGD_Worker.py $device1 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host2 numactl -N $numa2 python -u ../lib/EASGD_Worker.py $device2 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host3 numactl -N $numa3 python -u ../lib/EASGD_Worker.py $device3 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host4 numactl -N $numa4 python -u ../lib/EASGD_Worker.py $device4 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host5 numactl -N $numa5 python -u ../lib/EASGD_Worker.py $device5 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host6 numactl -N $numa6 python -u ../lib/EASGD_Worker.py $device6 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host7 numactl -N $numa7 python -u ../lib/EASGD_Worker.py $device7 : \
	   --mca mpi_common_cuda_event_max 10000 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --ompi-server file:./ompi-server.txt -n 1 -host $host8 numactl -N $numa8 python -u ../lib/EASGD_Worker.py $device8
	
