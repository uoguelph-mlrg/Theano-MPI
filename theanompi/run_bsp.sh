# source /opt/sharcnet/testing/set4theano-new.sh

### BSP rule####
if [[ -z $1 ]]; then
	echo 'need to specify the size of BSP'
	exit 1
else
	size=$1
fi

BSP_sync_type='cdd'
BSP_exch_strategy='nccl32'
modelfile='theanompi.models.alex_net'
modelclass='AlexNet'

echo "Theano-MPI started $size BSP($BSP_sync_type,$BSP_exch_strategy) workers"

args="$BSP_sync_type $BSP_exch_strategy $modelfile $modelclass"

env="--mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 --bind-to none"
case $size in
	1)
		mpirun --mca mpi_warn_on_fork 0 -n 1 --bind-to none python -u worker.py 'cuda0' $args
		;;
	2)
		mpirun $env python -u worker.py 'cuda0' $args : \
			   $env python -u worker.py 'cuda1' $args
		;;
	4)
		mpirun $env python -u worker.py 'cuda0' $args : \
			   $env python -u worker.py 'cuda1' $args : \
	   		   $env python -u worker.py 'cuda2' $args : \
	   		   $env python -u worker.py 'cuda3' $args
		;;
	8)
		mpirun $env numactl -N 0 python -u worker.py 'cuda0' $args : \
			   $env numactl -N 0 python -u worker.py 'cuda1' $args : \
	   		   $env numactl -N 0 python -u worker.py 'cuda2' $args : \
	   		   $env numactl -N 0 python -u worker.py 'cuda3' $args : \
	   	 	   $env numactl -N 1 python -u worker.py 'cuda4' $args : \
	   		   $env numactl -N 1 python -u worker.py 'cuda5' $args : \
	   	   	   $env numactl -N 1 python -u worker.py 'cuda6' $args : \
	   	   	   $env numactl -N 1 python -u worker.py 'cuda7' $args
		;;
	*)
		echo $"Not implemented with this size"
		exit 1
esac