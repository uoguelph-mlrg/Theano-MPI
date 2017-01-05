# source ./set4theano-new.sh

if [[ -z $1 ]]; then
	echo 'need to specify the size of BSP'
	exit 1
else
	size=$1
fi

case $size in
	1)
		mpirun --mca mpi_warn_on_fork 0 -n 1 --bind-to none python -u ../lib/BSP_Worker.py
		;;
	2)
		mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 2 --bind-to none python -u ../lib/BSP_Worker.py
		;;
	4)
		mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 4 --bind-to none python -u ../lib/BSP_Worker.py
		;;
	8)
		mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 8 --bind-to none python -u ../lib/BSP_Worker.py
		;;
		
	16)
		host1=cop1
		host2=cop2
		mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 8 -host $host1 python -u ../lib/BSP_Worker.py : \
			   --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 8 -host $host2 python -u ../lib/BSP_Worker.py
		;;
		
	
	*)
		echo $"Not implemented with this size"
		exit 1
esac
