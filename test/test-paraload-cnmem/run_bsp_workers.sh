source set4theano.sh

if [[ -z $1 ]]; then
	echo 'need to specify the size of BSP'
	exit 1
else
	size=$1
fi

case $size in
	1)
		mpirun --mca mpi_warn_on_fork 0 -x THEANO_FLAGS=allow_gc=True -n 1 --bind-to none python -u ../../lib/BSP_Worker.py
		;;
	2)
		mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -x THEANO_FLAGS=lib.cnmem=0.8,allow_gc=True -n 2 --bind-to none python -u ../../lib/BSP_Worker.py
		;;
	*)
		echo $"Not implemented with this size"
		exit 1
esac
