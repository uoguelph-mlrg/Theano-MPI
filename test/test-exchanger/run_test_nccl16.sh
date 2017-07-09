source ../../run/set4theano.sh

mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 2 --bind-to none python -u test_nccl16.py
	