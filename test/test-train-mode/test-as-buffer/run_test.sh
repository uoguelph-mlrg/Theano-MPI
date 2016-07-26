# source /opt/sharcnet/testing/set4theano.sh
# python -u test.py

mpirun --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -x CUDA_VISIBLE_DEVICES=0 -n 1 python -u test-mpi.py : \
	--mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -x CUDA_VISIBLE_DEVICES=1 -n 1 python -u test-mpi.py
