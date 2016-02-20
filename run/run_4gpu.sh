source /opt/sharcnet/testing/set4theano.sh

if [[ -z $1 ]]; then
	echo 'need a host as argument $1'
	exit 1
else
	host=$1
fi

if [[ $2 == 'val' ]]; then
	file='val_mpi.py'
	echo 'validating'
else
	file='train_mpi.py'
	echo 'training'
fi

mpirun --mca btl_smcuda_use_cuda_ipc 0 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x CUDA_VISIBLE_DEVICES=0 -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -n 1 -host $host numactl -N 0 python ../exc/"$file" True cudnn -1 \
:      --mca btl_smcuda_use_cuda_ipc 0 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x CUDA_VISIBLE_DEVICES=1 -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -n 1 -host $host numactl -N 0 python ../exc/"$file" True cudnn -1 \
:      --mca btl_smcuda_use_cuda_ipc 0 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x CUDA_VISIBLE_DEVICES=2 -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -n 1 -host $host numactl -N 1 python ../exc/"$file" True cudnn -1 \
:      --mca btl_smcuda_use_cuda_ipc 0 --prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x CUDA_VISIBLE_DEVICES=3 -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -n 1 -host $host numactl -N 1 python ../exc/"$file" True cudnn -1
