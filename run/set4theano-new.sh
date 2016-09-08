module unload intel mkl openmpi hdf python cuda
module load intel/15.0.3
module load openmpi/intel1503-std/1.8.7
module load hdf/serial/5.1.8.11
module load cuda/7.5.18
module load python/intel/2.7.10

export MKL_CBWR=AUTO
export PATH=/opt/sharcnet/testing/python_packages_intel15_ompi187/bin:$PATH
export LD_LIBRARY_PATH=/opt/sharcnet/cuda/7.5.18/lib64:/opt/sharcnet/testing/libgpuarray/lib:/opt/sharcnet/testing/nccl/lib:/opt/sharcnet/testing/cudnn/cudnn4:/opt/sharcnet/testing/caffe/caffe-libs/lib:$LD_LIBRARY_PATH
export CPATH=/opt/sharcnet/testing/cudnn/cudnn4:/opt/sharcnet/testing/libgpuarray/include:$CPATH
export LIBRARY_PATH=/opt/sharcnet/testing/cudnn/cudnn4:/opt/sharcnet/testing/libgpuarray/lib:$LIBRARY_PATH
export PYTHONPATH=/opt/sharcnet/testing/python_packages_intel15_ompi187/lib/python2.7/site-packages:/opt/sharcnet/testing/caffe/caffe-libs/lib/python2.7/site-packages:$PYTHONPATH
