module unload intel mkl openmpi
module unload cuda
module load cuda/7.5.18
module load intel/15.0.3
module load openmpi/intel-15.0.3/std/1.8.7
module load hdf/serial/5.1.8.11
module load python/intel/2.7.10
module unload cuda
module load cuda/7.0.28

export LD_LIBRARY_PATH=/opt/sharcnet/cuda/7.0.28/toolkit/lib64:/work/feimao/software_installs/cudnn4:/work/feimao/software_installs/caffe-new/caffe-libs/lib:$LD_LIBRARY_PATH
export CPATH=/work/feimao/software_installs/cudnn4:$CPATH
export LIBRARY_PATH=/work/feimao/software_installs/cudnn4:$LIBRARY_PATH
export PYTHONPATH=/work/feimao/software_installs/python_packages_intel15_ompi187/lib/python2.7/site-packages:/work/feimao/software_installs/caffe-new/caffe-libs/lib/python2.7/site-packages:$PYTHONPATH

export PYTHONPATH=~/.local/:$PYTHONPATH
