# Theano-MPI
Theano-MPI is a distributed framework for training deep learning models built in Theano based on data-parallelism. 
The data-parallelism is implemented in two ways: Bulk Synchronous Parallel and Elastic Averaging SGD. This project is an extension to [theano_alexnet](https://github.com/uoguelph-mlrg/theano_alexnet), aiming to scale up training framework to more than 8 GPUs and across nodes. Please see this [technical report](http://arxiv.org/abs/1605.08325) for an overview of implementation details.

It is compatible for training models built in different framework libraries, e.g., Lasagne, Keras, Blocks, as long as its model parameters can be exposed as a theano shared variable. See lib/base/models/ for details. Or you can build your own models from scratch using basic theano tensor operations and expose your model parameters as theano shared variable. See wiki for a tutorial on building customized neural networks.



## Dependencies
* [OpenMPI 1.8.7](http://www.open-mpi.org/) or at least MPI-2 standard equivalent.
* [mpi4py](https://pypi.python.org/pypi/mpi4py)
* [numpy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [PyCUDA](http://mathema.tician.de/software/pycuda/)
* [zeromq](http://zeromq.org/bindings:python)
* [hickle](https://github.com/telegraphic/hickle)

## How to run

### Prepare image data batches
Follow the precedure in [theano_alexnet](https://github.com/uoguelph-mlrg/theano_alexnet) README.md for downloading image data from ImageNet, shuffling training images, generating data batches, computing the mean image and generating label files. The preprocessed data files will be in hickle format. Each file contains 128 or more images. This is the file batch size *B*. Any divisor of *B* can be used as *batch size* during training. Set the *dir_head*, *train_folder*, *val_folder* in run/config.yaml to reflect the location of your preprocessed data.

### Run training sessions on copper
- 1. ssh copper.sharcnet.ca
- 2. ssh to one computing node e.g., cop3
- 3. set .theanorc to the following:
```
[global]

mode = FAST_RUN

floatX = float32

base_compiledir = /home/USERNAME/.theano

[cuda]

root=/opt/sharcnet/cuda/7.0.28/toolkit
```
- 4. cd into run/ and configure each section in the config.yaml. Configure the yaml file corresponding to the chosen model, e.g., alexnet.yaml, googlenet.yaml, vggnet.yaml or customized.yaml.
- to start a BSP training session: 
  - 1) In the "weight exchange" section in config.yaml, choose as follows:
  ```
  sync_rule: BSP
  ```
  - 2) choose a parameter exchanging strategy from "ar", "asa32", "asa16" and "copper", where "ar" means using Allreduce() from mpi4py, "asa32" and "asa16" mean using the Alltoall-sum-Allgather strategy with float32 and float16 respectively, "copper" means using the binary reduction strategy designed for copper GPU topology.
  - 3) execute "./run_bsp_workers.sh N", in which N is the desired number of workers.

- to start a EASGD training session: 
  - 1) If you want to start server and workers in one communicator, configure config.yaml file as follows:
   ```
   sync_rule: EASGD
   sync_start: True 
   avg_freq: 2 or desired value
   ```
  - 2) check the example ./run_easgd_4w_sync_start.sh (or ./run_easgd_4w.sh if sync_start==False),  decide how many workers you want to run and which hosts and GPUs you want to use for each worker and the server, make your customized run.sh script. 
  - 3) execute your ./run.sh.

## Note

To get the best running speed performance, the memory cache may need to be cleaned before running.

To get deterministic and reproducible results, turn off all randomness in the config 'random' section and use cudaconvnet from pylearn2 instead of the indeterministic dnn.conv and dnn.pool from cuDNN.

## Performance Testing

###BSP
Time per 5120 images in seconds: [allow_gc = True]

| Model | 1GPU  | 2GPU  | 4GPU  | 8GPU  | 16GPU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| AlexNet-128b | 31.20 | 15.65 | 7.78 | 3.90 | |
| GoogLeNet-32b | 134.90 | 67.38 | 33.60 | 16.81 | |
| VGGNet-32b | 410.3 | 216.0 | 113.8 | 64.7 | 38.5 |

<img src=https://github.com/uoguelph-mlrg/Parallel-training/raw/add-EASGD/show/val_a.png width=300/>
<img src=https://github.com/uoguelph-mlrg/Parallel-training/raw/add-EASGD/show/val_g.png width=300/>

## How to customize your model

See wiki
