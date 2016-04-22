# Parallel Training
Parallel training is a distributed framework for training deep learning models built in Theano based on data-parallelism. 
The data-parallelism is implemented in two ways: Bulk Synchronous Parallel and Elastic Averaging SGD. This project is an extension to [theano_alexnet](https://github.com/uoguelph-mlrg/theano_alexnet), aiming to scale up training framework to more than 8 GPUs and across nodes. 

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
- 1. ssh copper.sharcnet.ca
- 2. ssh to one computing node e.g., cop3
- 3. set .theanorc to the following:
```
[global]

mode = FAST_RUN

floatX = float32

base_compiledir = /tmp/USERNAME/.theano

[cuda]

root=/opt/sharcnet/cuda/7.0.28/toolkit
```
- 4. cd into run/ 
- to start a BSP training session: 
  - 1) configure config.yaml file as follows:
  ```
  sync_rule: BSP
  avg_freq: 1
  ```
  - 2) execute "./run_bsp_workers.sh N", in which N is the desired number of workers.
  - 
- to start a EASGD training session: 
  - 1) Decide if want to start server and workers in one communicator. Configure config.yaml file as follows:
   ```
   sync_rule: EASGD
   sync_start: True 
   avg_freq: 2 or desired value
   ```
  - 2) check the example ./run_easgd_4w_sync_start.sh (or ./run_easgd_4w.sh if sync_start==False),  decide how many workers you want to run and which hosts and GPUs you want to use for each worker and the server, make your customized run.sh script. 
  - 3) execute your ./run.sh.

## Note

Preprocessed data (1000 catagory, 128 batchsize) is located at /work/mahe6562/prepdata/. 

Make sure you have access to the data.

To get the best running speed performance, the memory cache may need to be cleaned before running.

To get deterministic and reproducible results, turn off all randomness in the config 'random' section and use corrmm as lib_conv which will use GpuCorrMM and Pool_2d from theano.blas and downsampling instead of the indeterministic dnn.conv and dnn.pool from cudnn.

## Performance Testing

###BSP
Time per 5120 images in seconds: [allow_gc = True]

| Model | 1GPU  | 2GPU  | 4GPU  | 8GPU  | 16GPU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| AlexNet-128b | 31.4 | 16.8 | 9.2 | 5.6 | |
| GoogLeNet-32b | 147.2 | 81.7 | 57.2 | 57.5 | |
| VGGNet-32b | 410.3 | 216.0 | 113.8 | 64.7 | 38.5 |

<img src=https://github.com/uoguelph-mlrg/Parallel-training/raw/add-EASGD/show/train.png width=300/><img src=https://github.com/uoguelph-mlrg/Parallel-training/raw/add-EASGD/show/val.png width=300/>

###EASGD

(To be added)
## How to customize your model and use this framework to train it

See wiki (To be added)
