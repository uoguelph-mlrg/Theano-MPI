# Parallel Training
Parallel training framework for training deep learning models built in Theano based on data-parallelism.
The data-parallelism is implemented in two ways: Bulk Synchronous Parallel and Elastic Averaging SGD.

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
 - to start a BSP training session: execute ./run_bsp.sh N, N is the desired number of workers in 
 - to start a EASGD training session: execute ./run_server_workers.sh gpu4 gpu5 gpu6, where "gpu4 gpu5 gpu6" are the desired worker devices and the server device defaults to gpu7. This will start the server process and each worker process in a separate screen session.

## Note

Preprocessed data (1000 catagory, 128 batchsize) is located at /work/mahe6562/prepdata/. 

Make sure you have access to the data.

## Performance Testing

<div class="fig figcenter fighighlight">
  <img src="/show/train.pdf">
  <div class="figcaption"> train </div>
</div>

<div class="fig figcenter fighighlight">
  <img src="/show/val.pdf">
  <div class="figcaption"> validation </div>
</div>

