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
 - to start a BSP training session: 
  - 1) configure config.yaml file as follows:
  ```
  sync_rule: BSP
  avg_freq: 1
  ```
  - 2) execute "./run_bsp_workers.sh N", in which N is the desired number of workers.
 - to start a EASGD training session: 
  - 1) Decide if want to start server and workers in one communicator. Configure config.yaml file as follows:
   ```
   sync_rule: EASGD
   sync_start: True 
   avg_freq: 2 or desired value
   ```
  - 2) check the example ./run_easgd_4w_sync_start.sh (or ./run_easgd_4w.sh if sync_start==False),  decide how many workers you want to run and which hosts and GPUs you want to use for each worker, make your customized run.sh script. Note the server device defaults to gpu7 on copper.
  - 3) execute your ./run.sh.

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

