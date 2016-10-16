# Theano-MPI
Theano-MPI is a distributed framework for training deep learning models built in Theano based on data-parallelism. 
The data-parallelism is implemented in two ways: Bulk Synchronous Parallel and Elastic Averaging SGD. This project is an extension to [theano_alexnet](https://github.com/uoguelph-mlrg/theano_alexnet), aiming to scale up the training framework to more than 8 GPUs and across nodes. Please take a look at this [technical report](http://arxiv.org/abs/1605.08325) for an overview of implementation details. To cite our work, please use the following bibtex entry.

```bibtex
@article{ma2016theano,
  title = {Theano-MPI: a Theano-based Distributed Training Framework},
  author = {Ma, He and Mao, Fei and Taylor, Graham~W.},
  journal = {arXiv preprint arXiv:1605.08325},
  year = {2016}
}
```

Theano-MPI is compatible for training models built in different framework libraries, e.g., Lasagne, Keras, Blocks, as long as its model parameters can be exposed as theano shared variables. See lib/base/models/ for details. Or you can build your own models from scratch using basic theano tensor operations and expose your model parameters as theano shared variables. See wiki for a tutorial on building customized neural networks.



## Dependencies
* [OpenMPI 1.8.7](http://www.open-mpi.org/) or an MPI-2 standard equivalent that supports CUDA.
* [mpi4py](https://pypi.python.org/pypi/mpi4py) built on OpenMPI 1.8.7
* [numpy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/) 0.8 or higher
* [zeromq](http://zeromq.org/bindings:python)
* [hickle](https://github.com/telegraphic/hickle)
* [CUDA](https://developer.nvidia.com/cuda-toolkit-70) 7.5 or higher
* [cuDNN](https://developer.nvidia.com/cudnn)
* [pygpu](http://deeplearning.net/software/libgpuarray/installation.html)
* [NCCL](https://github.com/NVIDIA/nccl)

## How to run

### Prepare image data batches
Follow the precedure in [theano_alexnet](https://github.com/uoguelph-mlrg/theano_alexnet) README.md for downloading image data from ImageNet, shuffling training images, generating data batches, computing the mean image and generating label files. The preprocessed data files will be in hickle format. Each file contains 128 or more images. This is the file batch size *B*. Any divisor of *B* can be used as *batch size* during training. Set *dir_head*, *train_folder*, *val_folder* in run/config.yaml to reflect the location of your preprocessed data.

### Run training sessions on copper
- 1. ssh copper.sharcnet.ca
- 2. ssh to one computing node e.g., cop3
- 3. set ~/.theanorc to the following:
```
[global]

mode = FAST_RUN

floatX = float32

base_compiledir = /home/USERNAME/.theano
```
- 4. cd into run/ and configure each section in the config.yaml. Configure the yaml file corresponding to the chosen model, e.g., alexnet.yaml, googlenet.yaml, vggnet.yaml or customized.yaml.
- to start a BSP training session: 
  - 1) In config.yaml, choose as follows:
  ```
  worker_type: BSP
  ```
  - 2) choose a parameter exchanging strategy from "ar", "nccl32" and "nccl16", where "ar" means using Allreduce() from mpi4py, "nccl32" and "nccl16" mean using the NCCL collective communication in float32 and float16 respectively.
  - 3) execute "./run_bsp_workers.sh N", in which N is the desired number of workers. 

- to start an EASGD training session [TODO]: 

## Performance Testing

###BSP [TODO]
Time per 5120 images in seconds: [allow_gc = True]

| Model | 1GPU  | 2GPU  | 4GPU  | 8GPU  |
| :---: | :---: | :---: | :---: | :---: |
| AlexNet-128b | 20.42 | 10.68 | 5.45 | 3.11 |
| GoogLeNet-32b | 73.24 | 35.88 | 18.45 | 9.59 |
| VGGNet-16b | 353.48 | 191.97 | 99.18 | 66.89 |
| VGGNet-32b | 332.32 | 163.70 | 82.65 | 49.42 |
<img src=https://github.com/uoguelph-mlrg/Parallel-training/raw/master/show/val_a.png width=500/>
<img src=https://github.com/uoguelph-mlrg/Parallel-training/raw/master/show/val_g.png width=500/>

## How to customize your model

See wiki for a tutorial of customizing a MLP model in the framework.

Also check out an example [incoperation](https://github.com/uoguelph-mlrg/Theano-MPI/blob/master/lib/base/models/lasagne_model_zoo/vgg.py) of the 16-layer VGGNet from [Lasagne model zoo](https://github.com/Lasagne/Recipes/blob/master/modelzoo/) to get an idea of how to import Lasagne models into Theano-MPI.

## Note

To get the best running speed performance, the memory cache may need to be cleaned before running.

Shuffling training examples before asynchronous training makes the loss surface a lot smoother during model converging.

Some known bugs and possible enhancement are listed in [Issues](https://github.com/uoguelph-mlrg/Theano-MPI/issues). We welcome all kinds of participation (bug reporting, discussion, pull request, etc) in improving the framework.
