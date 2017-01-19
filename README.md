# Theano-MPI
Theano-MPI is a framework for distributed training deep learning models built in Theano. It implements data-parallelism is implemented in serveral ways, e.g., Bulk Synchronous Parallel, ASGD and [Elastic Averaging SGD](https://arxiv.org/abs/1412.6651). This project is an extension to [theano_alexnet](https://github.com/uoguelph-mlrg/theano_alexnet), aiming to scale up the training framework to more than 8 GPUs and across nodes. Please take a look at this [technical report](http://arxiv.org/abs/1605.08325) for an overview of implementation details. To cite our work, please use the following bibtex entry.

```bibtex
@article{ma2016theano,
  title = {Theano-MPI: a Theano-based Distributed Training Framework},
  author = {Ma, He and Mao, Fei and Taylor, Graham~W.},
  journal = {arXiv preprint arXiv:1605.08325},
  year = {2016}
}
```

Theano-MPI is compatible for training models built in different framework libraries, e.g., Lasagne, Keras, Blocks, as long as its model parameters can be exposed as theano shared variables. Theano-MPI also comes with a light-weight layer library for you to build your own models. See [wiki](https://github.com/uoguelph-mlrg/Theano-MPI/wiki) for a tutorial on building customized neural networks. Also, check out an example [incoperation](https://github.com/uoguelph-mlrg/Theano-MPI/blob/master/theanompi/models/lasagne_model_zoo/vgg.py) of the 16-layer VGGNet from [Lasagne model zoo](https://github.com/Lasagne/Recipes/blob/master/modelzoo/) to get an idea of how to import Lasagne models into Theano-MPI.

## Dependencies

Theano-MPI depends on the following libraries and packages. We provide some guidance to the installing them in [wiki](https://github.com/uoguelph-mlrg/Theano-MPI/wiki/Guidance-to-installing-some-dependencies-of-Theano-MPI).
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

## Installation 

After all dependeices are ready, one can clone Theano-MPI and install it by the following.

```
 $ python setup.py install [--user]
```

## Usage

To accelerate your Theano model, Theano-MPI tries to identify two components from your model definition:

* the iteratively updating function of your model
* the parameter sharing rule between instances of this model


It is recommended to organize your model and data definition in the following way.

* `launch_session.py`
  * `models/*.py`
    * `__init__.py`
    * 'modelname.py' : defines the your customized model class
    * `data/*.py`
      * `dataname.py` : defines the your customized data class

Your model class should provides the following attributes:

* `self.params` : a list of Theano shared variables, i.e. trainable model parameters
* `self.data` : an instance of your customized data class
* `self.compile_iter_fns` : a method, your way of compiling train_iter_fn and val_iter_fn
* `self.train_iter` : a method, your way of using your train_iter_fn
* `self.val_iter` : a method, your way of using your val_iter_fn
* `self.adjust_hyperp` : a method, your way of adjusting hyperparameters, e.g., learning rate.
* `self.cleanup` : a method, necessary model and data clean-up steps.

Then your can choose the desired way of sharing params, e.g. BSP.

* BSP (Bulk Syncrhonous Parallel)
* ASGD (Asynchronous Parallel)
* EASGD (Elastic Averaging)

Finally you can edit your launch_session.py and train your models. Below is an example launch script for trainig a customized ModelClass on two GPUs. More examples can be found [here](https://github.com/uoguelph-mlrg/Theano-MPI/tree/master/examples).

```python

from theanompi import BSP

rule=BSP()
# modelfile: the relative path to the model file
# modelclass: the class name of the model to be imported from that file
rule.init(devices=['cuda0', 'cuda1'] , 
          modelfile = 'models.modelname', 
          modelclass = 'ModelClass') 
rule.wait()
```

To prepare the ImageNet data for this example, follow the precedure in [theano_alexnet](https://github.com/uoguelph-mlrg/theano_alexnet) README.md for downloading image data from ImageNet, shuffling training images, generating data batches, computing the mean image and generating label files. The preprocessed data files will be in hickle format. Each file contains 128 or more images. This is the file batch size *B*. Any divisor of *B* can be used as *batch size* during training. Set *dir_head*, *train_folder*, *val_folder* in run/config.yaml to reflect the location of your preprocessed data.

## Example Performance

###BSP tested on up to eight Tesla K80 GPUs
Time per 5120 images in seconds: [allow_gc = True]

| Model | 1GPU  | 2GPU  | 4GPU  | 8GPU  |
| :---: | :---: | :---: | :---: | :---: |
| AlexNet-128b | 20.42 | 10.68 | 5.45 | 3.11 |
| GoogLeNet-32b | 73.24 | 35.88 | 18.45 | 9.59 |
| VGGNet-16b | 353.48 | 191.97 | 99.18 | 66.89 |
| VGGNet-32b | 332.32 | 163.70 | 82.65 | 49.42 |
<img src=https://github.com/uoguelph-mlrg/Parallel-training/raw/master/show/val_a.png width=500/>
<img src=https://github.com/uoguelph-mlrg/Parallel-training/raw/master/show/val_g.png width=500/>

## Note

* To get the best running speed performance, the memory cache may need to be cleaned before running.

* Shuffling training examples before asynchronous training makes the loss surface a lot smoother during model converging.

* Some known bugs and possible enhancement are listed in [Issues](https://github.com/uoguelph-mlrg/Theano-MPI/issues). We welcome all kinds of participation (bug reporting, discussion, pull request, etc) in improving the framework.

## License

© Contributors, 2016-2017. Licensed under an [ECL-2.0](https://github.com/uoguelph-mlrg/Theano-MPI/blob/reorg/LICENSE) license.
