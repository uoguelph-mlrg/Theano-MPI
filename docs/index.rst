Welcome to Theano-MPI
==================

Theano-MPI is a python framework for distributed training of deep learning models built in Theano. It implements data-parallelism in serveral ways, e.g., Bulk Synchronous Parallel, `Elastic Averaging SGD`_ and `Gossip SGD`_. This project is an extension to `theano_alexnet`_, aiming to scale up the training framework to more than 8 GPUs and across nodes. Please take a look at this `technical report`_ for an overview of implementation details. 

Theano-MPI is compatible for training models built in different framework libraries, e.g., [Lasagne](https://github.com/Lasagne/Lasagne), [Keras](https://github.com/fchollet/keras), [Blocks](https://github.com/mila-udem/blocks), as long as its model parameters can be exposed as theano shared variables. Theano-MPI also comes with a light-weight layer library for you to build customized models. See [wiki](https://github.com/uoguelph-mlrg/Theano-MPI/wiki) for a quick guide on building customized neural networks based on them. Check out the examples of building [Lasagne VGGNet](https://github.com/uoguelph-mlrg/Theano-MPI/blob/master/theanompi/models/lasagne_model_zoo/vgg16.py), [Wasserstein GAN](https://github.com/uoguelph-mlrg/Theano-MPI/blob/master/theanompi/models/lasagne_model_zoo/wgan.py) and [Keras Wide-ResNet](https://github.com/uoguelph-mlrg/Theano-MPI/tree/master/theanompi/models/keras_model_zoo/wresnet.py).

User Guide
------------

The following pages explains how to install Theano-MPI. how to build and train a customized neural network model in Theano-MPI.

.. toctree::
  :maxdepth: 2

  user/installation
  user/example usage
  user/customize_layers

Evaluation
-------------

This section provides some evaluation results of Theano-MPI. 

.. toctree::
  :maxdepth: 2
  
  eval/eval

# API Reference
# -------------
#
# If you are looking for information on a specific function, class or
# method, this part of the documentation is for you.
#
# .. toctree::
#   :maxdepth: 2
#
#   modules/Sync-Rules
#   modules/Exchangers
#   modules/Recorder
#   modules/data

Citation
-------------

If you would like to cite our work, please use the following bibtex entry.

```bibtex
@article{ma2016theano,
  title = {Theano-MPI: a Theano-based Distributed Training Framework},
  author = {Ma, He and Mao, Fei and Taylor, Graham~W.},
  journal = {arXiv preprint arXiv:1605.08325},
  year = {2016}
}
```

.. _GitHub: https://github.com/uoguelph-mlrg/Theano-MPI 
.. _Elastic Averaging SGD: https://arxiv.org/abs/1412.6651
.. _Gossip SGD: https://arxiv.org/abs/1611.09726
.. _theano_alexnet: https://github.com/uoguelph-mlrg/theano_alexnet
.. _technical report: http://arxiv.org/abs/1605.08325