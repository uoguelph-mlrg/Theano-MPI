Welcome to Theano-MPI
==================

Theano-MPI is a python framework for distributed training of deep learning models built in Theano. It implements data-parallelism in serveral ways, e.g., Bulk Synchronous Parallel, `Elastic Averaging SGD`_ and `Gossip SGD`_. This project is an extension to `theano_alexnet`_, aiming to scale up the training framework to more than 8 GPUs and across nodes. Please take a look at this `technical report`_ for an overview of implementation details. 

Theano-MPI is compatible for training models built in different framework libraries, e.g., `Lasagne`_, `Keras`_, `Blocks`_, as long as its model parameters can be exposed as theano shared variables. Theano-MPI also comes with a light-weight layer library for you to build customized models. Check out the examples of building `Lasagne VGGNet`_, `Wasserstein GAN`_ and `Keras Wide-ResNet`_.

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


.. _GitHub: https://github.com/uoguelph-mlrg/Theano-MPI 
.. _Elastic Averaging SGD: https://arxiv.org/abs/1412.6651
.. _Gossip SGD: https://arxiv.org/abs/1611.09726
.. _theano_alexnet: https://github.com/uoguelph-mlrg/theano_alexnet
.. _technical report: http://arxiv.org/abs/1605.08325
.. _Lasagne: https://github.com/Lasagne/Lasagne
.. _Keras: https://github.com/fchollet/keras
.. _Blocks: https://github.com/mila-udem/blocks
.. _Lasagne VGGNet: https://github.com/uoguelph-mlrg/Theano-MPI/blob/master/theanompi/models/lasagne_model_zoo/vgg16.py
.. _Wasserstein GAN: https://github.com/uoguelph-mlrg/Theano-MPI/blob/master/theanompi/models/lasagne_model_zoo/wgan.py
.. _Keras Wide-ResNet: https://github.com/uoguelph-mlrg/Theano-MPI/tree/master/theanompi/models/keras_model_zoo/wresnet.py