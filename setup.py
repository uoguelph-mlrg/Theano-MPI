from __future__ import absolute_import, print_function, division
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

MAJOR, MINOR = 1 , 0 
DESCRIPTION = None
LONG_DESCRIPTION = None
AUTHOR = 'He Ma, Fei Mao, Graham Taylor'

setup(name='Theano-MPI',
          version='%d.%d' % (MAJOR, MINOR),
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          # classifiers=CLASSIFIERS,
          author=AUTHOR,
          # author_email=AUTHOR_EMAIL,
          # url=URL,
          license='ECL-2.0',
          platforms=['Linux'],
          packages=['theanompi', 'theanompi.lib', 'theanompi.models', 'theanompi.models.data', \
                    'theanompi.models.keras_model_zoo','theanompi.models.keras_model_zoo.data', \
                    'theanompi.models.lasagne_model_zoo', 'theanompi.models.lasagne_model_zoo.data' ],
          # package_dir={'theanompi':'theanompi'},
          package_data={'theanompi': [ '*.yaml']},
          scripts=['theanompi/bin/tmlauncher']
          # install_requires=['theano>=0.9.0','libgpuarray>=0.6.0'],
          )