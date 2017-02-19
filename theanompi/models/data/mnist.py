from __future__ import absolute_import

import numpy as np


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False,
                        forever=False):
                        
    '''
    source:
         https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066#file-wgan_mnist-py                    
    '''
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
        if not forever:
            break
            
            
class MNIST_data():
    
    def __init__(self, verbose):
        
        # data hyperparams
        
        self.data_path  = 'mnist.pkl.gz'
        
        self.channels = 3
        self.width =32
        self.height =32

        self.n_class = 10
        
        self.get_data()
        
        self.verbose = verbose
        
        self.batched=False
        self.shuffled=False
        self.sharded=False
        
    def get_data(self):

        dataset = self.data_path
        ''' Loads the dataset

        :type dataset: string
        :param dataset: the path to the dataset (here MNIST)
        source:
            http://deeplearning.net/tutorial/logreg.html
    
        '''
        import os,gzip,pickle
    
        #############
        # LOAD DATA #
        #############

        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                dataset
            )
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                dataset = new_path

        if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
            from six.moves import urllib
            origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, dataset)

        print('... loading data')

        # Load the dataset
        with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)
        # train_set, valid_set, test_set format: tuple(input, target)
        # input is a numpy.ndarray of 2 dimensions (a matrix)
        # where each row corresponds to an example. target is a
        # numpy.ndarray of 1 dimension (vector) that has the same length as
        # the number of rows in the input. It should give the target
        # to the example with the same index in the input.


        
        
        
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set
        
        img_mean = np.array((train_set_x.mean(axis=0)+valid_set_x.mean(axis=0))/2.)[np.newaxis,:]
        
        self.rawdata = [train_set_x, train_set_y, valid_set_x, valid_set_y, img_mean,
                test_set_x, test_set_y]
        
        
    def batch_data(self, batchsize):
    
        if self.batched==False:
            
            X_train  = self.rawdata[0]
            y_train = self.rawdata[1]
            X_val = self.rawdata[2]
            y_val = self.rawdata[3]
            
            X_train=X_train.reshape(len(X_train), 1, 28, 28)
            X_val=X_val.reshape(len(X_val), 1, 28, 28)
            
            self.n_batch_train = len(X_train)/batchsize
            self.n_batch_val = len(X_val)/batchsize
            
            self.batches_train = iterate_minibatches(X_train, y_train, batchsize, shuffle=True,
                                                  forever=True)  
            
            self.batches_val = iterate_minibatches(X_val, y_val, batchsize, shuffle=True,
                                                  forever=True)
                                                                          
            self.batched=True
    
    def shuffle_data(self):
        
        pass
        
    def shard_data(self):
        
        pass
            
        
        
        