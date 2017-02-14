"""
Code adjusted from https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066#file-wgan_mnist-py

"""

from __future__ import (absolute_import, print_function)

import numpy as np

import theano
import theano.tensor as T
import lasagne

# ##################### Build the neural network model #######################
# We create two models: The generator and the critic network.
# The models are the same as in the Lasagne DCGAN example, except that the
# discriminator is now a critic with linear output instead of sigmoid output.

def build_generator(input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
    try:
        from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
    except ImportError:
        raise ImportError("Your Lasagne is too old. Try the bleeding-edge "
                          "version: http://lasagne.readthedocs.io/en/latest/"
                          "user/installation.html#bleeding-edge-version")
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import sigmoid
    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*7*7))
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                                     output_size=14))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, crop='same', output_size=28,
                          nonlinearity=sigmoid)
    print ("Generator output:", layer.output_shape)
    return layer

def build_critic(input_var=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer)
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import LeakyRectify
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer (linear and without bias)
    layer = DenseLayer(layer, 1, nonlinearity=None, b=None)
    print ("critic output:", layer.output_shape)
    return layer
    

num_epochs=1000
epochsize=100
batchsize=64
initial_eta=5e-5
clip=0.01
         
class WGAN(object):
    
    def __init__(self, config):
        
        self.verbose = config['verbose']
        self.rank = config['rank']
        self.size = config['size']
        
        # data
        from theanompi.models.data.mnist import MNIST_data

        self.data = MNIST_data(self.verbose)
        self.data.batch_data(batchsize)
        
        self.batch_size=batchsize
        self.file_batch_size=batchsize
        self.n_subb=self.file_batch_size/self.batch_size
        
        # model
        self.build_model()
        
        self.params = self.generator_params+self.critic_params
        
        # training related
        self.epoch=0
        self.n_epochs=num_epochs
        
        self.generator_updates = 0
        self.critic_scores = []
        self.generator_scores = []
        
        
    def build_model(self):
        
        # Prepare Theano variables for inputs and targets
        self.noise_var = T.matrix('noise')
        self.input_var = T.tensor4('inputs')
        
        # Create neural network model
        generator = build_generator(self.noise_var)
        critic = build_critic(self.input_var)
        
        # Create expression for passing real data through the critic
        real_out = lasagne.layers.get_output(critic)
        # Create expression for passing fake data through the critic
        fake_out = lasagne.layers.get_output(critic,
                lasagne.layers.get_output(generator))
    
        # Create score expressions to be maximized (i.e., negative losses)
        self.generator_score = fake_out.mean()
        self.critic_score = real_out.mean() - fake_out.mean()
        
        
        # Create update expressions for training
        self.generator_params = lasagne.layers.get_all_params(generator, trainable=True)
        self.critic_params = lasagne.layers.get_all_params(critic, trainable=True)
        self.generator = generator
        self.critic = critic
        
    def compile_iter_fns(self):
        
        
        eta = theano.shared(lasagne.utils.floatX(initial_eta))
        self.eta=eta
        self.shared_lr=eta
        
        generator_updates = lasagne.updates.rmsprop(
                -self.generator_score, self.generator_params, learning_rate=eta)
        critic_updates = lasagne.updates.rmsprop(
                -self.critic_score, self.critic_params, learning_rate=eta)
                
                
        # Clip critic parameters in a limited range around zero (except biases)
        for param in lasagne.layers.get_all_params(self.critic, trainable=True,
                                                   regularizable=True):
            critic_updates[param] = T.clip(critic_updates[param], -clip, clip)
            
            
        # Instantiate a symbolic noise generator to use for training
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
        noise = srng.uniform((batchsize, 100))

        # Compile functions performing a training step on a mini-batch (according
        # to the updates dictionary) and returning the corresponding score:
        print('Compiling...')
        
        import time
        
        start = time.time()
        
        self.generator_train_fn = theano.function([], self.generator_score,
                                             givens={self.noise_var: noise},
                                             updates=generator_updates)
        self.critic_train_fn = theano.function([self.input_var], self.critic_score,
                                          givens={self.noise_var: noise},
                                          updates=critic_updates)

        # Compile another function generating some data
        self.gen_fn = theano.function([self.noise_var],
                                 lasagne.layers.get_output(self.generator,
                                                           deterministic=True))
                                                           
        self.val_fn = theano.function([self.input_var], 
                                        outputs=[self.critic_score, self.generator_score],
                                        givens={self.noise_var: noise})
                                                           
        if self.verbose: print ('Compile time: %.3f s' % (time.time()-start))
                               
    def train_iter(self, count, recorder):
        
        batches_train = self.data.batches_train

        if (self.generator_updates < 25) or (self.generator_updates % 500 == 0):
            critic_runs = 100
        else:
            critic_runs = 5
        
        c_score_list = []
        
        recorder.start()
        for _ in range(critic_runs):
            batch = next(batches_train)
            inputs, targets = batch
            c_score = self.critic_train_fn(inputs)
            c_score_list.append(c_score)
        self.critic_scores.extend(c_score_list)
        g_score = self.generator_train_fn()
        self.generator_scores.append(g_score)
        self.generator_updates += 1
        
        recorder.train_error(count, sum(c_score_list)/len(c_score_list), g_score)
        recorder.end('calc')
        
    def val_iter(self, count, recorder):
        
        batches_val = self.data.batches_val
        
        batch = next(batches_val)
        
        inputs, targets = batch
        
        c_score, g_score = self.val_fn(inputs)
        
        recorder.val_error(count, c_score, g_score, 0)
        
        
    def adjust_hyperp(self, epoch):
        
        import time

        print("  generator score:\t\t{}".format(np.mean(self.generator_scores)))
        print("  Wasserstein distance:\t\t{}".format(np.mean(self.critic_scores)))
        
        self.critic_scores[:] = []
        self.generator_scores[:] = []
        
        # And finally, we plot some generated data
        samples = self.gen_fn(lasagne.utils.floatX(np.random.rand(42, 100)))
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            plt.imsave('wgan_mnist_samples.png',
                       (samples.reshape(6, 7, 28, 28)
                               .transpose(0, 2, 1, 3)
                               .reshape(6*28, 7*28)),
                       cmap='gray')
        
        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            self.eta.set_value(lasagne.utils.floatX(self.initial_eta*2*(1 - progress)))
            
    def cleanup(self):
        
        pass
        
            
    def save(self, path='./'):
        
        if not os.path.exists(path):
            os.makedirs(path)
            print 'Creating folder: %s' % path
        
        np.savez(path+'%d_wgan_mnist_gen.npz' % self.epoch, *lasagne.layers.get_all_param_values(self.generator))
        np.savez(path+'%d_wgan_mnist_crit.npz' % self.epoch, *lasagne.layers.get_all_param_values(self.critic))
    
    def load(self, path_gen, path_cri):
        
        with np.load(path_gen) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.generator, param_values)
        
        with np.load(path_cri) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.critic, param_values)
        