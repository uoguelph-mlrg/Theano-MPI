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
def rmsprop(cost, params, learning_rate, momentum=0.5, rescale=5.):
    
    grads = T.grad(cost=cost, wrt=params)
    
    running_square_ = [theano.shared(np.zeros_like(p.get_value(),dtype=p.dtype), broadcastable=p.broadcastable)
                      for p in params]
    running_avg_ = [theano.shared(np.zeros_like(p.get_value(),dtype=p.dtype), broadcastable=p.broadcastable)
                   for p in params]
    memory_ = [theano.shared(np.zeros_like(p.get_value(),dtype=p.dtype), broadcastable=p.broadcastable)
                       for p in params]
    
    grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    grad_norm = T.sqrt(grad_norm)
    scaling_num = rescale
    scaling_den = T.maximum(rescale, grad_norm)
    # Magic constants
    combination_coeff = 0.9
    minimum_grad = 1E-4
    updates = []
    for n, (param, grad) in enumerate(zip(params, grads)):
       grad = T.switch(not_finite, 0.1 * param,
                       grad * (scaling_num / scaling_den))
       old_square = running_square_[n]
       new_square = combination_coeff * old_square + (
           1. - combination_coeff) * T.sqr(grad)
       old_avg = running_avg_[n]
       new_avg = combination_coeff * old_avg + (
           1. - combination_coeff) * grad
       rms_grad = T.sqrt(new_square - new_avg ** 2)
       rms_grad = T.maximum(rms_grad, minimum_grad)
       memory = memory_[n]
       update = momentum * memory - learning_rate * grad / rms_grad

       update2 = momentum * momentum * memory - (
           1 + momentum) * learning_rate * grad / rms_grad
           
       updates.append((old_square, new_square))
       updates.append((old_avg, new_avg))
       updates.append((memory, update))
       updates.append((param, param + update2))
    return updates

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
    layer = batch_norm(Deconv2DLayer(layer, 128, 5, stride=2, crop='same',
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
    layer = Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu)
    layer = Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu)
    # fully-connected layer
    layer = DenseLayer(layer, 1024, nonlinearity=lrelu)
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
        self.current_info=None
        
        
    def build_model(self):
        
        # Prepare Theano variables for inputs and targets
        self.noise_var = T.matrix('noise')
        self.input_var = T.tensor4('inputs')
        
        # Create neural network model
        generator = build_generator(self.noise_var)
        critic = build_critic(self.input_var)
        
        # Create expression for passing real data through the critic
        self.real_out = lasagne.layers.get_output(critic)
        # Create expression for passing fake data through the critic
        self.fake_out = lasagne.layers.get_output(critic,
                lasagne.layers.get_output(generator))
        
        
        # Create update expressions for training
        self.generator_params = lasagne.layers.get_all_params(generator, trainable=True)
        self.critic_params = lasagne.layers.get_all_params(critic, trainable=True)
        self.generator = generator
        self.critic = critic
        
    def compile_iter_fns(self):
        
        
        eta = theano.shared(lasagne.utils.floatX(initial_eta))
        self.eta=eta
        self.shared_lr=eta
        
        
        loss_critic = self.real_out.mean() - self.fake_out.mean()
        critic_updates = rmsprop(
                -1*loss_critic, self.critic_params, learning_rate=eta)
                
        loss_gen = -1*self.fake_out.mean()
        generator_updates = rmsprop(
                loss_gen, self.generator_params, learning_rate=eta)
                
                
        # Clip critic parameters in a limited range around zero (except biases)
        critic_clip_updates=[]
        for param in lasagne.layers.get_all_params(self.critic, trainable=True,
                                                   regularizable=True):
                                                   
            critic_clip_updates.append([param, T.clip(param, -clip, clip)])
            
            
        # Instantiate a symbolic noise generator to use for training
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
        noise = srng.uniform((batchsize, 100))

        # Compile functions performing a training step on a mini-batch (according
        # to the updates dictionary) and returning the corresponding score:
        print('Compiling...')
        
        import time
        
        start = time.time()
        
        self.generator_train_fn = theano.function([], loss_gen,
                                             givens={self.noise_var: noise},
                                             updates=generator_updates)
        self.critic_train_fn = theano.function([self.input_var],loss_critic,
                                          givens={self.noise_var: noise},
                                          updates=critic_updates)
        self.critic_clip_fn = theano.function([],updates=critic_clip_updates)

        # Compile another function generating some data
        self.gen_fn = theano.function([self.noise_var],
                                 lasagne.layers.get_output(self.generator,
                                                           deterministic=True))
                                                           
        self.val_fn = theano.function([self.input_var], 
                                        outputs=[loss_critic, loss_gen],
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
            self.critic_clip_fn()
            
            count+=1
            
            
        self.critic_scores.extend(c_score_list)
        g_score = self.generator_train_fn()
        self.generator_scores.append(g_score)
        self.generator_updates += 1
        
        recorder.train_error(count, sum(c_score_list)/len(c_score_list), g_score)
        recorder.end('calc')
        
        return count
        
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
            sample_path='./samples/'
            import os
            if not os.path.exists(sample_path):
                print('Creating folder: %s' % sample_path)
                os.makedirs(sample_path)
                
            plt.imsave(sample_path+'%dwgan_mnist_samples.png' % epoch,
                       (samples.reshape(6, 7, 28, 28)
                               .transpose(0, 2, 1, 3)
                               .reshape(6*28, 7*28)),
                       cmap='gray')
        
        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            self.eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))
            
    def cleanup(self):
        
        pass
        
            
    def save(self, path='./'):
        
        import os
        if not os.path.exists(path):
            print('Creating folder: %s' % path)
            os.makedirs(path)
        
        np.savez(path+'%d_wgan_mnist_gen.npz' % self.epoch, *lasagne.layers.get_all_param_values(self.generator))
        np.savez(path+'%d_wgan_mnist_crit.npz' % self.epoch, *lasagne.layers.get_all_param_values(self.critic))
    
    def load(self, path_gen, path_cri):
        
        with np.load(path_gen) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.generator, param_values)
        
        with np.load(path_cri) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.critic, param_values)
        
if __name__ == '__main__':
    
    raise RuntimeError('to be tested using test_model.py:\n$ python test_model.py lasagne_model_zoo.wgan WGAN')
        