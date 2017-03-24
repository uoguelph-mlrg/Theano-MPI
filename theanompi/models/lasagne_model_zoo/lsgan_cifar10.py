"""
Code adjusted from https://gist.github.com/f0k/9b0bb51040719eeafec7eba473a9e79b

"""

from __future__ import (absolute_import, print_function)

import numpy as np

import theano
import theano.tensor as T
import lasagne

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
    # # fully-connected layer
    # layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 1024*4*4))
    layer = ReshapeLayer(layer, ([0], 1024, 4, 4))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 512, 5, stride=2, crop='same',
                                     output_size=8))
    layer = batch_norm(Deconv2DLayer(layer, 256, 5, stride=2, crop='same',
                                     output_size=16))
    layer = Deconv2DLayer(layer, 3, 5, stride=2, crop='same', output_size=32,
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
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 256, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 512, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # # fully-connected layer
    # layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer (linear)
    layer = DenseLayer(layer, 1, nonlinearity=None)
    print ("critic output:", layer.output_shape)
    return layer
    

num_epochs=100
epochsize=100
batchsize=128
initial_eta=1e-4
         
class LSGAN(object):
    
    def __init__(self, config):
        
        self.verbose = config['verbose']
        self.rank = config['rank']
        self.size = config['size']
        
        self.name = 'LeastSquare_GAN'
        
        # data
        from theanompi.models.data.cifar10 import Cifar10_data

        self.data = Cifar10_data(self.verbose)
        self.data.rawdata[0] = self.data.rawdata[0]/np.float32(255.)
        self.data.rawdata[2] = self.data.rawdata[2]/np.float32(255.)
        self.data.batch_data(batchsize)
        
        self.batch_size=batchsize
        self.file_batch_size=batchsize
        self.n_subb=self.file_batch_size/self.batch_size
        
        # model
        self.build_model()
        
        self.params = self.critic_params #+self.generator_params
        
        # training related
        self.epoch=0
        self.n_epochs=num_epochs
        
        self.generator_updates = 0
        self.critic_scores = []
        self.generator_scores = []
        self.c_list=[]
        self.g_list=[]
        self.current_info=None
        
        self.init_view=False
        
    def build_model(self):
        
        rng=np.random.RandomState(1234)
        lasagne.random.set_rng(rng)
        
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
        
    def compile_iter_fns(self, *args, **kwargs):
        
        # Create loss expressions to be minimized
        # a, b, c = -1, 1, 0  # Equation (8) in the paper
        a, b, c = 0, 1, 1  # Equation (9) in the paper
        loss_gen = lasagne.objectives.squared_error(self.fake_out, c).mean()
        # loss_gen = -1*self.fake_out.mean()
        loss_critic = (lasagne.objectives.squared_error(self.real_out, b).mean() +
                       lasagne.objectives.squared_error(self.fake_out, a).mean())
        # loss_critic = self.real_out.mean() - self.fake_out.mean()
        self.shared_lr = theano.shared(lasagne.utils.floatX(initial_eta))
        
        generator_updates = lasagne.updates.rmsprop(
                loss_gen, self.generator_params, learning_rate=self.shared_lr)
        critic_updates = lasagne.updates.rmsprop(
                loss_critic, self.critic_params, learning_rate=self.shared_lr)
            
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
        
        c_score_list = []
        recorder.start()

        inputs, targets = next(batches_train)

        c_score = self.critic_train_fn(inputs)
        c_score_list.append(c_score)
        self.critic_scores.extend(c_score_list)
        
        g_score = self.generator_train_fn()
        
        self.generator_scores.append(g_score)
        
        # print(c_score,g_score)
        
        recorder.train_error(count, sum(c_score_list)/len(c_score_list), g_score)
        recorder.end('calc')
        
    def val_iter(self, count, recorder):
        
        batches_val = self.data.batches_val
        
        inputs, targets = next(batches_val)
        
        c_score, g_score = self.val_fn(inputs)
        
        recorder.val_error(count, c_score, g_score, 0)  # print loss_critic, loss_gen and a 0 instead of cost, error and error_top_5 
    
    def reset_iter(self, *args, **kwargs):
        pass
        
    def print_info(self, recorder):
        
        print('\nEpoch %d' % self.epoch)
        g_=np.mean(self.generator_scores)
        c_=np.mean(self.critic_scores)
        self.g_list.extend([g_])
        self.c_list.extend([c_])
        print("  generator loss:\t\t{}".format(g_))
        print("  critic loss:\t\t{}".format(c_))
        
        self.critic_scores[:] = []
        self.generator_scores[:] = []
        
        samples = self.gen_fn(lasagne.utils.floatX(np.random.rand(4*4, 100)))
        samples = samples.reshape(4, 4, 3, 32, 32).transpose(0, 3, 1, 4, 2).reshape(4*32, 4*32, 3)
        
        if self.init_view == False:
            self.init_view = True
            self.save_flag=False
            recorder.plot_init(name='scores', save=self.save_flag)
            recorder.plot_init(name='sample', save=self.save_flag)
            
        recorder.plot(name='sample', image=samples)
        recorder.plot(name='scores', lines=[self.c_list,self.g_list], lw=2, save=self.save_flag)
                       
        
    def adjust_hyperp(self, epoch):

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            self.shared_lr.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))
            
    def cleanup(self):
        
        pass
        
            
    def save(self, path='./'):
        
        import os
        if not os.path.exists(path):
            print('Creating folder: %s' % path)
            os.makedirs(path)
        
        np.savez(path+'%d_lsgan_cifar10_gen.npz' % self.epoch, *lasagne.layers.get_all_param_values(self.generator))
        np.savez(path+'%d_lsgan_cifar10_crit.npz' % self.epoch, *lasagne.layers.get_all_param_values(self.critic))
    
    def load(self, path_gen, path_cri):
        
        with np.load(path_gen) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.generator, param_values)
        
        with np.load(path_cri) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.critic, param_values)
        
if __name__ == '__main__':
    
    raise RuntimeError('to be tested using test_model.py:\n$ python test_model.py lasagne_model_zoo.lsgan LSGAN')
        