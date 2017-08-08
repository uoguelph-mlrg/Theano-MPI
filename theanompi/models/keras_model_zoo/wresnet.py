from __future__ import print_function

import numpy as np
from keras.datasets import cifar10
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.layers import merge, Input
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 90
data_augmentation = False
n = 4  # depth = 6*n + 4
k = 4  # widen factor

# the CIFAR10 images are 32x32 RGB with 10 labels
img_rows, img_cols = 32, 32
img_channels = 3
learninig_rate=0.001
lr_policy='step'
lr_step = [50, 70, 90]

def bottleneck(incoming, count, nb_in_filters, nb_out_filters, dropout=None, subsample=(2, 2)):
    outgoing = wide_basic(incoming, nb_in_filters, nb_out_filters, dropout, subsample)
    for i in range(1, count):
        outgoing = wide_basic(outgoing, nb_out_filters, nb_out_filters, dropout, subsample=(1, 1))

    return outgoing


def wide_basic(incoming, nb_in_filters, nb_out_filters, dropout=None, subsample=(2, 2)):
    nb_bottleneck_filter = nb_out_filters

    if nb_in_filters == nb_out_filters:
        # conv3x3
        y = BatchNormalization(mode=0, axis=1)(incoming)
        y = Activation('relu')(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Convolution2D(nb_bottleneck_filter, nb_row=3, nb_col=3,
                          subsample=subsample, init='he_normal', border_mode='valid')(y)

        # conv3x3
        y = BatchNormalization(mode=0, axis=1)(y)
        y = Activation('relu')(y)
        if dropout is not None:
            y = Dropout(dropout)(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Convolution2D(nb_bottleneck_filter, nb_row=3, nb_col=3,
                          subsample=(1, 1), init='he_normal', border_mode='valid')(y)

        return merge([incoming, y], mode='sum')

    else:  # Residual Units for increasing dimensions
        # common BN, ReLU
        shortcut = BatchNormalization(mode=0, axis=1)(incoming)
        shortcut = Activation('relu')(shortcut)

        # conv3x3
        y = ZeroPadding2D((1, 1))(shortcut)
        y = Convolution2D(nb_bottleneck_filter, nb_row=3, nb_col=3,
                          subsample=subsample, init='he_normal', border_mode='valid')(y)

        # conv3x3
        y = BatchNormalization(mode=0, axis=1)(y)
        y = Activation('relu')(y)
        if dropout is not None:
            y = Dropout(dropout)(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Convolution2D(nb_out_filters, nb_row=3, nb_col=3,
                          subsample=(1, 1), init='he_normal', border_mode='valid')(y)

        # shortcut
        shortcut = Convolution2D(nb_out_filters, nb_row=1, nb_col=1,
                                 subsample=subsample, init='he_normal', border_mode='same')(shortcut)

        return merge([shortcut, y], mode='sum')


class Wide_ResNet(object):
    
    '''
    Modified from:
    https://gist.github.com/kashif/0ba0270279a0f38280423754cea2ee1e
    '''
    
    def __init__(self, config):
        
        self.verbose = config['verbose']
        self.size = config['size']
        self.rank = config['rank']
        
        self.name = 'Wide_ResNet'
        
        
        # data
        from data.cifar10 import Cifar10_data
        self.data = Cifar10_data(verbose=False)

        self.build_model()

        # iter related
        
        self.current_t = 0
        self.last_one_t=False
        self.current_v = 0
        self.last_one_v=False
        
        self.n_subb = 1
        self.n_epochs = nb_epoch
        self.epoch=0
    
    
    def build_model(self):
    

        img_input = Input(shape=(img_channels, img_rows, img_cols))

        # one conv at the beginning (spatial size: 32x32)
        x = ZeroPadding2D((1, 1))(img_input)
        x = Convolution2D(16, nb_row=3, nb_col=3)(x)

        # Stage 1 (spatial size: 32x32)
        x = bottleneck(x, n, 16, 16 * k, dropout=0.3, subsample=(1, 1))
        # Stage 2 (spatial size: 16x16)
        x = bottleneck(x, n, 16 * k, 32 * k, dropout=0.3, subsample=(2, 2))
        # Stage 3 (spatial size: 8x8)
        x = bottleneck(x, n, 32 * k, 64 * k, dropout=0.3, subsample=(2, 2))

        x = BatchNormalization(mode=0, axis=1)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8), strides=(1, 1))(x)
        x = Flatten()(x)
        preds = Dense(nb_classes, activation='softmax')(x)

        self.model = Model(input=img_input, output=preds)
        
        self.keras_get_params()
        
    def compile_iter_fns(self, *args, **kwargs):
        
        try:
            sync_type = kwargs['sync_type']
        except:
            raise RuntimeError('Keras wresnet needs sync_type keyword argument')
        
        if sync_type != 'avg':
            raise RuntimeError('currently wresnet only support compiling with sync_type avg, add BSP_SYNC_TYPE=avg in your session cfg')
        
        import time
        
        start = time.time()
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        self.shared_lr = self.model.optimizer.lr
        self.base_lr = self.shared_lr.get_value()
        
        if self.verbose: print('Compiling......')
        self.model._make_train_function()
        self.model._make_test_function()
                      
        if self.verbose: print('Compile time: %.3f s' % (time.time()-start))
        
        self.data.batch_data(self.model, batch_size)
        self.data.extend_data(rank=self.rank, size=self.size)
        self.data.shuffle_data(mode='train', common_seed=1234)
        self.data.shuffle_data(mode='val')
        self.data.shard_data(mode='train', rank=self.rank, size=self.size) # to update data.n_batch_train
        self.data.shard_data(mode='val', rank=self.rank, size=self.size) # to update data.n_batch_val
                      
    
    def train_iter(self, count, recorder):
        
        
        if self.current_t ==0:
            self.data.shuffle_data(mode='train',common_seed=self.epoch)
            self.data.shard_data(mode='train',rank=self.rank, size=self.size)
        
        recorder.start()
        cost, acc = self.model.train_function(self.data.train_batches_shard[self.current_t])
        recorder.train_error(count, cost, 1.0-acc)
        recorder.end('calc')
        
        if self.current_t == self.data.n_batch_train - 1:
            self.last_one_t = True
        else:
            self.last_one_t = False
        
        if self.last_one_t == False:
            self.current_t+=1
        else:
            self.current_t=0
              
    def val_iter(self, count, recorder):
        
        if self.current_v==0:
            self.data.shuffle_data(mode='val')
            self.data.shard_data(mode='val',rank=self.rank, size=self.size)
            
        cost, acc = self.model.test_function(self.data.val_batches_shard[self.current_v])
        
        recorder.val_error(count, cost, 1.0-acc, 0)
        
        if self.current_v == self.data.n_batch_val - 1:
            self.last_one_v = True
        else:
            self.last_one_v = False
        
        if self.last_one_v == False:
            self.current_v+=1
        else:
            self.current_v=0
            
    def reset_iter(self, mode):
        
        '''used at the begininig of another mode'''
        
        if mode=='train':

            self.current_t = 0
            self.subb_t=0
            self.last_one_t = False
        else:

            self.current_v = 0
            self.subb_v=0
            self.last_one_v = False
            
    def adjust_hyperp(self, epoch):
        
        'to be called once per epoch'
    
        if lr_policy == 'step':

            if epoch in lr_step: 
    
                tuned_base_lr = self.shared_lr.get_value() /10.

                self.shared_lr.set_value(np.float32(tuned_base_lr))
        
    def scale_lr(self, size):
        
        self.shared_lr.set_value(np.array(self.base_lr*size, dtype='float32'))
        
        
    def cleanup(self):
        
        pass
        
    def keras_get_params(self):
        
        self.params=[]
        
        for l in self.model.layers:
        
            self.params.extend(l.trainable_weights)
        
        
        
        


if __name__=='__main__':
    
    
    config={}
    config['verbose'] = True

        
    model = Wide_ResNet(config)
    
    model.compile_iter_fns()
    
    # get recorder
    from theanompi.lib.recorder import Recorder
    recorder = Recorder(comm=None, printFreq=5120/batch_size, modelname='wide_resnet', verbose=True)
    
    # do somethind similar to:
    # self.model.fit(X_train, Y_train,
    #       batch_size=batch_size,
    #       nb_epoch=nb_epoch,
    #       validation_data=(X_test, Y_test),
    #       shuffle=True)
          
          
    # train

    for batch_i in range(model.data.n_batch_train):

        for subb_i in range(model.n_subb):

            model.train_iter(batch_i, recorder)

        recorder.print_train_info(batch_i)

    # val
    for batch_j in range(model.data.n_batch_val):
    
        for subb_j in range(model.n_subb):
    
            model.val_iter(batch_i, recorder)
            
    recorder.print_val_info(batch_i)
    