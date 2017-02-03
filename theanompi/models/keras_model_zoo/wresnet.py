from __future__ import print_function

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
from keras.engine.training import slice_X, make_batches

batch_size = 128
nb_classes = 10
nb_epoch = 200
data_augmentation = False
n = 4  # depth = 6*n + 4
k = 4  # widen factor

# the CIFAR10 images are 32x32 RGB with 10 labels
img_rows, img_cols = 32, 32
img_channels = 3


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
    
    def __init__(self):
        
        self.verbose = True #config['verbose']
        
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.Y_train = np_utils.to_categorical(y_train, nb_classes)
        self.Y_test = np_utils.to_categorical(y_test, nb_classes)
        
        self.build_model()
        
        self.X_train = X_train.astype('float32')
        self.X_test = X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255
        
        # iter related
        
        self.current_t = 0
        self.last_one_t=False
        self.current_v = 0
        self.last_one_v=False
        
        self.n_subb = 1
        
    def batch_data(self):
        
        x, y, sample_weights = self.model._standardize_user_data(
                    self.X_train, self.Y_train,
                    sample_weight=None,
                    class_weight=None,
                    check_batch_axis=False,
                    batch_size=batch_size)
        
        val_x, val_y, val_sample_weights = self.model._standardize_user_data(
                        self.X_test, self.Y_test,
                        sample_weight=None,
                        check_batch_axis=False,
                        batch_size=batch_size)
        
               
        
        ins = x + y + sample_weights
        val_ins = val_x + val_y + val_sample_weights
        
        if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins+=[1.]
            val_ins+=[0.]
        
        self.n_batch_train = ins[0].shape[0]/batch_size
        self.train_batches = []
        index_arr = range(ins[0].shape[0])
        for batch_index in range(self.n_batch_train):
            
            batch_ids = index_arr[batch_index * batch_size:
                                (batch_index+1)*batch_size]
                                        
            if isinstance(ins[-1], float):
                # do not slice the training phase flag
                ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_X(ins, batch_ids)
                                        
            self.train_batches.append(ins_batch)
            
            
        self.n_batch_val = val_ins[0].shape[0]/batch_size
        self.val_batches = []
        index_arr = range(val_ins[0].shape[0])
        for batch_index in range(self.n_batch_val):
            
            batch_ids = index_arr[batch_index * batch_size:
                                (batch_index+1)*batch_size]
                                        
            if isinstance(val_ins[-1], float):
                # do not slice the training phase flag
                ins_batch = slice_X(val_ins[:-1], batch_ids) + [val_ins[-1]]
            else:
                ins_batch = slice_X(val_ins, batch_ids)
                                        
            self.val_batches.append(ins_batch)
            
        
    
    
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
        
    def compile_iter_fns(self):
        
        import time
        
        start = time.time()
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        if self.verbose: print('Compiling......')
        self.model._make_train_function()
        self.model._make_test_function()
                      
        if self.verbose: print('Compile time: %.3f s' % (time.time()-start))
                      
    
    def train_iter(self, count, recorder):
        
        recorder.start()
        cost, acc = self.model.train_function(self.train_batches[self.current_t])
        recorder.train_error(count, cost, 1.0-acc)
        recorder.end('calc')
        
        if self.current_t == self.n_batch_train - 1:
            self.last_one_t = True
        else:
            self.last_one_t = False
        
        if self.last_one_t == False:
            self.current_t+=1
        else:
            self.current_t=0
              
    def val_iter(self, count, recorder):
        
        
        cost, acc = self.model.test_function(self.val_batches[self.current_v])
        
        recorder.val_error(count, cost, 1.0-acc, 0)
        
        if self.current_v == self.n_batch_val - 1:
            self.last_one_v = True
        else:
            self.last_one_v = False
        
        if self.last_one_v == False:
            self.current_v+=1
        else:
            self.current_v=0
        
        


if __name__=='__main__':
    
    model = Wide_ResNet()
    
    model.compile_iter_fns()
    
    model.batch_data()
    
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

    for batch_i in range(model.n_batch_train):

        for subb_i in range(model.n_subb):

            model.train_iter(batch_i, recorder)

        recorder.print_train_info(batch_i)

    # val
    for batch_j in range(model.n_batch_val):
    
        for subb_j in range(model.n_subb):
    
            model.val_iter(batch_i, recorder)
            
    recorder.print_val_info(batch_i)
    