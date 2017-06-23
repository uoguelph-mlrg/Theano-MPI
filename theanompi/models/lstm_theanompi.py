'''
code adjusted from tutorial:
http://deeplearning.net/tutorial/lstm.html

To use this modified version:
1.follow the tutorial to download data from stanford "Large Movie Review Dataset" and preprocess data.
2.download imdb.py (http://deeplearning.net/tutorial/code/imdb.py)
3.use the modified lstm.py in this repo (https://github.com/uoguelph-mlrg/Theano-MPI/blob/master/theanompi/models/lstm.py).
4.use session_lstm_2gpu.cfg in the example folder to start 2GPU training.
 
'''

import numpy

class LSTM(object):
    '''
    Build a tweet sentiment analyzer
    '''
    def __init__(self, config,
    
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=20,  # The maximum number of epoch to run
        dispFreq=40,  # Display to stdout the training progress every N updates
        decay_c=0.,  # Weight decay for the classifier applied to the U weights.
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000,  # Vocabulary size
        # optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        encoder='lstm',  # TODO: can be removed must be lstm.
        saveto='lstm_model.npz',  # The best model will be saved there
        validFreq=3000,  # Compute the validation error after this number of update.
        saveFreq=3000,  # Save the parameters after every saveFreq updates
        maxlen=500,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        dataset='imdb',

        # Parameter for extra option
        noise_std=0.,
        use_dropout=True,  # if False slightly faster, but worst test error
                           # This frequently need a bigger model.
        reload_model=None,  # Path to a saved model we want to start from.
        test_size=-1,  # If >0, we keep only this number of test example.
        ):
        
        # Model options
        self.model_options = locals().copy()
        
        self.verbose = config['verbose']
        self.rank = config['rank'] # will be used in sharding and distinguish rng
        self.size = config['size']
        
        import theano
        theano.config.on_unused_input = 'warn'
        self.name = 'LSTM'

        # if self.rank==0: print("model options", self.model_options)
        
        
        # data
        import os
        SEED = (self.rank+1)*123 #int(os.getpid())
        numpy.random.seed(SEED)
        
        import lstm
        from lstm import get_dataset, get_minibatches_idx
        lstm.SEED=SEED
        
        load_data, self.prepare_data = get_dataset(dataset)

        if self.rank==0: print('Loading data')
        train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                                       maxlen=maxlen)
                                       
        class IMDB_Data(object):
            
            def __init__(self, rank, size, train, valid, test, ):
                
                self.rank = rank
                self.size = size
                
                train,valid,test=self.shard_data(train,valid,test)
                
                self.rawdata=[train,valid,test]
                self.n_batch_train=len(train[0])// batch_size
                self.n_batch_val=len(valid[0])// valid_batch_size
                self.n_batch_test=len(test[0])// valid_batch_size
                
            def shard_data(self, train, valid, test):

                return numpy.array(train)[:,self.rank::self.size], \
                       numpy.array(valid)[:,self.rank::self.size], \
                       numpy.array( test)[:,self.rank::self.size],
                
            
        self.data=IMDB_Data(self.rank, self.size, train, valid, test)
                                       
        train, valid, test = self.data.rawdata
        self.train, self.valid, self.test = train, valid, test
        
        if test_size > 0:
            # The test set is sorted by size, but we want to keep random
            # size example.  So we must select a random selection of the
            # examples.
            idx = numpy.arange(len(test[0]))
            numpy.random.shuffle(idx)
            idx = idx[:test_size]
            test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
        
        ydim = numpy.max(train[1]) + 1

        self.model_options['ydim'] = ydim
        


        self.kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
        self.kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
        # Get new shuffled index for the training set.
        self.kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
        from itertools import cycle
        self.kf_iter = cycle(self.kf)

        if self.rank==0: print("%d train examples" % len(train[0]))
        if self.rank==0: print("%d valid examples" % len(valid[0]))
        if self.rank==0: print("%d test examples" % len(test[0]))
        
        
        # model
        self.build_model()
        
        # if self.rank==0: print('Optimization')
        
        # training
        
        self.history_errs = []
        self.best_p = None
        self.bad_count = 0

        if validFreq == -1:
            validFreq = len(train[0]) // batch_size
        if saveFreq == -1:
            saveFreq = len(train[0]) // batch_size

        self.uidx = 0  # the number of update done
        self.estop = False  # early stop
        
        self.n_samples = 0
        
        self.eidx=0
        self.epoch=self.eidx
        self.n_epochs=max_epochs
        self.n_subb=1
        
        
    
    def build_model(self):
        
        import theano
        
        from lstm import build_model, init_tparams,init_params,load_params
        

        if self.rank==0: print('Building model')
        
        
        # This create the initial parameters as numpy ndarrays.
        # Dict name (string) -> numpy ndarray

        params = init_params(self.model_options)

        if self.model_options['reload_model']:
            load_params('lstm_model.npz', params)

        tparams = init_tparams(params)

        (self.use_noise, self.x, self.mask,
        self.y, self.cost, self.pred) = build_model(tparams, self.model_options)

        if self.model_options['decay_c'] > 0.:
             decay_c = theano.shared(numpy_floatX(self.model_options['decay_c']), name='decay_c')
             weight_decay = 0.
             weight_decay += (tparams['U'] ** 2).sum()
             weight_decay *= decay_c
             self.cost += weight_decay
             
        self.tparams = tparams
        
        self.params = self.tparams.values()
        
        
    def compile_iter_fns(self, *args, **kwargs):
        
        import theano
        
        import time
        start=time.time()
        
        # f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
        self.f_pred = theano.function([self.x, self.mask], self.pred.argmax(axis=1), name='f_pred')
        
        # f_cost = theano.function([x, mask, y], cost, name='f_cost')
        import theano.tensor as tensor
        grads = tensor.grad(self.cost, wrt=list(self.tparams.values()))
        # f_grad = theano.function([x, mask, y], grads, name='f_grad')

        lr = tensor.scalar(name='lr')
        
        from lstm import adadelta
        self.f_grad_shared, self.f_update = adadelta(lr, self.tparams, grads,
                                         self.x, self.mask, self.y, self.cost)
        
        if self.rank==0: print('compile time %.3f' % (time.time()-start))
                                             
        
        
    def train_iter(self, count,recorder):
        
        from lstm import unzip
        
        recorder.start()
        
        train_index = next(self.kf_iter)[1]
        
        self.uidx += 1
        self.use_noise.set_value(1.)

        # Select the random examples for this minibatch
        y = [self.train[1][t] for t in train_index]
        x = [self.train[0][t]for t in train_index]
         
        # Get the data in numpy.ndarray format
        # This swap the axis!
        # Return something of shape (minibatch maxlen, n samples)
        x, mask, y = self.prepare_data(x, y)
        self.n_samples += x.shape[1]
        
        recorder.end('wait')
        
        recorder.start()
        
        cost = self.f_grad_shared(x, mask, y)
        self.f_update(self.model_options['lrate'])
        
        recorder.train_error(self.uidx, cost, 0)
        
        recorder.end('calc')
        
        if numpy.isnan(cost) or numpy.isinf(cost):
            if self.rank==0: print('bad cost detected: ', cost)
            return 1., 1., 1.

        if numpy.mod(self.uidx, self.model_options['dispFreq']) == 0:
            self.eidx=self.uidx //self.data.n_batch_train
            self.epoch=self.eidx
            if self.rank==0: print('Epoch ', self.eidx, 'Update %d/%d' % (self.uidx ,self.data.n_batch_train) , 'Cost ', cost)

        if self.model_options['saveto'] and numpy.mod(self.uidx, self.model_options['saveFreq']) == 0:
            if self.rank==0: print('Saving...')

            if self.best_p is not None:
                params = self.best_p
            else:
                params = unzip(self.tparams)
            numpy.savez(self.model_options['saveto'], history_errs=self.history_errs, **params)
            # import pickle
            #
            # pickle.dump(self.model_options, open('%s.pkl' % self.model_options['saveto'], 'wb'), -1)
            if self.rank==0: print('Done')
            
        
        
    def val_iter(self, count, recorder):
        
    # if numpy.mod(self.uidx, self.model_options['validFreq']) == 0:
        self.use_noise.set_value(0.)
        from lstm import pred_error
        train_err = pred_error(self.f_pred, self.prepare_data, self.train, self.kf)
        valid_err = pred_error(self.f_pred, self.prepare_data, self.valid,
                               self.kf_valid)
        test_err = pred_error(self.f_pred, self.prepare_data, self.test, self.kf_test)

        self.history_errs.append([valid_err, test_err])
        
        recorder.val_error(self.uidx, train_err, valid_err, test_err)

        if (self.best_p is None or
            valid_err <= numpy.array(self.history_errs)[:,
                                                   0].min()):
            
            from lstm import unzip
            self.best_p = unzip(self.tparams)
            self.bad_counter = 0

        if self.rank==0: print('Train ', train_err, 'Valid ', valid_err,
               'Test ', test_err)
        if self.rank==0: print('Seen %d samples' % self.n_samples)
        
        patience = self.model_options['patience']
        if (len(self.history_errs) > patience and
            valid_err >= numpy.array(self.history_errs)[:-patience,
                                                   0].min()):
            self.bad_counter += 1
            if self.bad_counter > patience:
                if self.rank==0: print('Early Stop!')
                self.estop = True
                
        if self.epoch > self.model_options['max_epochs']:
            self.estop=True
        
        if self.estop == True:
            return 'stop'
        else:
            return self.data.n_batch_val
        
    def reset_iter(self, *args, **kwargs):
        
        pass
        
    def adjust_hyperp(self,*args, **kwargs):
        pass
        
    def scale_lr(self,*args, **kwargs):
        pass
        
    def cleanup(self,*args, **kwargs):
        
        from lstm import zipp, unzip, get_minibatches_idx, pred_error
        
        if self.best_p is not None:
            zipp(self.best_p, self.tparams)
        else:
            self.best_p = unzip(self.tparams)

        self.use_noise.set_value(0.)
        kf_train_sorted = get_minibatches_idx(len(self.train[0]), self.model_options['batch_size'])
        train_err = pred_error(self.f_pred, self.prepare_data, self.train, kf_train_sorted)
        valid_err = pred_error(self.f_pred, self.prepare_data, self.valid, kf_valid)
        test_err = pred_error(self.f_pred, self.prepare_data, self.test, kf_test)

        if self.rank==0: print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
        if saveto:
            numpy.savez(self.model_options['saveto'], train_err=train_err,
                        valid_err=valid_err, test_err=test_err,
                        history_errs=self.history_errs, **self.best_p)
        # print('The code run for %d epochs, with %f sec/epochs' % (
        #     (self.eidx + 1), (end_time - start_time) / (1. * (self.eidx + 1))))
        # print( ('Training took %.1fs' %
        #         (end_time - start_time)), file=sys.stderr)
        
         
         
         
         