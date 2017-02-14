
if __name__ == '__main__':
    
    import sys
    modelfile = sys.argv[1]
    modelclass = sys.argv[2]
        
    
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    
    # setting up device
    import os
    if 'THEANO_FLAGS' in os.environ:
        raise ValueError('Use theanorc to set the theano config')
    os.environ['THEANO_FLAGS'] = 'device={0}'.format('cuda0')
    import theano.gpuarray
    # This is a bit of black magic that may stop working in future
    # theano releases
    ctx = theano.gpuarray.type.get_context(None)
    
    config={}
    config['verbose'] = True
    # config['device'] = 'cuda0'
    config['rank'] = comm.rank
    config['size'] = comm.size
    
    
    import importlib
    mod = importlib.import_module(modelfile)
    modcls = getattr(mod, modelclass)
    
    model = modcls(config)
    
    # get recorder
    from theanompi.lib.recorder import Recorder
    recorder = Recorder(comm, printFreq=5120/model.file_batch_size, modelname=modelclass, verbose=True)
    
    
    model.compile_iter_fns()
    
    snapshot_freq= 5
    snapshot_path= './snapshots/'
    count=0
    
    for epoch in range(model.n_epochs):
            
        model.epoch=epoch
        
        recorder.start_epoch()
        # train
    
        for batch_i in range(model.data.n_batch_train):
        
            for subb_i in range(model.n_subb):
        
                model.train_iter(batch_i, recorder)
                
                count+=1
        
            recorder.print_train_info(batch_i)
        
        # val
        for batch_j in range(model.data.n_batch_val):
        
            for subb_j in range(model.n_subb):
        
                model.val_iter(batch_i, recorder)

        #recorder.gather_val_info()
        recorder.print_val_info(batch_i)
        
        model.adjust_hyperp(epoch=epoch)
        
        recorder.save(count, model.shared_lr.get_value())
            
        if epoch % snapshot_freq == 0: 
            from theanompi.lib.helper_funcs import save_model
            save_model(model, snapshot_path, verbose=True)
                    
        model.epoch+=1

    
    
    
    model.cleanup()
    
    exit(0)
    
    
    # inference demo
    
    batch_size = 1
    batch_crop_mirror = True
    
    # or
    # batch_size=batch_size
    # batch_crop_mirror=False
    
    trained_params = [param.get_value() for param in model.params]
    
    model = GoogLeNet(config)
    
    for p, p_old in zip(model.params, trained_params):
        p.set_value(p_old)
    
    model.compile_inference()
    
    
    
    test_image = np.zeros((3,227,227,1),dtype=theano.config.floatX) # inference on an image 
    
    soft_prob = model.inf_fn(test_image)
    
    num_top = 5
    
    y_pred_top_x = np.argsort(soft_prob, axis=1)[:, -num_top:] # prob sorted from small to large
    
    print ''
    print 'top-5 prob:'
    print y_pred_top_x[0]
    
    
    print ''
    print 'top-5 prob catagories:'
    print [soft_prob[0][index] for index in y_pred_top_x[0]]
    
    print ''
    # git clone https://github.com/hma02/show_batch.git
    # run mk_label_dict.py to generate label_dict.npy
    label_dict = np.load('label_dict.npy').tolist()
    
    print 'discription:'
    for cat in y_pred_top_x[0]:
        print "%s: %s" % (cat,label_dict[cat])
        print ''
    