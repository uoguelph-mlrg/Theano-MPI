
if __name__ == '__main__':
    
    import sys
    modelfile = sys.argv[1]
    modelclass = sys.argv[2]
    
    # setting up device
    device='cuda0'
    backend='cudandarray' if device.startswith('gpu') else 'gpuarray'
    if backend=='gpuarray':
        import os
        if 'THEANO_FLAGS' in os.environ:
            raise ValueError('Use theanorc to set the theano config')
        os.environ['THEANO_FLAGS'] = 'device={0},'.format(device) #+'gpuarray.single_stream=False' +',cycle_detection=fast'
        import theano.gpuarray
        # This is a bit of black magic that may stop working in future
        # theano releases
        ctx = theano.gpuarray.type.get_context(None)
    else:
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(device)
    
    config={}
    config['verbose'] = True
    # config['device'] = 'cuda0'
    config['rank'] = 0
    config['size'] = 1
    
    
    import importlib
    mod = importlib.import_module(modelfile)
    modcls = getattr(mod, modelclass)
    
    model = modcls(config)
    

    model.compile_iter_fns(sync_type='cdd')
    
    
    # load data

    img_mean=model.data.rawdata[-1]
    
    import hickle as hkl
    arr = hkl.load(model.data.train_img_shard[0]) - img_mean
    
    from theanompi.models.data.utils import crop_and_mirror
    
    arr = crop_and_mirror(arr, mode='val', 
                        rand_crop=True, 
                        flag_batch=True, 
                        cropsize=224)
                        
 
    model.shared_x.set_value(arr)
        
    model.shared_y.set_value(model.data.train_labels_shard[0])
    
    
    # benchmarking
    
    import time
    
    t0=time.time()

    for i in range(100):
        cost,error= model.train_iter_fn(0)
            
    for p in model.params:
        p.get_value(borrow=True, return_internal_type=True).sync()
        
    t1=time.time()
    
    
    print 'function execution time: %.4f, cost:%.2f, error:%.2f' % ((t1-t0), cost, error)
    
    
        