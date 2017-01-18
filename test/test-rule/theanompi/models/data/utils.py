
def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    import cPickle
    data = cPickle.load(f)
    f.close()
    return data 
    
    
def get_bad_list(n_batches, commsize):
   
    bad_left = n_batches % commsize
    bad_left_list =[]
    for bad in range(bad_left):
        bad_left_list.append(n_batches-(bad+1)) 
    return bad_left_list
    
def extend_data(rank, size, img_batches, label_batches):
        
    _img_batches = img_batches
    n_files = len(_img_batches)
    _label_batches = label_batches[:n_files]  # cut unused labels 
       
    # get a list of training filenames that cannot be allocated to any rank
    bad_left_list = get_bad_list(n_files, size)
    if rank == 0: 
        print 'bad list is %s, nfiles is %d, size is %d' % (str(bad_left_list),n_files, size)
    need = (size - len(bad_left_list))  % size  
    if need !=0: 
        _img_batches.extend(_img_batches[-1*need:])
        _label_batches.extend(_label_batches[-1*need:])
    n_files = len(_img_batches)
    
    assert n_files % size == 0
    
    return _img_batches, _label_batches
