import numpy as np

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

def get_rand3d(rand_crop, mode):  

    if rand_crop== False or mode == 'val':
        
        return np.float32([0.5, 0.5, 0]) 
        
    else:
        # time_seed = int(time.time())*int(config['worker_id'])%1000
        # np.random.seed(time_seed)
        tmp_rand = np.float32(np.random.rand(3))
        tmp_rand[2] = round(tmp_rand[2])

        return tmp_rand
        
            
def get_params_crop_and_mirror(param_rand, data_shape, cropsize):
    
    import math

    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = int(round(param_rand[0] * center_margin * 2))
    crop_ys = int(round(param_rand[1] * center_margin * 2))
    if False:
        # this is true then exactly replicate Ryan's code, in the batch case
        crop_xs = int(math.floor(param_rand[0] * center_margin * 2))
        crop_ys = int(math.floor(param_rand[1] * center_margin * 2))

    flag_mirror = bool(round(param_rand[2]))

    return crop_xs, crop_ys, flag_mirror


def crop_and_mirror(data, mode, rand_crop, flag_batch=True, cropsize=227):
    '''
    when param_rand == (0.5, 0.5, 0), it means no randomness
    '''
    
    
    
    param_rand = get_rand3d(rand_crop, mode)
    
    # print param_rand

    # if param_rand == (0.5, 0.5, 0), means no randomness and do validation
    if param_rand[0] == 0.5 and param_rand[1] == 0.5 and param_rand[2] == 0:
        flag_batch = True

    if flag_batch:
        # mirror and crop the whole batch
        crop_xs, crop_ys, flag_mirror = \
            get_params_crop_and_mirror(param_rand, data.shape, cropsize)

        # random mirror
        if flag_mirror:
            data = data[:, :, ::-1, :]

        # random crop
        data = data[:, crop_xs:crop_xs + cropsize,
                    crop_ys:crop_ys + cropsize, :]

    else:
        # mirror and crop each batch(each image?) individually
        # to ensure consistency, use the param_rand[1] as seed
        np.random.seed(int(10000 * param_rand[1]))

        data_out = np.zeros((data.shape[0], cropsize, cropsize,
                             data.shape[3])).astype('float32')

        for ind in range(data.shape[3]):
            # generate random numbers
            tmp_rand = np.float32(np.random.rand(3))
            tmp_rand[2] = round(tmp_rand[2])

            # get mirror/crop parameters
            crop_xs, crop_ys, flag_mirror = \
                get_params_crop_and_mirror(tmp_rand, data.shape, cropsize)

            # do image crop/mirror
            img = data[:, :, :, ind]
            if flag_mirror:
                img = img[:, :, ::-1]
            img = img[:, crop_xs:crop_xs + cropsize,
                      crop_ys:crop_ys + cropsize]
            data_out[:, :, :, ind] = img

        data = data_out

    return np.ascontiguousarray(data, dtype='float32')