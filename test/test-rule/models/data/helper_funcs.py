# This version of train_funcs.py is modified based on the theano_alexnet project. See the original project here:
# https://github.com/uoguelph-mlrg/theano_alexnet, and its copy right:
# Copyright (c) 2014, Weiguang Ding, Ruoyan Wang, Fei Mao and Graham Taylor
# All rights reserved.

import glob
import time
import os

import numpy as np

import hickle as hkl

np.random.seed(23455)

def unpack_configs(config, ext_data='.hkl', ext_label='.npy'):
    flag_para_load = config['para_load']
    flag_top_5 = config['flag_top_5']
    # Load Training/Validation Filenames and Labels
    train_folder = config['dir_head'] + config['train_folder']
    val_folder = config['dir_head'] + config['val_folder']
    label_folder = config['dir_head'] + config['label_folder']
    train_filenames = sorted(glob.glob(train_folder + '/*' + ext_data))
    val_filenames = sorted(glob.glob(val_folder + '/*' + ext_data))
    train_labels = np.load(label_folder + 'train_labels' + ext_label)
    val_labels = np.load(label_folder + 'val_labels' + ext_label)
    img_mean = np.load(config['dir_head'] + config['mean_file'])
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32')
    
    return (flag_para_load, flag_top_5,
            train_filenames, val_filenames, train_labels, val_labels, img_mean)


def extend_data(config,filenames, labels, env):

    size = config['size']
    rank = config['rank']
    file_batch_size = config['file_batch_size']
    
    lmdb_cur_list=None    
    
    if config['data_source']=='hkl':
    
        n_files = len(filenames)
        labels = labels[:n_files*file_batch_size]  # cut unused labels 
           
        # get a list of training filenames that cannot be allocated to any rank
        bad_left_list = get_bad_list(n_files, size)
        if rank == 0: print 'bad list is '+str(bad_left_list)
        need = (size - len(bad_left_list))  % size  
        if need !=0: 
            filenames.extend(filenames[-1*need:])
            labels=labels.tolist()
            labels.extend(labels[-1*need*file_batch_size:])
        n_files = len(filenames)
            
    elif config['data_source']=='lmdb':
        img_num = env.stat()['entries']
        n_files = img_num//file_batch_size # cut unused labels 
        labels = labels[:n_files*file_batch_size]
        
        bad_left_list = get_bad_list(n_files, size)
        if rank == 0: print 'bad list is ' + str(bad_left_list)
        need = (size - len(bad_left_list))  % size  
        
        lmdb_cur_list = [index*file_batch_size for index in range(n_files)]
        
        if need !=0: 
            lmdb_cur_list.extend(lmdb_cur_list[-1*need:])
            labels=labels.tolist()
            labels.extend(labels[-1*need*file_batch_size:])
            n_files = len(lmdb_cur_list)
            
    elif config['data_source']=='both':
    
        n_files = len(filenames)
        labels = labels[:n_files*file_batch_size] # cut unused labels 
         
        print 'total hkl files' , n_files          
        # get a list of training filenames that cannot be allocated to any rank
        bad_left_list = get_bad_list(n_files, size)
        if rank == 0: print 'bad list is '+str(bad_left_list)
        need = (size - len(bad_left_list))  % size  
        
        lmdb_cur_list = [index*file_batch_size for index in range(n_files)]
        
        if need !=0:
            lmdb_cur_list.extend(lmdb_cur_list[-1*need:])
            filenames.extend(filenames[-1*need:])
            labels=labels.tolist()
            labels.extend(labels[-1*need*file_batch_size:])
            n_files = len(filenames)
    
    return filenames, labels, lmdb_cur_list, n_files
    
def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    import cPickle
    data = cPickle.load(f)
    f.close()
    return data 
    

def get_rand3d(rand_crop, mode):  
    
    if mode == 'val':
        return np.float32([0.5, 0.5, 0])
        
    else:
        
        if rand_crop == True:
        
            # time_seed = int(time.time())*int(config['worker_id'])%1000
            # np.random.seed(time_seed)
            tmp_rand = np.float32(np.random.rand(3))
            tmp_rand[2] = round(tmp_rand[2])

            return tmp_rand
        else:
            return np.float32([0.5, 0.5, 0]) 
            
def get_params_crop_and_mirror(param_rand, data_shape, cropsize):

    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = int(round(param_rand[0] * center_margin * 2))
    crop_ys = int(round(param_rand[1] * center_margin * 2))
    if False:
        # this is true then exactly replicate Ryan's code, in the batch case
        crop_xs = int(math.floor(param_rand[0] * center_margin * 2))
        crop_ys = int(math.floor(param_rand[1] * center_margin * 2))

    flag_mirror = bool(round(param_rand[2]))

    return crop_xs, crop_ys, flag_mirror


def crop_and_mirror(data, param_rand, flag_batch=True, cropsize=227):
    '''
    when param_rand == (0.5, 0.5, 0), it means no randomness
    '''
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