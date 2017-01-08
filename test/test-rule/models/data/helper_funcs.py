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