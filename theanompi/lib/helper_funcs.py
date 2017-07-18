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



# for CUDA-aware MPI
bufint = lambda arr: memoryview(arr)

def dtype_to_mpi(t):
    from mpi4py import MPI
    if hasattr(MPI, '_typedict'):
        mpi_type = MPI._typedict[np.dtype(t).char]
    elif hasattr(MPI, '__TypeDict__'):
        mpi_type = MPI.__TypeDict__[np.dtype(t).char]
    else:
        raise ValueError('cannot convert type')
    return mpi_type


        
def save_weights(layers, weights_dir, epoch):
    
    if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            print('Creating folder: %s' % weights_dir)
            
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.save_weight(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.save_weight(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.save_weight(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.save_weight(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.save_weight(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.save_weight(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))



def load_weights(layers, weights_dir, epoch, l_range=None):
    
    for idx in range(len(layers)):
        
        if l_range !=None and (idx not in l_range):
            continue
            
        if hasattr(layers[idx], 'W'):
            layers[idx].W.load_weight(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.load_weight(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.load_weight(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.load_weight(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.load_weight(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.load_weight(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))

def collect_weight_path(layers, weights_dir, epoch, l_range=None):
    
    weight_path_list = []
    
    for idx in range(len(layers)):
        
        if l_range !=None and (idx not in l_range):
            continue
            
        if hasattr(layers[idx], 'W'):
            weight_path_list.append(
                weights_dir + 'W' + '_' + str(idx) + '_' + str(epoch) + '.npy')
        if hasattr(layers[idx], 'W0'):
            weight_path_list.append(
                weights_dir+ 'W0' + '_' + str(idx) + '_' + str(epoch)+ '.npy')
        if hasattr(layers[idx], 'W1'):
            weight_path_list.append(
                weights_dir+ 'W1' + '_' + str(idx) + '_' + str(epoch)+ '.npy')
        if hasattr(layers[idx], 'b'):
            weight_path_list.append(
                weights_dir+ 'b' + '_' + str(idx) + '_' + str(epoch)+ '.npy')
        if hasattr(layers[idx], 'b0'):
            weight_path_list.append(
                weights_dir+ 'b0' + '_' + str(idx) + '_' + str(epoch)+ '.npy')
        if hasattr(layers[idx], 'b1'):
            weight_path_list.append(
                weights_dir+ 'b1' + '_' + str(idx) + '_' + str(epoch)+ '.npy')
    
    return weight_path_list
    
def load_weights_from_memory(layers, weights_list, l_range=None):
    
    
    i_list=0
    for idx in range(len(layers)):
        
        if l_range !=None and (idx not in l_range):
            continue
            
        if hasattr(layers[idx], 'W'):
            layers[idx].W.val.set_value(weights_list[i_list])
            i_list +=1
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.val.set_value(weights_list[i_list])
            i_list +=1
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.val.set_value(weights_list[i_list])
            i_list +=1
        if hasattr(layers[idx], 'b'):
            layers[idx].b.val.set_value(weights_list[i_list])
            i_list +=1
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.val.set_value(weights_list[i_list])
            i_list +=1
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.val.set_value(weights_list[i_list])
            i_list +=1
    
    

def save_momentums(vels, weights_dir, epoch):
    
    if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            print('Creating folder: %s' % weights_dir)
            
    for ind in range(len(vels)):
        np.save(os.path.join(weights_dir, 'mom_' + str(ind) + '_' + str(epoch)),
                vels[ind].get_value())


def load_momentums(vels, weights_dir, epoch):
    for ind in range(len(vels)):
        vels[ind].set_value(np.load(os.path.join(
            weights_dir, 'mom_' + str(ind) + '_' + str(epoch) + '.npy')))
            
def check_model(model):
            
    try:
        
        assert hasattr(model, 'params') == True
        
        assert isinstance(model.params, list)
        
        import theano
        
        assert isinstance(model.params[0], theano.gpuarray.type.GpuArraySharedVariable)
        
        assert hasattr(model, 'data') == True
        
        assert hasattr(model, 'epoch') == True
        
        assert hasattr(model, 'n_epochs') == True
        
        assert hasattr(model, 'n_subb') == True # number of sub batches in a minibatch , default:1
        
        assert hasattr(model, 'compile_iter_fns') == True and callable(getattr(model, 'compile_iter_fns')) == True
        
        assert hasattr(model, 'train_iter') == True and callable(getattr(model, 'train_iter')) == True
        
        assert hasattr(model, 'val_iter') == True and callable(getattr(model, 'val_iter')) == True
        
        assert hasattr(model, 'reset_iter') == True and callable(getattr(model, 'reset_iter')) == True
        
        assert hasattr(model, 'adjust_hyperp') == True and callable(getattr(model, 'adjust_hyperp')) == True
        
        # assert hasattr(model, 'scale_lr') == True and callable(getattr(model, 'scale_lr')) == True
        
        assert hasattr(model, 'cleanup') == True and callable(getattr(model, 'cleanup')) == True
        
        
        
        
        
    
    except AssertionError:
        
        print('Model def lacks some attributes and/or methods\nattributes include: data, \n                    epoch (initialized to 0),\n                    n_epochs (max epochs),\n                    n_subb (number of sub batches in a minibatch, default to 1)\n\nmethods include: compile_iter_fns(self, *args, **kwargs), \n                 train_iter(self, *args, **kwargs) ,\n                 val_iter(self, *args, **kwargs) ,  \n                 reset_iter(self, *args, **kwargs) ,\n                 adjust_hyperp(self, *args, **kwargs) ,\n                 scale_lr(self, *args, **kwargs) ,  \n                 cleanup(self, *args, **kwargs) ,   \n')
        raise
        
def check_model_cdd(model):
    
    try:

        assert hasattr(model, 'vels') == True

        assert isinstance(model.vels, list)

        assert hasattr(model, 'vels2') == True

        assert isinstance(model.vels2, list)

    except AssertionError:
        
        import theano
        
        model.vels= [theano.shared(param_i.get_value() * 0.)
        for param_i in model.params]
        
        model.vels2= [theano.shared(param_i.get_value() * 0.)
        for param_i in model.params]


def save_model(model, path, verbose): 
    
    if hasattr(model, 'save') == True:
        model.save(path)
    
    else:

        try:
            layers = model.layers
            save_weights(layers, path, model.epoch)
        except AttributeError:
            import pickle
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path+model.name+"params_%d.pkl" % model.epoch, 'wb') as f:
                pickle.dump(model.params, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        np.save(path + 'lr_' + str(model.epoch) + \
                        '.npy', 0)
        #vels = model.vels 
        #save_momentums(vels, self.config['weights_dir'], self.epoch)

    if verbose:
        print('\nweights saved at epoch %d' % model.epoch)

    try:
        with open(path+"val_info.txt", "a") as f:
            f.write("\nepoch: {} val_info {}:".format(model.epoch, \
                                                model.current_info))
    except:
        pass
