# Display results offline

import numpy as np
import sys
import yaml
    
#############
# load data #
#############
loadpath_base = 'inforec/'
import glob
loadpaths = sorted(glob.glob(loadpath_base+'*.pkl'))

labels = ['_'+l.split('/',1)[-1].split('_',1)[-1].split('.',1)[0] for l in loadpaths]
print loadpaths
print labels
    
    
def load(loadpath, config):

    from theanompi.lib.recorder import Recorder

    recorder = Recorder(**config)

    recorder.load(loadpath)

    return recorder
    
if __name__ == '__main__' :
    
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    
    recorders = []
    
    for i in range(len(loadpaths)):
        
        config={}
        config['comm']=comm
        config['printFreq']=40, 
        config['modelname']='ResNet50'
        config['verbose']=True
        
        recorders.append(load(loadpaths[i], config))
        
        if i != len(loadpaths)-1: 
            recorders[i].show(label=labels[i], color_id = i, show=False)
        else:
            recorders[i].show(label=labels[i], color_id = i, show=True)
        
    
