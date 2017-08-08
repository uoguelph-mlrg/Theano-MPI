
# Display results offline

import numpy as np
import sys
import yaml

loadpath_base = 'inforec/'
import glob
loadpaths = sorted(glob.glob(loadpath_base+'/inforec.pkl'))
    
if __name__ == '__main__' :

    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    config={}
    config['comm']=comm
    config['printFreq']=40, 
    config['modelname']='ResNet50'
    config['verbose']=True   

    from theanompi.lib.recorder import Recorder

    recorder = Recorder(**config)

    recorder.load(loadpaths[0])


    recorder.show()
