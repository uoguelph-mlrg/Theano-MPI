
# Display results offline

import numpy as np
import sys
import yaml
    
#############
# load data #
#############
loadpath_base = 'inforec_test/'
import glob
loadpaths = sorted(glob.glob(loadpath_base+'*.pkl'))
labels = ['_'+l.split('/',1)[-1].split('_',1)[-1].split('.',1)[0] for l in loadpaths]
print loadpaths
print labels
    
    
def load(path,loadpath):

    with open(path, 'r') as f:
        config = yaml.load(f)
    config['verbose'] = False        

    sys.path.append('../lib/')

    from base.recorder import Recorder

    recorder = Recorder(config)

    recorder.load(filepath = loadpath)

    return recorder
    
if __name__ == '__main__' :
    
    recorders = []
    
    for i in range(len(loadpaths)):
        
        recorders.append(load(path = '../run/config.yaml',loadpath = loadpaths[i]))
        
        if i != len(loadpaths)-1: 
            recorders[i].show(label=labels[i], color_id = i, show=False)
        else:
            recorders[i].show(label=labels[i], color_id = i, show=True)
        
    
