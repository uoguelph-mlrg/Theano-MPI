
# Display results offline

import numpy as np
import sys
import yaml

loadpath_base = '../run/inforec/'
import glob
loadpaths = sorted(glob.glob(loadpath_base+'inforec.pkl'))

#############
# load data #
#############
def load(path):

    with open(path, 'r') as f:
        config = yaml.load(f)
    #config['rank'] = 0 
    config['verbose'] = False    

    sys.path.append('../lib/')

    from base.recorder import Recorder

    recorder = Recorder(config)

    recorder.load(loadpaths[0])

    return recorder
    
if __name__ == '__main__' :

    recorder = load(path = '../run/config.yaml')

    recorder.show()
