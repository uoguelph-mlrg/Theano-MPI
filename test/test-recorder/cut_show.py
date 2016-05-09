import numpy as np
import sys
import yaml
    
#############
# load data #
#############
loadpath_base = './'
import glob
loadpaths = sorted(glob.glob(loadpath_base+'*.pkl'))
labels = ['_'+l.split('/',1)[-1].split('_',1)[-1].split('.',1)[0] for l in loadpaths]
print loadpaths
print labels
    
    
def load(path,loadpath):

    with open(path, 'r') as f:
        config = yaml.load(f)
    config['verbose'] = False        

    sys.path.append('../../lib/')

    from base.recorder import Recorder

    recorder = Recorder(config)

    recorder.load(filepath = loadpath)

    return recorder
    
if __name__ == '__main__' :
    
    recorders = []
    
    old = load(path = '../../run/config.yaml',loadpath = loadpaths[0])
    old.show(label='old', color_id = 0, show=False)
    new = load(path = '../../run/config.yaml',loadpath = loadpaths[0]).cut(load_epoch=20)
    new.show(label='new', color_id = 1, show=True)
        