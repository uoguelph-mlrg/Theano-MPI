
# Display results offline

import numpy as np
import sys
import yaml
    
#############
# load data #
#############
def load(path):

    with open(path, 'r') as f:
        config = yaml.load(f)
    config['rank'] = 0        

    sys.path.append('../exc/')

    from recorder import Recorder

    recorder = Recorder(config)

    recorder.load()

    return recorder
    
if __name__ == '__main__' :

    recorder = load(path = '../run/config.yaml')

    recorder.show()
