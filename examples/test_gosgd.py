'''
Example for training on 2 worker GPUs by the rule of Gossip SGD
'''
from theanompi import GOSGD

rule=GOSGD()

# modelfile: the relative path to the model file
# modelclass: the class name of the model to be imported
rule.init(devices=['cuda1', 'cuda2'] , # cuda0: server, cuda1 and cuda2: workers
          modelfile = 'theanompi.models', 
          modelclass = 'Cifar10_model') 
rule.wait()