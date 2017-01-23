'''
Example for training Cifar10_model on 2 worker GPUs by the rule of EASGD
'''
from theanompi import EASGD

rule=EASGD()

# modelfile: the relative path to the model file
# modelclass: the class name of the model to be imported
rule.init(devices=['cuda0', 'cuda1', 'cuda2'] , # cuda0: server, cuda1 and cuda2: workers
          modelfile = 'theanompi.models', 
          modelclass = 'AlexNet') 
rule.wait()


