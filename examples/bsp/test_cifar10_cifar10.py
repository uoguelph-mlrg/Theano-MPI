'''
Example for training on 3 worker GPUs by the rule of BSP
'''
from theanompi import BSP

rule=BSP()

# modelfile: the relative path to the model file
# modelclass: the class name of the model to be imported
rule.init(devices=['cuda0', 'cuda1', 'cuda2'] ,
          modelfile = 'theanompi.models.cifar10', 
          modelclass = 'Cifar10_model') 
rule.wait()
