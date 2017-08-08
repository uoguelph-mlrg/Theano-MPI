'''
Example for training on 3 worker GPUs by the rule of BSP
'''
from theanompi import BSP

BSP.sync_type='avg'

rule=BSP()

# modelfile: the relative path to the model file
# modelclass: the class name of the model to be imported
rule.init(devices=['cuda0', 'cuda1', 'cuda2'] ,
          modelfile = 'theanompi.models.keras_model_zoo.wresnet', 
          modelclass = 'Wide_ResNet') 
rule.wait()
