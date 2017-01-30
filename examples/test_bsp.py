'''
Example for training on 2GPUs by the rule of BSP

'''

from theanompi import BSP

rule=BSP()

# modelfile: the relative path to the model file
# modelclass: the class name of the model to be imported
rule.init(devices=['cuda0', 'cuda1'] , 
          modelfile = 'theanompi.models', 
          modelclass = 'AlexNet') 
rule.wait()


