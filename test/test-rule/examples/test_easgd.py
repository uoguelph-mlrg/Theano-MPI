# what is the iteration process (defined in model itself) and what is the rule based on which to exchange (defined in theanompi): like how to exchange, when to excange

from theanompi import EASGD

rule=EASGD()

# modelfile: the relative path to the model file
# modelclass: the class name of the model to be imported
rule.init(devices=['cuda1', 'cuda2'] , 
          modelfile = 'theanompi.models', 
          modelclass = 'Cifar10_model') 
rule.wait()


