# what is the iteration process (defined in model itself) and what is the rule based on which to exchange (defined in theanompi): like how to exchange, when to excange

from theanompi import BSP

rule=BSP()

# modelfile: the relative path to the model file
# modelclass: the class name of the model to be imported
rule.init(devices=['cuda0', 'cuda1'] , modelfile = 'models.cifar10', modelclass = 'Cifar10_model') 
rule.wait()


