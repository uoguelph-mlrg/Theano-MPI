
from theanompi import BSP

rule=BSP(None)

devices= ['cuda0']

rule.init(devices)
rule.wait()


