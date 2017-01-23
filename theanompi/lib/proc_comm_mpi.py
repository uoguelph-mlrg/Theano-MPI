'''parallel communication process

1. for prcoessing sending and receiving request from training process,

sending request:
fetch training params using IPC
receiving request:
setting training params using IPC
 
2. for P2P transfer the params between two param_comm's

send the params of training process to other para_comm 
recieve the params from other para_comm 

TODO Jan23
'''

