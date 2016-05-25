
class ModelBase(object):
    # To be used in the parallel-training framework, 
    # the model should be defined in this way

    def __init__(self):
        self.name = None
        self.current_info = None
        self.img_mean = None
    def build_model(self):
    	pass
	
    def compile_train(self):
    	pass

    def compile_val(self):
    	pass
	
    def adjust_lr(self):
    	pass
	
    def load_params(self):
    	pass
	
    def set_params(self):
    	pass
	
    def get_params(self):
    	pass
        

def updates_dict(config, model):
    
    use_momentum=config['use_momentum'], 
    use_nesterov_momentum=config['use_nesterov_momentum']
    
    try:
        size = config['size']
        verbose = config['rank'] == 0
    except KeyError:
        size = 1
        verbose = True
        
    params, grads, weight_types = model.params, model.grads, model.weight_types
    
    vels, vels2 = model.vels, model.vels2
    
    
    lr = model.shared_lr #T.scalar('lr')  # symbolic learning rate
    mu = model.mu # def: 0.9 # momentum
    eta = model.eta  #0.0002 # weight decay    
    
    updates_w = []
    updates_v = []
    updates_dv = []

    if use_momentum:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i,vel_i2, weight_type in \
                zip(params, grads, vels,vels2, weight_types):

            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")

            if use_nesterov_momentum:
                vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
            else:
                vel_i_next = mu * vel_i - real_lr * real_grad

            updates_v.append((vel_i, vel_i_next))
            updates_w.append((vel_i, vel_i_next))
            updates_w.append((param_i, param_i + vel_i_next))
            updates_dv.append((param_i, param_i + vel_i2))

    else:
        for param_i, grad_i, vel_i,vel_i2, weight_type in \
                zip(params, grads, vels,vels2, weight_types):
                
            if weight_type == 'W':
                updates_v.append((vel_i,- lr * grad_i - eta * lr * param_i))
                updates_w.append((vel_i,- lr * grad_i - eta * lr * param_i))
                updates_w.append((param_i, param_i - lr * grad_i - eta * lr * param_i))

            elif weight_type == 'b':
                updates_v.append((vel_i, - 2 * lr * grad_i))
                updates_w.append((vel_i, - 2 * lr * grad_i))
                updates_w.append((param_i, param_i - 2 * lr * grad_i))
                
            else:
                raise TypeError("Weight Type Error")
                
            updates_dv.append((param_i, param_i + vel_i2))
               
    return updates_w, updates_v, updates_dv                