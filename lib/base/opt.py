def BSP_MSGD(model, use_nesterov_momentum,worker_type):
    
    params, grads, weight_types = model.params, model.grads, model.weight_types
    
    vels, vels2 = model.vels, model.vels2
    
    lr = model.lr #shared_lr #T.scalar('lr')  # symbolic learning rate
    mu = model.mu # def: 0.9 # momentum
    eta = model.eta  #0.0002 # weight decay

    updates_w = [] # for avg
    
    updates_v = [] # for cdd
    updates_dv = [] # for cdd
    
    assert len(weight_types) == len(params)
    
    k=0

    for param_i, grad_i, weight_type in \
            zip(params, grads, weight_types):

        if weight_type == 'W':
            real_grad = grad_i + eta * param_i
            real_lr = lr
        elif weight_type == 'b':
            real_grad = grad_i
            real_lr = 2. * lr
        else:
            raise TypeError("Weight Type Error")

        if use_nesterov_momentum:
            vel_i_next = mu ** 2 * vels[k] - (1 + mu) * real_lr * real_grad
        else:
            vel_i_next = mu * vels[k] - real_lr * real_grad
            
        if worker_type == 'cdd':

            updates_v.append((vels[k], vel_i_next))
            updates_dv.append((param_i, param_i + vels2[k]))
            
        elif worker_type == 'avg':
            
            updates_w.append((vels[k], vel_i_next))
            updates_w.append((param_i, param_i + vel_i_next))
            
        k=k+1
        
    return updates_w, updates_v, updates_dv
    

def BSP_SGD(model,worker_type):
    
    params, grads, weight_types = model.params, model.grads, model.weight_types
    
    vels, vels2 = model.vels, model.vels2
    
    lr = model.lr #shared_lr #T.scalar('lr')  # symbolic learning rate
    mu = model.mu # def: 0.9 # momentum
    eta = model.eta  #0.0002 # weight decay

    updates_w = [] # for avg
    
    updates_v = [] # for cdd
    updates_dv = [] # for cdd
    
    assert len(weight_types) == len(params)
    
    
    k=0
    
    for param_i, grad_i, weight_type in \
            zip(params, grads, weight_types):
            
    
        if weight_type == 'W':
            
            if worker_type == 'cdd':
                
                update =          - lr * grad_i - eta * lr * param_i
                
            elif worker_type == 'avg':
                
                update =  param_i - lr * grad_i - eta * lr * param_i

        elif weight_type == 'b':
            
            if worker_type == 'cdd':
            
                update =         - 2 * lr * grad_i
                
            elif worker_type == 'avg':
                
                update = param_i - 2 * lr * grad_i
                
        if worker_type == 'cdd':
            
            updates_v.append((vels[k], update))
            updates_dv.append((param_i, param_i + vels2[k]))
            
        elif worker_type == 'avg':
            
            # updates_w.append((vel_i, - 2 * lr * grad_i))
            updates_w.append((param_i, update))
            
            
        k=k+1
        
    return updates_w, updates_v, updates_dv