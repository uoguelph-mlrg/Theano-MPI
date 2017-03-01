
def pre_model_iter_fn(model, sync_type, f_train=True, f_val=True):
    
    # to make sure model compiles necessary functions (get_vels() and descent() for cdd, or train() for avg) and allocate necessary extra param memory (vels,vels2 for cdd, or nothing for avg)
        
    # allocate supporting params for this worker type
    if f_train:
        
        if sync_type == 'cdd':
    
            import theano
    
            model.vels = [theano.shared(param_i.get_value() * 0.)
                for param_i in model.params]
    
            model.vels2 = [theano.shared(param_i.get_value() * 0.)
                        for param_i in model.params]
                
            updates_v, updates_dv = prepare_update_dict(model, sync_type='cdd')
    
            get_vel_args = {"inputs":[model.subb_ind], "outputs":[model.cost,model.error], "updates":updates_v, \
                                                           "givens":[(model.x,  model.shared_x_slice), 
                                                                     (model.y,  model.shared_y_slice),
                                                                     (model.lr, model.shared_lr)]}
                                                             
            descent_vel_args = {"inputs":[], "outputs":[], "updates":updates_dv}
                                                
            model.compile_train(get_vel_args, descent_vel_args) # needs compile model before para_load_init() # 2 (local to worker type)
    
            model.get_vel, model.descent_vel = model.compiled_train_fn_list
        
        
        else: # avg or other sync types
    
            import theano
    
            model.vels = [theano.shared(param_i.get_value() * 0.)
                for param_i in model.params]
            
            model.vels2 = [theano.shared(param_i.get_value() * 0.)
                        for param_i in model.params]
    
            updates_w, = prepare_update_dict(model, sync_type='avg')
    
            train_args = {"inputs":[model.subb_ind], "outputs": [model.cost,model.error], "updates": updates_w, \
                                                                      "givens": [(model.x,  model.shared_x_slice), 
                                                                                 (model.y,  model.shared_y_slice),
                                                                                 (model.lr, model.shared_lr)]}
    
            model.compile_train(train_args)
    
            model.train_fn , = model.compiled_train_fn_list
        
        
        model.train_iter_fn = choose_iter_fn(model, sync_type)
    
    if f_val:    
        
        model.compile_val()
    
        model.val_iter_fn = model.val_fn

def choose_iter_fn(model, sync_type):
    
    if sync_type == 'cdd':
            
        def cdd_iter_fn(subb_ind):
            model.descent_vel()
            cost, error = model.get_vel(subb_ind)
            return cost, error
        
        return cdd_iter_fn
                                
    elif sync_type == 'avg':
        
        return model.train_fn

def prepare_update_dict(model, sync_type, clip=True):
    

    if model.use_momentum:
        
        updates_w, updates_v, updates_dv = BSP_MSGD(model, model.use_nesterov_momentum,sync_type, clip)
            
    else:
        
        updates_w, updates_v, updates_dv = BSP_SGD(model, sync_type, clip)
            
    if sync_type == 'cdd':
    
        update_dict = [updates_v, updates_dv]
    
    elif sync_type == 'avg':
        
        update_dict = [updates_w]
        
        
    return update_dict
    
def _clip_paramlist(param_list, scale=10):
    
    import theano.tensor as T
    
    res=[]
    for param in param_list:
        res.append(T.clip(param, -scale, scale)) # clip the param to be less than 10
    
    return res
           
def BSP_MSGD(model, use_nesterov_momentum,sync_type, clip):
    
    params, grads, weight_types = model.params, model.grads, model.weight_types
    
    if clip==True:
        grads=_clip_paramlist(grads)
    
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
            
        if sync_type == 'cdd':

            updates_v.append((vels[k], vel_i_next))
            updates_dv.append((param_i, param_i + vels2[k]))
            
        elif sync_type == 'avg':
            
            updates_w.append((vels[k], vel_i_next))
            updates_w.append((param_i, param_i + vel_i_next))
            
        k=k+1
        
    return updates_w, updates_v, updates_dv
    

def BSP_SGD(model,sync_type, clip):
    
    params, grads, weight_types = model.params, model.grads, model.weight_types
    
    if clip==True:
        grads=_clip_paramlist(grads)
        
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
            
            if sync_type == 'cdd':
                
                update =          - lr * grad_i - eta * lr * param_i
                
            elif sync_type == 'avg':
                
                update =  param_i - lr * grad_i - eta * lr * param_i

        elif weight_type == 'b':
            
            if sync_type == 'cdd':
            
                update =         - 2 * lr * grad_i
                
            elif sync_type == 'avg':
                
                update = param_i - 2 * lr * grad_i
                
        if sync_type == 'cdd':
            
            updates_v.append((vels[k], update))
            updates_dv.append((param_i, param_i + vels2[k]))
            
        elif sync_type == 'avg':
            
            # updates_w.append((vel_i, - 2 * lr * grad_i))
            updates_w.append((param_i, update))
            
            
        k=k+1
        
    return updates_w, updates_v, updates_dv
    
    
def MSGD(model, use_nesterov_momentum,sync_type, clip):
    
    params, grads, weight_types = model.params, model.grads, model.weight_types
    
    if clip==True:
        grads=_clip_paramlist(grads)
        
    vels, vels2 = model.vels, model.vels2
    
    lr = model.lr #shared_lr #T.scalar('lr')  # symbolic learning rate
    mu = model.mu # def: 0.9 # momentum
    eta = model.eta  #0.0002 # weight decay

    updates_w = [] # for avg
    
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
        
        updates_w.append((vels[k], vel_i_next))    
        updates_w.append((param_i, param_i + vel_i_next))
        updates_w.append((vels2[k], vels2[k] + vel_i_next))
            
        k=k+1
        
    return updates_w
    
def SGD(model,sync_type, clip):
    
    params, grads, weight_types = model.params, model.grads, model.weight_types
    
    if clip==True:
        grads=_clip_paramlist(grads)
        
    vels, vels2 = model.vels, model.vels2
    
    lr = model.lr #shared_lr #T.scalar('lr')  # symbolic learning rate
    mu = model.mu # def: 0.9 # momentum
    eta = model.eta  #0.0002 # weight decay

    updates_w = [] # for avg
    
    assert len(weight_types) == len(params)
    
    k=0
    
    for param_i, grad_i, weight_type in \
            zip(params, grads, weight_types):
            
        if weight_type == 'W':
                
            update =  param_i - lr * grad_i - eta * lr * param_i
            update_vel2 =  - lr * grad_i - eta * lr * param_i

        elif weight_type == 'b':
                
            update = param_i - 2 * lr * grad_i
            update_vel2 = - 2 * lr * grad_i

        updates_w.append((param_i, update))
        updates_w.append((vels2[k], update_vel2))
            
            
        k=k+1
        
    return updates_w
    
