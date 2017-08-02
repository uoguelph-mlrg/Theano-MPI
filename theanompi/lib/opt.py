
def pre_model_iter_fn(model, k=1, f_train=True, f_val=True):
    
    # to make sure model compiles necessary functions (get_vels() and descent() for cdd, or train() for avg) and allocate necessary extra param memory (vels,vels2 for cdd, or nothing for avg)
        
    # allocate supporting params for this worker type
    if f_train:
            
        updates_v, updates_dv = prepare_update_dict(model,  k=k)
        
        updates_v=fix_update_bcasts(dict(updates_v))
        updates_dv=fix_update_bcasts(dict(updates_dv))

        get_vel_args = {"inputs":[model.subb_ind], "outputs":[model.cost,model.error], 
                                                   "updates":updates_v, 
                                                    "givens":[(model.x,  model.shared_x_slice), 
                                                              (model.y,  model.shared_y_slice),
                                                              (model.lr, model.shared_lr)]}
                                                         
        descent_vel_args = {"inputs":[], "outputs":[], "updates":updates_dv,
                                                        "givens":[(model.lr, model.shared_lr)]}
                                            
        model.compile_train(get_vel_args, descent_vel_args) # needs compile model before para_load_init() # 2 (local to worker type)

        model.get_vel, model.descent_vel = model.compiled_train_fn_list
        
        
        model.train_iter_fn = choose_iter_fn(model)
    
    if f_val:    
        
        model.compile_val()
    
        model.val_iter_fn = model.val_fn
        
def fix_update_bcasts(updates):
    import theano.tensor as T
    for param, update in updates.items():
        if param.broadcastable != update.broadcastable:
            updates[param] = T.patternbroadcast(update, param.broadcastable)
    return updates

def choose_iter_fn(model):
    
    # TODO maybe not be correct to perform step3 step1 -> step2
    
    def cdd_iter_fn(subb_ind):
        model.descent_vel()
        cost, error = model.get_vel(subb_ind)
        return cost, error
    
    return cdd_iter_fn

def prepare_update_dict(model, k=1):

    if model.use_momentum:
        
        updates_v, updates_dv = BSP_MSGD(model, model.use_nesterov_momentum, k=k)
            
    else:
        
        updates_v, updates_dv = BSP_SGD(model, k=k)
            
    
    return updates_v, updates_dv
    
def _clip_paramlist(param_list, scale=10):
    
    import theano.tensor as T
    
    res=[]
    for param in param_list:
        res.append(T.clip(param, -scale, scale)) # clip the param to be less than 10
    
    return res


def _BSP_MSGD(model, use_nesterov_momentum, k=1):
    
    '''
    aggregate gradient
    
    '''
    
    params, grads, weight_types = model.params, model.grads, model.weight_types
        
    import theano
    
    model.vels=[]
    model.vels2=[]
    
    lr = model.lr #shared_lr #T.scalar('lr')  # symbolic learning rate
    mu = model.mu # def: 0.9 # momentum
    eta = model.eta  #0.0002 # weight decay
    
    updates_pre_g_aggre = [] # pre gradient aggregation
    updates_post_g_aggre = [] # post gradient aggregation
    
    for ind, (param_i, grad_i, weight_type) in enumerate(
                        zip(params, grads, weight_types)):
                        
        
        if param_i.name in ['gamma', 'beta']: # explicitly not exchanging BN parameters, directly updating
            
            tmp0=theano.shared(param_i.get_value() * 0.)
            
            if param_i.name == 'gamma':
                real_grad = grad_i  # no weight decay for BN parameters
                real_lr = lr
            elif param_i.name == 'beta':
                real_grad = grad_i 
                real_lr = 2. * lr
                
            if use_nesterov_momentum:
                # vel_i_next = mu ** 2 * vels[ind] - (1 + mu) * real_lr * real_grad  #  =real_lr*u_i_next
                u_i_next = mu ** 2 * tmp0 - (1+mu) * real_grad
            else:
                # vel_i_next = mu * vels[ind] + real_lr * real_grad # =real_lr*u_i_next
                u_i_next = mu * tmp0 + real_grad
            
            updates_pre_g_aggre.append((tmp0, u_i_next))
            updates_pre_g_aggre.append((param_i, param_i - real_lr*u_i_next)) # step 3: update local param with vels2
            
        else:
            
            tmp1=theano.shared(param_i.get_value() * 0.)
            
            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")
            
            if k==1:
                
                if use_nesterov_momentum:
                    # vel_i_next = mu ** 2 * vels[ind] - (1 + mu) * real_lr * real_grad  #  =real_lr*u_i_next
                    u_i_next = mu ** 2 * tmp1 - (1+mu) * real_grad
                else:
                    # vel_i_next = mu * vels[ind] + real_lr * real_grad # =real_lr*u_i_next
                    u_i_next = mu * tmp1 + real_grad
                
            
                updates_pre_g_aggre.append((tmp1, u_i_next))
                updates_pre_g_aggre.append((param_i, param_i - real_lr*u_i_next))
                
            else:
                
                tmp2=theano.shared(param_i.get_value() * 0.)
                
                updates_pre_g_aggre.append((tmp1, real_grad/float(k))) # step 1: process per-worker gradient into tmp1
                
                # step 2 (during exchanging): allreduce per-worker gradients from tmp1 into tmp2
                
                tmp3=theano.shared(param_i.get_value() * 0.) # allocate tmp3 for storing momentum history
                if use_nesterov_momentum:
                    # vel_i_next = mu ** 2 * vels[ind] - (1 + mu) * real_lr * real_grad  #  =real_lr*u_i_next
                    u_i_next = mu ** 2 * tmp3 - (1+mu) * tmp2
                else:
                    # vel_i_next = mu * vels[ind] + real_lr * real_grad # =real_lr*u_i_next
                    u_i_next = mu * tmp3 + tmp2

                updates_post_g_aggre.append((tmp3, u_i_next))
                 
                
                updates_post_g_aggre.append((param_i, param_i - real_lr*u_i_next)) # step 3: update local param with tmp2 and tmp3
                
                model.vels.append(tmp1)  # tmp1 -> tmp2
                model.vels2.append(tmp2)
                
        # in practice BSP:
        # training (step3-> step1) - > comm (step 2)
        
    return updates_pre_g_aggre, updates_post_g_aggre
    
    
    
def BSP_MSGD(model, use_nesterov_momentum, k=1):
    
    '''
    
    aggregate momentum instead of gradient
    
    '''
    
    params, grads, weight_types = model.params, model.grads, model.weight_types
        
    import theano
    
    model.vels=[]
    model.vels2=[]
    
    lr = model.lr #shared_lr #T.scalar('lr')  # symbolic learning rate
    eta = model.eta  #0.0002 # weight decay
    
    updates_pre_g_aggre = [] # pre gradient aggregation
    updates_post_g_aggre = [] # post gradient aggregation
    
    for ind, (param_i, grad_i, weight_type) in enumerate(
                        zip(params, grads, weight_types)):
                        
        
        if param_i.name in ['gamma', 'beta']: # explicitly not exchanging BN parameters, directly updating
            
            tmp0=theano.shared(param_i.get_value() * 0.)
            
            if param_i.name == 'gamma':
                real_grad = grad_i  # no weight decay for BN parameters
                real_lr = lr
            elif param_i.name == 'beta':
                real_grad = grad_i 
                real_lr = 2. * lr
                
            if use_nesterov_momentum:
                # vel_i_next = mu ** 2 * vels[ind] - (1 + mu) * real_lr * real_grad  #  =real_lr*u_i_next
                u_i_next = mu ** 2 * tmp0 - (1+mu) * real_grad
            else:
                # vel_i_next = mu * vels[ind] + real_lr * real_grad # =real_lr*u_i_next
                u_i_next = mu * tmp0 + real_grad
            
            updates_pre_g_aggre.append((tmp0, u_i_next))
            updates_pre_g_aggre.append((param_i, param_i - real_lr*u_i_next)) # step 3: update local param with vels2
            
        else:
            
            tmp1=theano.shared(param_i.get_value() * 0.)
            
            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")
            
            
            if use_nesterov_momentum:
                # vel_i_next = mu ** 2 * vels[ind] - (1 + mu) * real_lr * real_grad  #  =real_lr*u_i_next
                u_i_next = mu ** 2 * tmp1 - (1+mu) * real_grad
            else:
                # vel_i_next = mu * vels[ind] + real_lr * real_grad # =real_lr*u_i_next
                u_i_next = mu * tmp1 + real_grad
                
            if k==1:
                updates_pre_g_aggre.append((tmp1, u_i_next))
                updates_pre_g_aggre.append((param_i, param_i - real_lr*u_i_next))
                
            else:
                
                
                tmp2=theano.shared(param_i.get_value() * 0.)

                updates_pre_g_aggre.append((tmp1, u_i_next)) # step 1: process per-worker gradient into tmp1
                # step 2 (during exchanging): allreduce per-worker gradients from tmp1 into tmp2
                updates_post_g_aggre.append((param_i, param_i - real_lr*tmp2/float(k))) # step 3: update local param with tmp2
                
                model.vels.append(tmp1) # tmp1 -> tmp2
                model.vels2.append(tmp2)
                
        # in practice BSP:
        # training (step3-> step1) - > comm (step 2)
        
    return updates_pre_g_aggre, updates_post_g_aggre
    
        
def BSP_SGD(model, k=1):
    
    params, grads, weight_types = model.params, model.grads, model.weight_types
        
    import theano
    
    model.vels=[]
    model.vels2=[]
    
    lr = model.lr #shared_lr #T.scalar('lr')  # symbolic learning rate
    eta = model.eta  #0.0002 # weight decay
    
    updates_pre_g_aggre = [] # pre gradient aggregation
    updates_post_g_aggre = [] # post gradient aggregation
    
    for ind, (param_i, grad_i, weight_type) in enumerate(
                        zip(params, grads, weight_types)):
        
        if param_i.name in ['gamma', 'beta']: # explicitly not exchanging BN parameters
        
            if param_i.name == 'gamma':
                real_grad = grad_i  # no weight decay for BN parameters
                real_lr = lr
            elif param_i.name == 'beta':
                real_grad = grad_i 
                real_lr = 2. * lr
            
            updates_pre_g_aggre.append((param_i, param_i - real_lr * real_grad)) # step 3: update local param with vels2
            
        else:
            
            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")

            if k==1:
                
                updates_pre_g_aggre.append((param_i, param_i - real_lr * real_grad))
                
            else:
                
                tmp1=theano.shared(param_i.get_value() * 0.)
                tmp2=theano.shared(param_i.get_value() * 0.)
                
                updates_pre_g_aggre.append((tmp1, real_lr * real_grad/float(k))) # step 1: process per-worker gradient into tmp1
                # step 2 (during exchanging): allreduce per-worker gradients from tmp1 into tmp2
                updates_post_g_aggre.append((param_i, param_i - tmp2)) # step 3: update local param with tmp2
            
                model.vels.append(tmp1) # tmp1 -> tmp2
                model.vels2.append(tmp2)
            
        # in practice BSP:
        # training (step3-> step1) - > comm (step 2)
        
    return updates_pre_g_aggre, updates_post_g_aggre