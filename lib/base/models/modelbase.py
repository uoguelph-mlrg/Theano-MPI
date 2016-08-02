
class ModelBase(object):
    # To be used in the parallel-training framework, 
    # the model should be defined in this way

    def __init__(self):
        self.name = None
        self.current_info = None
        self.img_mean = None
        
        # shared variable for storing momentum before exchanging momentum(delta w)
        self.vels = None
        
        # shared variable for accepting momentum during exchanging momentum(delta w)
        self.vels2 = None
        
        self.updates_dict = [] # prepare for different worker type and exchanging strategy
        self.compile_train_fn_list = []
        self.compiled_train_fn_list = []
        
        # all kinds of possible functions in Theano-MPI, can be extended if needed
        self.train = None
        self.train_vel_acc = None # training with accumulated vel stored
        self.get_vel = None
        self.descent_vel = None
        self.val = None
        self.inference = None
        
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