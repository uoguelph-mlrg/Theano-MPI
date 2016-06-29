import numpy as np

import sys

sys.path.append("../")

from BSP_Worker import BSP_PTWorker

class Val_Worker(BSP_PTWorker):
    '''
    The Validation Worker class based on BSP_PTWorker
    For parallel validation and interpolation
    needs deterministic calculation: set lib_conv = cudaconvnet in config.yaml
    '''
    
    def __init__(self, port, config, device):
        BSP_PTWorker.__init__(self, port = port, \
                                config = config, \
                                device = device)
        
    
    
    def load_snapshot_from_file(self, load_path, load_epoch):
        
        from base.helper_funcs import collect_weight_path
        
        name_list = collect_weight_path(self.model.layers, load_path, load_epoch)
        
        snapshot_list = []
        
        
        
        for name in name_list:
            
            snapshot = np.load(name)
            
            snapshot_list.append(snapshot)
        
        return snapshot_list
        
    def load_model_from_snapshot(self, snapshot_list):
        
        from base.helper_funcs import load_weights_from_memory
        
        load_weights_from_memory(self.model.layers, snapshot_list)
        
    def merge_snapshot(self,snapshot_list1,snapshot_list2, merging_ratio):
        
        merged_s_list = []
        
        for s1,s2 in zip(snapshot_list1,snapshot_list2):
            merged_s = s1 * (1.0-merging_ratio) + s2 * merging_ratio
            merged_s_list.append(merged_s)
        
        return merged_s_list
        
    def distance_traveled(self,snapshot_list1,snapshot_list2):
        
        pass
        
        # distance_list = []
        #
        # for s1,s2 in zip(snapshot_list1,snapshot_list2):
        #
        #     print len(s1.shape)
            
           #distance_list.append(np.linalg.norm(s1 - s2, len(s1.shape)))
            
        #print distance_list
            
                                
    def run(self):
        
        # override BSP_PTWorker class method
        
        load_path = '/work/mahe6562/models-alexnet-1gpu-128b-cop4-6-18/'
    
        for load_epoch in [0,10,20,30,40,50,60,62]:
    
            snapshot_list = worker.load_snapshot_from_file(load_path, load_epoch)
    
            worker.load_model_from_snapshot(snapshot_list)
                
            self.comm.Barrier()

            self.val()
            
            if self.verbose:
            
                with open(load_path+"val_info_re.txt", "a") as f:
                    f.write("\nepoch: {} val_info {}:".format(load_epoch, \
                                                            self.model.current_info))
            
                # self.recorder.save(self.count, self.model.shared_lr.get_value(), \
                #                 filepath = self.config['record_dir'] + 'inforec.pkl')
                
        
            
    def run_interpolation(self):
        
        '''interpolation for Daniel'''
        
        path_1GPU = '/work/mahe6562/models-alexnet-1gpu-128b-cop4-6-18/'
        path_8GPU_32b_BSP = '/work/mahe6562/models-alexnet-8gpu-32b-cop2-4-4/'
        path_8GPU_128b_BSP = '/work/mahe6562/models-alexnet-8gpu-128b-cop4-6-10/'
        path_8GPU_128b_EASGD = '/work/mahe6562/models-alexnet-9gpu-128b-cop2-6-13/'
        
        testing_pairs = [
                            [[path_1GPU,62],[path_8GPU_128b_BSP,62]],
                            [[path_1GPU,62],[path_8GPU_32b_BSP,70]],
                            [[path_1GPU,62],[path_8GPU_128b_EASGD,62]],
                            ]
                            
        testing_pair_labels = ['1GPUe62_8GPU128bBSPe62',
                            '1GPUe62_8GPU32bBSPe62',
                               '1GPUe62_8GPU128bEASGDe62'
                            ]
        
        
        for pair,label in zip(testing_pairs,testing_pair_labels):
            
            if self.verbose:
                print 'testing pair'+label
            
            snapshot_list1 = worker.load_snapshot_from_file(pair[0][0],pair[0][1])
            snapshot_list2 = worker.load_snapshot_from_file(pair[1][0],pair[1][1])
            
            alpha_list = np.arange(0,1,0.05).tolist()
            #alpha_list = [0,  0.5, 1.0]
        
            for alpha in alpha_list:
                
                self.comm.Barrier()
                if self.verbose: print 'alpha=%.2f' % alpha
        
                snapshot_list_merged =self.merge_snapshot(snapshot_list1,snapshot_list2, merging_ratio=alpha)
        
                worker.load_model_from_snapshot(snapshot_list_merged)
        
                self.comm.Barrier()

                self.val()
            
            self.recorder.save(self.count, self.model.shared_lr.get_value(), \
                    filepath = self.config['record_dir'] + 'inforec_'+label+'.pkl')
                
                
    def run_distance(self):
        
        pass
            
            


if __name__ == '__main__':
    
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        
    import sys
    try:
        device = sys.argv[1]
    except IndexError:
        # raise ValueError('Need to specify a GPU device')
        import os
        gpuid = str(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        device = 'gpu'+gpuid
        
    worker = Val_Worker(port=5555, config=config, device=device)

    #worker.run()
    worker.run_interpolation()