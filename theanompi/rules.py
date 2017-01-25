from __future__ import absolute_import
import shlex
import sys
import os
import subprocess
import signal

START_INFO = "Theano-MPI started %d workers for \n 1.updating %s params through iterations and\n 2.exchange the params with %s\nSee output log."

class Rule(object):
    
    '''
    base launcher of various synchronization rules
    
    implementation idea from platoon:
    https://github.com/mila-udem/platoon/blob/master/scripts/platoon-launcher
 
    '''
    
    def __init__(self):
        
        self.pid = None
        
        self.rulename = 'None'
        
    def init(self):
        
        '''
        start MPI processes by calling mpirun and get the pid
        '''
        pass
        
    def wait(self):
        
        if self.pid == None:
            
            'Rule %s not initialized' % self.rulename
            
            return
        
        try:
            pid, status = os.waitpid(self.pid, 0)
            if pid != self.pid:
                print("\nWARNING! Received status for unknown process {}".format(pid))
                sys.exit(3)
            if os.WIFEXITED(status):
                rcode = os.WEXITSTATUS(status)
                print("\n Rule session {0} terminated with return code: {1}.".format(pid,rcode))
            
        except (RuntimeError, KeyboardInterrupt):

            print("Killing worker processes...")

            os.kill(self.pid, signal.SIGTERM)
            pid, status = os.waitpid(self.pid, 0)
            
            sys.exit(3)
        
        

class BSP(Rule):
    
    '''Bulk Synchronous Parallel
    
    When to exchange: workers run iterations synchronously and exchange after each iteration
    
    '''
    
    sync_type = 'avg' # 'avg' or 'cdd'
    exch_strategy = 'nccl32' # nccl16 or nccl32
    
    def __init__(self):
        Rule.__init__(self)
        
        self.rulename = 'BSP(%s)' % BSP.sync_type
        
    def init(self, devices, modelfile, modelclass):
        
        N_WORKERS = len(devices)
        
        env = dict(os.environ)

        command = ["mpirun"]
        
        for index, device in enumerate(devices):
            
            # command += ["--output-filename", "%s" % 'out']
            command += ["--mca", "mpi_warn_on_fork", "0"]
            command += ["--mca", "btl_smcuda_use_cuda_ipc", "1"]
            command += ["--mca", "mpi_common_cuda_cumemcpy_async", "1"]
            command += ["--mca", "mpi_max_info_val", "10240"]
            #command += ["-np", str(len(hosts))]
            #command += ["-H", ','.join(hosts)]
            #command += ["--map-by", "ppr:4:node"]
            command += shlex.split("-x " + " -x ".join(env.keys()))
            command += ["-n", "%d" % 1]
            command += ["--bind-to", "none"]
            # command += ["--report-bindings"]
 
            worker_file_dir = os.path.dirname(os.path.realpath(__file__))
            command += [sys.executable, "-u", worker_file_dir+"/worker.py"] 
        
            command += [device, BSP.sync_type, BSP.exch_strategy, modelfile,  modelclass]
            
            if index!= N_WORKERS-1:
                command += [":"]
                
        p = subprocess.Popen(command)
        
        print(START_INFO % ( N_WORKERS, modelclass, self.rulename))
        
        self.pid=p.pid


class EASGD(Rule):
    
    '''Elastic Averaging SGD
    
    When to exchange: workers run iterations asynchronously and exchange only with the server
    
    See: 
        https://arxiv.org/abs/1412.6651
    '''
    def __init__(self):
        Rule.__init__(self)
        
        self.rulename = 'EASGD'
    
    def init(self, devices, modelfile, modelclass):
        
        N_WORKERS = len(devices) - 1
        
        env = dict(os.environ)

        command = ["mpirun"]
        
        try:
            os.remove("./ompi-server.txt")
        except OSError:
            pass
        
        for index, device in enumerate(devices):
            
            # command += ["--output-filename", "%s" % 'out']
            command += ["--mca", "mpi_warn_on_fork", "0"]
            command += ["--mca", "btl_smcuda_use_cuda_ipc", "1"]
            command += ["--mca", "mpi_common_cuda_cumemcpy_async", "1"]
            command += shlex.split("-x " + " -x ".join(env.keys()))
            #command += ["-np", str(len(hosts))]
            #command += ["-H", ','.join(hosts)]
            #command += ["--map-by", "ppr:4:node"]
            # if index==0:  # the sock server
            #     command += ["--report-uri", "./ompi-server.txt"]
            # else: # the sock workers
            #     command += ["--ompi-server","file:./ompi-server.txt"]
            
            command += ["-n", "%d" % 1]
            # command += ["--bind-to", "none"]
            # command += ["--report-bindings"]
 
            worker_file_dir = os.path.dirname(os.path.realpath(__file__))
            if index==0: # the server
                command += [sys.executable, "-u", worker_file_dir+"/easgd_server.py"] 
            else:
                command += [sys.executable, "-u", worker_file_dir+"/easgd_worker.py"] 
                
            command += [device, modelfile,  modelclass]
            
            if index!= N_WORKERS:
                command += [":"]
                
        p = subprocess.Popen(command)
        
        print(START_INFO % ( N_WORKERS, modelclass, self.rulename))
        
        self.pid=p.pid
        
        
class ASGD(Rule):
    
    def __init__(self):
        Rule.__init__(self)
        pass

        
class GOSGD(Rule):
    
    '''Gossip SGD
    
    When to exchange: workers run iterations asynchronously and exchange when drawing a success
    
    See: 
        https://arxiv.org/abs/1611.09726
    '''
    
    def __init__(self):
        Rule.__init__(self)
        
        self.rulename = 'GOSGD'
        
    def init(self, devices, modelfile, modelclass):
        
        N_WORKERS = len(devices)
        
        env = dict(os.environ)

        command = ["mpirun"]
        
        for index, device in enumerate(devices):
            
            # command += ["--output-filename", "%s" % 'out']
            command += ["--mca", "mpi_warn_on_fork", "0"]
            command += ["--mca", "btl_smcuda_use_cuda_ipc", "1"]
            command += ["--mca", "mpi_common_cuda_cumemcpy_async", "1"]
            command += ["--mca", "mpi_max_info_val", "10240"]
            #command += ["-np", str(len(hosts))]
            #command += ["-H", ','.join(hosts)]
            #command += ["--map-by", "ppr:4:node"]
            command += shlex.split("-x " + " -x ".join(env.keys()))
            command += ["-n", "%d" % 1]
            command += ["--bind-to", "none"]
            # command += ["--report-bindings"]
 
            worker_file_dir = os.path.dirname(os.path.realpath(__file__))
            command += [sys.executable, "-u", worker_file_dir+"/gosgd_worker.py"] 
        
            command += [device, modelfile,  modelclass]
            
            if index!= N_WORKERS-1:
                command += [":"]
                
        p = subprocess.Popen(command)
        
        print(START_INFO % ( N_WORKERS, modelclass, self.rulename))
        
        self.pid=p.pid
    
    
        
    
    