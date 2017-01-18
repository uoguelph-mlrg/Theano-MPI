from __future__ import absolute_import
import shlex
import sys
import os
import subprocess
import signal

class Rule(object):
    
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
            
            'Rule %s not initiated' % self.rulename
            
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
    
    sync_type = 'avg' # or 'cdd'
    exch_strategy = 'nccl32' # asa32, asa16, copper, copper16, nccl16, nccl32 or ar
    
    # Bulk Synchronous Parallel rule
    
    # When to exchange: workers run iterations synchronously and exchange after each iteration
    
    
    def __init__(self):
        Rule.__init__(self)
        
        self.rulename = 'BSP'
        
    def init(self, devices, modelfile, modelclass):
        
        N_WORKERS = len(devices)
        
        env = dict(os.environ)

        command = ["mpirun"]
        
        for index, device in enumerate(devices):
            
            # command += ["--output-filename", "%s" % 'out']
            command += ["--mca", "mpi_warn_on_fork", "0"]
            command += ["--mca", "btl_smcuda_use_cuda_ipc", "1"]
            command += ["--mca", "mpi_common_cuda_cumemcpy_async", "1"]
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
        
        print("Theano-MPI started %d workers for \n 1.updating %s params through iterations and\n 2.exchange the params with BSP(%s)\nSee output log." % ( N_WORKERS, modelclass, BSP.sync_type))
        
        self.pid=p.pid


class EASGD(Rule):
    
    '''Elastic Averaging SGD
    
    '''
    def __init__(self):
        Rule.__init__(self)
        
        self.rulename = 'BSP'
    
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
            if index==0:  # the server
                command += ["--report-uri", "./ompi-server.txt"]
            else: # the workers
                command += ["--ompi-server","file:./ompi-server.txt"]
            
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
        
        print("Theano-MPI started %d workers and one server for \n 1.updating %s params through iterations and\n 2.exchange the params with EASGD\nSee output log." % ( N_WORKERS -1 , modelclass))
        
        self.pid=p.pid
        
        
class ASGD(Rule):
    
    def __init__(self):
        Rule.__init__(self)
        pass

        
class GoSGD(Rule):
    
    '''Gossip SGD
    
    https://arxiv.org/abs/1611.09726
    '''
    
    def __init__(self):
        Rule.__init__(self)
        pass
    
    
        
    
    