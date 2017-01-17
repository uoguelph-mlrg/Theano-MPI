import shlex
import sys
import os
import subprocess
import signal

class BSP(object):
    
    sync_type = 'avg' # or 'cdd'
    
    # Bulk Synchronous Parallel rule
    
    # When to exchange: workers run iterations synchronously and exchange after each iteration
    
    
    def __init__(self):
        
        self.pid =None
        
    def init(self, devices, modelfile, modelclass):
        
        N_WORKERS = len(devices)
        
        env = dict(os.environ)

        command = ["mpirun"]
        
        for index, device in enumerate(devices):
            
            command += ["--output-filename", "%s" % 'out']
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
            command += [sys.executable, "-u", "worker.py"] 
        
            command += [device, BSP.sync_type, modelfile,  modelclass]
            
            if index!= N_WORKERS-1:
                command += [":"]
                
        p = subprocess.Popen(command)
        
        print("Theano-MPI started %d workers working on \n 1.iterating on updating %s and\n 2.exchange their params with BSP(%s)\nSee output log." % ( N_WORKERS, modelclass, BSP.sync_type))
        
        self.pid=p.pid
        
    def wait(self):
        
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

        
        
    
    