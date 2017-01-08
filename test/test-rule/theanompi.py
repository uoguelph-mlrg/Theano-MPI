import shlex
import sys
import os
import subprocess
import signal


sync_type = 'avg' # or 'cdd'


class BSP(object):
    
    def __init__(self,config):
        
        self.pid =None
        self.sync_type=sync_type
        
    def init(self, devices):
        
        N_WORKERS = len(devices)
        
        env = dict(os.environ)

        command = ["mpirun"]
        
        for index, device in enumerate(devices):
            
            #command += ["--output-filename", "%s" % 'out']
            command += ["--mca", "mpi_warn_on_fork", "0"]
            command += ["--mca", "btl_smcuda_use_cuda_ipc", "1"]
            command += ["--mca", "mpi_common_cuda_cumemcpy_async", "1"]
            #command += ["-np", str(len(hosts))]
            #command += ["-H", ','.join(hosts)]
            #command += ["--map-by", "ppr:1:node"]
            command += shlex.split("-x " + " -x ".join(env.keys()))
            command += ["-n", "%d" % 1]
            command += ["--bind-to", "none"]
            command += [sys.executable, "-u", "worker.py"] 
        
            command += [device, self.sync_type]
            
            if index!= N_WORKERS-1:
                command += [":"]
                
        p = subprocess.Popen(command)
        
        print("started %d workers " % N_WORKERS)
        
        self.pid=p.pid
        
    def wait(self):
        
        try:
            pid, status = os.waitpid(self.pid, 0)
            if pid != self.pid:
                print("\nWARNING! Received status for unknown process {}".format(pid))
                sys.exit(3)
            if os.WIFEXITED(status):
                rcode = os.WEXITSTATUS(status)
                print("\n## terminated with return code: {}.".format(rcode))
                if rcode != 0:
                    print("\nAn error has occured.")
                    sys.exit(1)
                else:
                    print("worker %d finished." % pid)
            
        except (RuntimeError, KeyboardInterrupt):

            print("Killing worker processes...")

            os.kill(self.pid, signal.SIGTERM)
            pid, status = os.waitpid(self.pid, 0)
            
            sys.exit(3)

        
        
    
    