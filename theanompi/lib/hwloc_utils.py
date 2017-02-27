'''
Optional process and memory binding module for improved performance

'''

import hwloc 
policy=hwloc.MEMBIND_BIND
flag_mem = 0 #hwloc.MEMBIND_PROCESS
flag_cpu = hwloc.CPUBIND_PROCESS
 
def range_to_list(_range):
    
    ids=[]
    
    if _range.split(',')[0] == _range and _range.split('-')[0] == _range:
        ids.extend(int(_range))
    elif _range.split(',')[0] != _range:
        for x in map(str.strip, _range.split(',')):
            if x.isdigit():
                ids.append(int(x))
                continue
            if '-' in x:
                xr = map(str.strip,x.split('-'))
                ids.extend(range(int(xr[0]),int(xr[1])+1))
                continue
            else:
                raise Exception, 'unknown range type: "%s"'%x
    elif _range.split('-')[0] != _range:
        xr = map(str.strip, _range.split('-'))
        ids.extend(range(int(xr[0]),int(xr[1])+1))
        
    res = ",".join(map(str,ids))
    
    return res

              
# @param:
#       cpulist: a comma delimited string of cpu core numbers
#       label:   a string to be stored as part of a key in env
def bind_to_socket_mem(cpulist, label=None):
    
    if type(cpulist)!=str:

        raise RuntimeError('cpulist should be a comma delimited str')
    
    topology = hwloc.Topology()

    topology.load()
    
    cpuset = topology.get_cpubind(flag_cpu)
    
    cpuset.list_sscanf(cpulist)
    
    topology.set_cpubind(cpuset, flag_cpu)
    topology.set_membind(cpuset, policy,flag_mem)
    
    import os
    os.environ['CPULIST_%s' % label] = range_to_list(cpulist)
        
    
def detect_socket_num(debug=True, label=None):
    
    topology = hwloc.Topology()
    topology.load()
    # Check a process's socketnum
    #1. get current cpubind 
    cpuset = topology.get_cpubind(flag_cpu)
    
    cpuset_mem, policy = topology.get_membind(flag_mem)
    #2. get the parent of the last cpu location
    node = topology.get_obj_covering_cpuset(cpuset)
    
    socketnum=node.nodeset.first
    
    if debug:
        
        import os
        print '%s pid %d run on cpuset %s (%s) sock %s, bind to mem cpuset %s (%s)' % \
                            (label, os.getpid(),cpuset, cpuset.list_asprintf(), 
                                socketnum, cpuset_mem, cpuset_mem.list_asprintf())
    
    
    return cpuset, socketnum

if __name__ == '__main__':
    
    import sys

    cpulist=sys.argv[1]

    bind_to_socket_mem(cpulist)

    detect_socket_num()