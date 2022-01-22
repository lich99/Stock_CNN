
import os
import re 

def query_gpu(qargs=[]):
    qargs =['index','gpu_name', 'memory.free']+ qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return results

def select_gpu(results, thres=4096):
    avali = []
    try:
        for i, line in enumerate(results):
            if int(re.findall('(.*), (.*?) MiB', line)[0][-1]) > thres:
                avali.append(i)
        return avali
    except:
        return ''