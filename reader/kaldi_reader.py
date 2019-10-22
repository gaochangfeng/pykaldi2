from __future__ import print_function
from kaldi.util.table import SequentialMatrixReader
import numpy as np

def Read_KaldiFeat(file_path,file_type="scp",delta = False):    
    s1 = "%s:%s"%(file_type,file_path)
    keys = []
    datas = []
    with SequentialMatrixReader(s1) as f:
        for key,data in f:            
            keys.append(key)     
            data = data.numpy()    
            if delta:
                delta1 = np.diff(data,n=1,axis=0)
                delta2 = np.diff(data,n=2,axis=0)
                t = np.shape(delta2)[0]
                data = np.concatenate([data[:t],delta1[:t],delta2],axis=-1) 
            datas.append(data)    
    return keys,datas