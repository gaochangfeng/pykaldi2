import numpy as np 
import torch
import yaml
import random
from .f_dataset import FeatDataSet
from .dataloader import ChunkDataloader
import threading
import copy
from threading import Semaphore
import time


class DataCenter(object):
    def __init__(self,data_config):
        self.config = {}
        with open(data_config) as f:
            data = yaml.safe_load(f)
            self.source_paths = [j for i, j in data['clean_source'].items()]
            if 'dir_noise' in data:
                self.dir_noise_paths = [j for i, j in data['dir_noise'].items()]
            else:
                self.dir_noise_paths = None
            if 'rir' in data:
                self.rir_paths = [j for i, j in data['rir'].items()]
            else:
                self.rir_paths = None
        self.data_num = len(self.source_paths)
        self.pos = 0
        self.seq = list(range(self.data_num))
        self.loader = None
        print(self.seq)
        self.Wmutex = Semaphore(1)
        self.Rmutex = Semaphore(0)
    '''
    def GetChunkDataLoder(self,seq_len,seq_hop,batch_size,distributed,data_loader_threads):
        while self.pos < self.data_num:
            data_set = FeatDataSet(seq_len,seq_hop)
            data_set.load_ark_file([self.source_paths[self.seq[self.pos]]])
            train_dataloader = ChunkDataloader(data_set,
                                           batch_size=batch_size,
                                           distributed=distributed,
                                           num_workers=data_loader_threads)
            self.pos = self.pos+1
            print("before yalid",self.pos)
            yield train_dataloader
            print("after yalid",self.pos)
    '''
    def GetChunkDataLoder(self,seq_len,seq_hop,batch_size,distributed,data_loader_threads):
        t = threading.Thread(target=self.chunk_write_thread,args=(seq_len,seq_hop,batch_size,distributed,data_loader_threads, ))
        t.start()
        while self.pos < self.data_num:
            t1 = time.time()
            print("wait write")
            self.Rmutex.acquire()
            train_dataloader = copy.copy(self.loader)
            self.pos = self.pos + 1
            self.Wmutex.release()
            if train_dataloader is not None:
                print("wait write for " + str(time.time() - t1))
                yield train_dataloader            
        t.join()
    
            
    def refresh(self):
        self.pos = 0
        random.shuffle(self.seq)



    def chunk_write_thread(self,seq_len,seq_hop,batch_size,distributed,data_loader_threads):        
        while True:
            print("wait read")
            self.Wmutex.acquire()
            if self.pos >= self.data_num:
                self.Rmutex.release()
                self.loader = None
                return
            data_set = FeatDataSet(seq_len,seq_hop)
            data_set.load_ark_file([self.source_paths[self.seq[self.pos]]])
            train_dataloader = ChunkDataloader(data_set,
                                           batch_size=batch_size,
                                           distributed=distributed,
                                           num_workers=data_loader_threads)
            self.loader = train_dataloader
            self.Rmutex.release()
        
        
        
        
