import numpy as np 
import torch
import yaml
from models.netam import TransformerAN
from models.netam import LSTMnetAM
from models.genmodel import GenerateModel
from utils import FileLogger
from bin.tools.tester import Tester,MaxDecider

model_config1 = {'name':'models.netam:LSTMnetAM','input_size':13, 'hidden_size':256, 'num_layers':3, 'dropout':0.1, 'bidirectional':True, 'output_size':10}

E,B,C,D = (10,4,80,13)

if __name__ == "__main__":
    data = torch.Tensor(E,B,C,D)
    label = 2*torch.ones(E,B,C).int()
    print(data.size())
    #am1 = LSTMnetAM(D,256,3,0.1,True,10)
    am1 = GenerateModel(model_config1)
    am2 = TransformerAN(D,256,1024,4,3,0.1,3)
    my_test = Tester(am2,MaxDecider)
    for i in range(E):
        o2 = am2(data[i])
        print(o2.size())
        d2 = my_test.decider(data[i])
        a2 = my_test.accurate(data[i],label[i])
        print(a2)
    
