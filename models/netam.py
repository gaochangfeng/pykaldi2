import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
from .import LSTMStack

class BaseAM(nn.Module):
    def forward(self,data):
        raise NotImplementedError("AM forward not implement")

    def recognize(self,data):
        raise NotImplementedError("AM recognise not implement")

    def align(self,data,label):
        raise NotImplementedError("AM align not implement")


class NnetAM(BaseAM):
    
    def __init__(self, nnet, hidden_size, output_size):
        super(NnetAM, self).__init__()

        self.nnet = nnet
        self.output_size = output_size
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        nnet_output = self.nnet(data)
        output = self.output_layer(nnet_output)

        return output


class LSTMnetAM(BaseAM):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, output_size):
        super(LSTMnetAM, self).__init__()

        self.lstm = LSTMStack(input_size, hidden_size, num_layers, dropout, bidirectional)
        if bidirectional:
            self.hidden_size = 2*hidden_size
        else:
            self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_layer = nn.Linear(self.hidden_size, output_size)

    def forward(self, data):
        nnet_output, (h,c) = self.lstm(data)
        output = self.output_layer(nnet_output)

        return output
