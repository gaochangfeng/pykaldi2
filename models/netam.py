import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
from .net.lstm import LSTMStack
from .net.transformer.encoder import Encoder as TransformerEncoder

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

    def recognize(self,data):
        output = self.forward(data)
        output = th.softmax(output)

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

class TransformerAN(BaseAM):
    
    def __init__(self, input_size, hidden_size, linear_units, heads, num_layers, dropout, output_size):
        super(TransformerAN, self).__init__()
        self.trans = TransformerEncoder(idim=input_size,
                 attention_dim=hidden_size,
                 attention_heads=heads,
                 linear_units=linear_units,
                 num_blocks=num_layers,
                 dropout_rate=dropout,
                 positional_dropout_rate=dropout,
                 attention_dropout_rate=0.0,
                 input_layer="linear"
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self,data,mask=None):
        output,mask = self.trans(data,mask)
        output = self.output_layer(output)
        return output,mask

    def recognize(self,data,mask):
        output,mask = self.forward(data,mask)
        output = th.softmax(output)

        return output,mask