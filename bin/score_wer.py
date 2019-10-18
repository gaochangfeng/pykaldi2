import yaml
import argparse
import numpy as np
import os
import sys
import time
import json
import pickle

import torch as th
import torch.nn as nn

from reader.preprocess import GlobalMeanVarianceNormalization
from data import SpeechDataset, ChunkDataloader, SeqDataloader
from models import LSTMStack, NnetAM, LSTMnetAM
from utils.score import Scorer
from utils.logger import FileLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-worddict", type=str, help="the word dictionary")
    parser.add_argument("-text_ref", type=str, help="the reference text")
    parser.add_argument("-text_decode", type=str, help="the decoder's output")
    args = parser.parse_args()
    
    if not os.path.isfile(args.worddict):
        raise ValueError("no word dictionary find in " + args.worddict)

    if not os.path.isfile(args.text_ref):
        raise ValueError("no word dictionary find in " + args.text_ref)

    if not os.path.isfile(args.text_decode):
        raise ValueError("no word dictionary find in " + args.text_decode)

    my_s = Scorer(args.worddict)
    ref_dict = {}
    dec_dict = {}
    with open(args.text_ref) as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(' ')
            if len(l)<=1:
                print("Warning,"+l[0]+" do not have text")
            else:
                ref_dict[l[0]] = l[1:]

    with open(args.text_decode) as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(' ')
            if len(l)<=1:
                print("Warning,"+l[0]+" do not have text")
            else:
                dec_dict[l[0]] = l[1:]

    err_num=0
    all_num=0
    for id in dec_dict.keys():
        err_num += my_s.edit_distance(ref_dict[id],dec_dict[id])
        all_num += len(dec_dict[id])
    print('wer = ' + str(1.0*err_num/all_num))

    
if __name__ == "__main__":
        main()