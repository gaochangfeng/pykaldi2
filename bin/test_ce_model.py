"""
Copyright (c) 2019 Microsoft Corporation. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

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
from data import SpeechDataset, ChunkDataloader, SeqDataloader, FeatDataSet
from models.genmodel import GenerateModel
from models.criterion.cross_entropy import SeqCrossEntorpy
from bin.tools.tester import Tester,MaxDecider,SameChecker
from utils import utils,FileLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir")
    parser.add_argument("-dataPath", default='', type=str, help="path of data files")
    parser.add_argument("-train_config")
    parser.add_argument("-data_config")
    parser.add_argument("-data_type",type=str,default="wav",help="data type input")
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("-sweep_size", default=200, type=float, help="process n hours of data per sweep (default:200)")
    parser.add_argument("-global_mvn", default=False, type=bool, help="if apply global mean and variance normalization")
    parser.add_argument("-resume_from_model", type=str, help="the model from which you want to resume training")
    parser.add_argument("-dropout", type=float, help="set the dropout ratio")
    parser.add_argument('-print_freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-hvd', default=False, type=bool, help="whether to use horovod for training")

    args = parser.parse_args()

    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    config["sweep_size"] = args.sweep_size
    with open(args.data_config) as f:
        data = yaml.safe_load(f)
        config["source_paths"] = [j for i, j in data['clean_source'].items()]
        if 'dir_noise' in data:
            config["dir_noise_paths"] = [j for i, j in data['dir_noise'].items()]
        if 'rir' in data:
            config["rir_paths"] = [j for i, j in data['rir'].items()]

    config['data_path'] = args.dataPath

    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

     # Initialize Horovod
    if args.hvd:
        import horovod.torch as hvd
        hvd.init()
        th.cuda.set_device(hvd.local_rank())
        print("Run experiments with world size {}".format(hvd.size()))

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)
    if args.data_type == "wav":
        trainset = SpeechDataset(config)
    elif args.data_type == "feat":
        trainset = FeatDataSet(config)
    else:
        raise TypeError("only wav or feat data_type")
    train_dataloader = ChunkDataloader(trainset,
                                       batch_size=args.batch_size,
                                       distributed=args.hvd,
                                       num_workers=args.data_loader_threads)

    if args.global_mvn:
        transform = GlobalMeanVarianceNormalization()
        print("Estimating global mean and variance of feature vectors...")
        transform.learn_mean_and_variance_from_train_loader(trainset,
                                                        trainset.stream_idx_for_transform,
                                                        n_sample_to_use=2000)
        trainset.transform = transform
        print("Global mean and variance transform trained successfully!")

        with open(args.exp_dir+"/transform.pkl", 'wb') as f:
            pickle.dump(transform, f, pickle.HIGHEST_PROTOCOL)

    print("Data loader set up successfully!")
    print("Number of minibatches: {}".format(len(train_dataloader)))

    # ceate model
    model_config = config["model_config"]
    model = GenerateModel(model_config)
    # Start training
    th.backends.cudnn.enabled = True
    if th.cuda.is_available():
        model.cuda()

    tester = Tester(model,MaxDecider,SameChecker)
    assert os.path.isfile(args.resume_from_model), "ERROR: model file {} does not exit!".format(args.resume_from_model)
    checkpoint = th.load(args.resume_from_model)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' ".format(args.resume_from_model))

    logger = FileLogger(args.exp_dir+'/acc.log','test')
    acc_list = []
    for i, data in enumerate(train_dataloader, 0):
        feat = data["x"]
        label = data["y"]
        x = feat.to(th.float32)
        y = label.unsqueeze(2).int()
        if th.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        acc = tester.accurate(x,y).item()
        acc_list.append(acc)
        logger.info("batch %d:%f"%(i,acc))
    all_acc = np.mean(acc_list)
    logger.info("all:%f\n"%(acc))

if __name__ == '__main__':
    main()
