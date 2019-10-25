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
from data.datacenter import DataCenter
from models.genmodel import GenerateModel
from models.criterion.cross_entropy import SeqCrossEntorpy
from bin.tools.trainer import Trainer,update_opt_func
from utils import utils,FileLogger

@update_opt_func
def opt_updata(optimizer,rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= rate
    return optimizer

class Chunk_Trainer(Trainer):
    def train_batch(self,data,label,l,c,r,*arg,max_grad_norm=5,**args):
        out = self.forward(data,*arg,**args)
        loss = self.loss_fn(out[:,l:l+c],label[:,l:l+c],*arg,**args)
        self._backward(loss,max_grad_norm)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir")
    parser.add_argument("-dataPath", default='', type=str, help="path of data files")
    parser.add_argument("-train_config")
    parser.add_argument("-data_config")
    parser.add_argument("-data_type",type=str,default="wav",help="data type input")
    parser.add_argument("-model", type=str, default="models.LSTMnetAM", help="the model from which you want to resume training")
    parser.add_argument("-lr", default=0.0001, type=float, help="Override the LR in the config")
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("-max_grad_norm", default=5, type=float, help="max_grad_norm for gradient clipping")
    parser.add_argument("-sweep_size", default=200, type=float, help="process n hours of data per sweep (default:200)")
    parser.add_argument("-num_epochs", default=1, type=int, help="number of training epochs (default:1)")
    parser.add_argument("-global_mvn", default=False, type=bool, help="if apply global mean and variance normalization")
    parser.add_argument("-resume_from_model", type=str, help="the model from which you want to resume training")
    parser.add_argument("-dropout", type=float, help="set the dropout ratio")
    parser.add_argument("-anneal_lr_epoch", default=2, type=int, help="start to anneal the learning rate from this epoch") 
    parser.add_argument("-anneal_lr_ratio", default=0.5, type=float, help="the ratio to anneal the learning rate")
    parser.add_argument('-print_freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-hvd', default=False, type=bool, help="whether to use horovod for training")

    args = parser.parse_args()

    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

     # Initialize Horovod
    if args.hvd:
        import horovod.torch as hvd
        hvd.init()
        th.cuda.set_device(hvd.local_rank())
        print("Run experiments with world size {}".format(hvd.size()))

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    data_center = DataCenter(args.data_config)

    print("Data center set up successfully!")
    print("Data source:",data_center.source_paths)

    # ceate model
    model_config = config["model_config"]
    model = GenerateModel(model_config)
    # Start training
    th.backends.cudnn.enabled = True
    if th.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    if args.hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    criterion = SeqCrossEntorpy(ignore_index=-100)
    trainer = Chunk_Trainer(model,criterion,optimizer)
    start_epoch = 0
    if args.resume_from_model:
        assert os.path.isfile(args.resume_from_model), "ERROR: model file {} does not exit!".format(args.resume_from_model)
        checkpoint = th.load(args.resume_from_model)
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' ".format(args.resume_from_model))
    logger = FileLogger(args.exp_dir+'/run.log','train')

    for epoch in range(start_epoch, args.num_epochs):

        if epoch > args.anneal_lr_epoch:
            trainer.update_opt(opt_updata,args.anneal_lr_ratio)
        run_time, loss = run_train_epoch(trainer, data_center, epoch, args ,config)
        # save model
        if not args.hvd or hvd.rank()== 0:
            checkpoint={}
            checkpoint['model']=model.state_dict()
            checkpoint['optimizer']=optimizer.state_dict()
            checkpoint['epoch']=epoch
            output_file=args.exp_dir + '/model.'+ str(epoch) +'.tar'
            th.save(checkpoint, output_file)
            s = 'epoch%d: Time %6.3f Loss %.4e\n' % (epoch, run_time, loss)
            logger.info(s)


def run_train_epoch(trainer, data_center, epoch, args, config):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    grad_norm = utils.AverageMeter('grad_norm', ':.4e')
    progress = utils.ProgressMeter(data_center.data_num, batch_time, losses, grad_norm,
                             prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    # trainloader is an iterator. This line extract one minibatch at one time
    left = config['data_config']['left']
    center = config['data_config']['center']
    right = config['data_config']['right']
    data_center.refresh()
    all = 0
    print("epoch")
    for train_dataloader in data_center.GetChunkDataLoder(config['data_config']['seg_len'],config['data_config']['seg_shift'],\
                                                            args.batch_size,args.hvd,args.data_loader_threads):
        for i, data in enumerate(train_dataloader, 0):
            feat = data["x"]
            label = data["y"]
            x = feat.to(th.float32)
            y = label.unsqueeze(2).long()            
            if th.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            loss = trainer.train_batch(x,y,left,center,right)
            losses.update(loss.item(), x.size(0))
            batch_time.update(time.time() - end)
            
            if i % args.print_freq == 0:
                progress.print(i + all)
        all = all + len(train_dataloader)

    return time.time()-end,losses.avg

if __name__ == '__main__':
    main()
