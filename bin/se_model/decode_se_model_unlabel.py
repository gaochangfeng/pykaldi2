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
from data import SeqDataloader,FeatDataSet,SpeechDataset,SeqFeatDataSet
from models.genmodel import GenerateModel
from bin.tools.kaldi_decoder import Kaldi_Decoder as Decoder
from kaldi.util.table import SequentialMatrixReader
from utils import utils,FileLogger
import kaldi.util as kaldi_util

#beam,max_active,mdl,fst,word,acoustic_scale=1.0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir")
    parser.add_argument("-dataPath", default='', type=str, help="path of data files")
    parser.add_argument("-train_config")
    parser.add_argument("-data_type",type=str,default="wav",help="data type input")
    parser.add_argument("-acoustic_scale",type=float,default=1.0,help="acoustic_scale")
    parser.add_argument("-beam_size", default=10, type=int, help="Override the beam size in the config")
    parser.add_argument("-max_active", default=7000, type=int, help="Override the batch size in the config")
    parser.add_argument("-mdl", default="final.mdl", type=str, help="Override the batch size in the config")
    parser.add_argument("-fst", default="HCLG.fst", type=str, help="Override the batch size in the config")
    parser.add_argument("-word", default="words.txt", type=str, help="Override the batch size in the config")
    parser.add_argument("-prior_path", default="final.occs", type=str, help="the prior for decoder, usually named as final.occs in kaldi setup")
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("-sweep_size", default=200, type=float, help="process n hours of data per sweep (default:200)")
    parser.add_argument("-global_mvn", default=False, type=bool, help="if apply global mean and variance normalization")
    parser.add_argument("-resume_from_model", type=str, help="the model from which you want to resume training")


    args = parser.parse_args()

    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    config['data_path'] = args.dataPath

    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))


    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    # ceate model
    model_config = config["model_config"]
    model = GenerateModel(model_config)
    # Start training
    th.backends.cudnn.enabled = True
    if th.cuda.is_available():
        model.cuda()
    if os.path.isfile(args.prior_path):
        prior = kaldi_util.io.read_matrix(args.prior_path).numpy()
        log_prior = th.tensor(np.log(prior[0]/np.sum(prior[0])), dtype=th.float)
    else:
        log_prior = None

    assert os.path.isfile(args.resume_from_model), "ERROR: model file {} does not exit!".format(args.resume_from_model)
    checkpoint = th.load(args.resume_from_model)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' ".format(args.resume_from_model))
    decoder = Decoder(args.beam_size,args.max_active,args.mdl,args.fst,args.word,acoustic_scale=args.acoustic_scale)
    logger = FileLogger(args.exp_dir+'/decode.log','decode')
    logger.info("Decode Begin")
    start_time = time.time()
    
    with SequentialMatrixReader("scp:"+args.dataPath) as l:
        for key,feat in l:
            print(key)
            feat = th.from_numpy(feat.numpy()).unsqueeze(0)
            x = feat.to(th.float32)                        
            x = x.cuda()                                    
            pre = model.recognize(x).cpu()[0]
            if log_prior is not None:
                pre = pre - log_prior
            pre = pre.numpy()
            utt = decoder.decode_loglike(pre)
            logger.info(utt['text'])
            with open(args.exp_dir+'/out_txt.txt','a') as f:
                f.write(key+' '+utt['text']+'\n')
    timecost = time.time()-start_time
    logger.info("Debug End")
    logger.info("cost time:"+str(timecost))

if __name__ == '__main__':
    main()

# pre = data["y"][0] 
# pre = th.zeros(pre.size(0), 5720).scatter_(1, pre, 0.99)
# pre = pre +0.009
# pre = np.log(pre.detach().cpu().numpy())
# utt = decoder.decode_loglike(pre)