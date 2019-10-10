import sys
import torch.utils.data as data
import torch
import numpy as np
import os
from data.sr_dataset import _utt2seg
from kaldi_io import read_mat_scp

class FeatDataSet(data.Dataset):
    def __init__(self, config):
        print("init")
        self._config = config
        self.seg_len = config['data_config']['seg_len']
        self.seg_shift = config['data_config']['seg_shift']
        self.feat_buff = None
        self.label_buff = None
        self.aux_label = None
        self.load_ark_file(config)
        self.train_samples = self.generate_list()

    def load_ark_file(self,config):
        print("load")
        self.feat_buff = np.random.rand(10000,13)
        self.label_buff = np.random.randint(low=0,high=1,size=[10000])
        #utt_name, feat_data = zip(*read_mat_scp(data_scp))
        self._load_streams(config["source_paths"], config['data_path'], is_speech=True, is_rir=False)

    def _load_streams(self, source_list, data_path, is_speech=True, is_rir=False):
        for i in range(len(source_list)):
            corpus_type = source_list[i]['type']
            corpus_wav_path = data_path+source_list[i]['wav']
            label_paths = []
            label_names = []
            if 'label' in source_list[i]:
                label_paths.append(data_path+source_list[i]['label'])
                label_names.append('label')
            else:
                corpus_label_path = None
            if 'aux_label' in source_list[i]:
                label_paths.append(data_path+source_list[i]['aux_label'])
                label_names.append('aux_label')
            else:
                corpus_label_path = None
            print("%s::_load_streams: loading %s from %s..." % (self.__class__.__name__, corpus_type, corpus_wav_path))
            utt_name, feat_data = zip(*read_mat_scp(corpus_wav_path))
            for feat in feat_data:
                print(np.shape(feat))
            print(label_paths)


    def generate_list(self):
        print("gensrate_list")
        utt_id = 1
        if self._config['data_config']['sequence_mode']:
            if self._config.load_label:
                train_samples = [(self.feat_buff, utt_id, self.label_buff, self.aux_label)]
            else:
                train_samples = [(self.feat_buff, utt_id)]
        else: 
            fbank_seg = _utt2seg(self.feat_buff.T, self.seg_len, self.seg_shift)
            if len(fbank_seg) == 0:
                return []

            if self._config['data_config']['load_label']:
                label_seg = _utt2seg(self.label_buff.T, self.seg_len, self.seg_shift)
                train_samples = [(fbank_seg[i].T, utt_id, label_seg[i].T) for i in range(len(label_seg))]
            else:
                train_samples = [(fbank_seg[i].T, utt_id) for i in range(len(fbank_seg))]
        return train_samples

    def __getitem__(self, index):

        return self.train_samples[index]


    def __len__(self):
        return len(self.train_samples)

    def __repr__(self):
        print("repr")
        return "fmt_str"