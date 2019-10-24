import sys
import torch.utils.data as data
import torch
import numpy as np
import os
from data.sr_dataset import _utt2seg
from kaldi_io import read_mat_scp
from reader.stream import my_cat


class FeatDataSet(data.Dataset):
    def __init__(self, config):
        print("init")
        self._config = config
        self.seg_len = config['data_config']['seg_len']
        self.seg_shift = config['data_config']['seg_shift']
        self.feat_buff = None
        self.label_buff = None
        self.utt_name = None
        self.aux_label = None
        self.train_samples = []
        self.load_ark_file(config)
        

    def load_ark_file(self,config):
        print("load")
        # self.feat_buff = np.random.rand(10000,13)
        # self.label_buff = np.random.randint(low=0,high=1,size=[10000])
        utt_id_wav,utt_data_wav,utt_label_wav = self._load_streams(config["source_paths"], config['data_path'], is_speech=True, is_rir=False)
        for i in range(len(utt_data_wav)):#travel different data source
            assert len(utt_data_wav[i])==len(utt_label_wav[i]) #check the label seq has same length with the feat seq
            for j in range(len(utt_data_wav[i])):#travel different wavfile 
                #print(config['data_config']['left'],np.shape(utt_data_wav[i][j])[-1])
                #padxl = np.zeros([config['data_config']['left'],np.shape(utt_data_wav[i][j])[-1]])
                #padyl = np.zeros(config['data_config']['left'])
                #padxr = np.zeros([config['data_config']['right'],np.shape(utt_data_wav[i][j])[-1]])
                #padyr = np.zeros(config['data_config']['right'])
                #utt_data_wav[i][j] = np.concatenate((padxl,utt_data_wav[i][j],padxr),axis=0)
                #utt_label_wav[i][j][0] = np.concatenate((padyl,utt_label_wav[i][j][0],padyr),axis=0)
                #print(np.shape(utt_data_wav[i][j]))
                self.train_samples+=self.generate_list(utt_id_wav[i][j],utt_data_wav[i][j],utt_label_wav[i][j][0],aux_label=None)
                #cat_wav = np.concatenate((padxl,utt_data_wav[i][j],padxr),axis=0)
                #cat_lab = np.concatenate((padyl,utt_label_wav[i][j][0],padyr),axis=0)
                #self.train_samples+=self.generate_list(utt_id_wav[i][j],cat_wav,cat_lab,aux_label=None)

    def _load_streams(self, source_list, data_path, is_speech=True, is_rir=False):
        '''
            utt_id_wav:list(source)+list(wav id)
            utt_data_wav:list(source)+list(wav)+list(time)+list(feat)
            utt_label_wav:list(source)+list(wav)+list(1)+list(label)
        '''
        utt_id_wav = []
        utt_data_wav = []
        utt_label_wav = []

        label_paths = []
        label_names = []
        for i in range(len(source_list)):
            corpus_type = source_list[i]['type']
            corpus_wav_path = data_path+source_list[i]['wav']            
            if 'label' in source_list[i]:
                label_paths.append(data_path+source_list[i]['label'])
                label_names.append('label')
            else:
                corpus_label_path = None
            print("%s::_load_streams: loading %s from %s..." % (self.__class__.__name__, corpus_type, corpus_wav_path))
            utt_name, feat_data = zip(*read_mat_scp(corpus_wav_path))
            utt_id_wav.append(utt_name)
            utt_data_wav.append(list(feat_data))
        
        for i in range(len(label_paths)):
            given_utt_id = set(utt_id_wav[i])
            print(label_paths[i],"given utt:",len(utt_id_wav[i]))
            lines = my_cat(label_paths[i])
            lines.sort()        # each lines start with utterance ID, hence effectively sort the labels with utterance ID.
            curr_utt_id_label = [i.split(" ")[0] for i in lines]
            selected_utt_id = set(curr_utt_id_label) & given_utt_id
            print("select utt:",len(selected_utt_id))
            if len(selected_utt_id)!=len(given_utt_id):
                print(given_utt_id-selected_utt_id,"is not exist in label file")
            # selected_utt_id = set(curr_utt_id_label)
            selected_label_list = get_label(lines,selected_utt_id)
            utt_label_wav.append(list(selected_label_list))

        return utt_id_wav,utt_data_wav,utt_label_wav


    def generate_list(self,utt_id,feat_buff,label_buff,aux_label):
        if len(feat_buff)!=len(label_buff):
            print("Warning!",utt_id," feat length is not equal with label length",len(feat_buff),len(label_buff))
            minlen = min(len(feat_buff),len(label_buff))
            feat_buff = feat_buff[:minlen]
            label_buff = label_buff[:minlen]
        if self._config['data_config']['sequence_mode']:
            if self._config.load_label:
                train_samples = [(feat_buff, utt_id, label_buff, aux_label)]
            else:
                train_samples = [(feat_buff, utt_id)]
        else: 
            fbank_seg = _utt2seg(feat_buff.T, self.seg_len, self.seg_shift)
            if len(fbank_seg) == 0:
                return []

            if self._config['data_config']['load_label']:
                label_seg = _utt2seg(label_buff.T, self.seg_len, self.seg_shift)
                train_samples = [(fbank_seg[i].T, utt_id, label_seg[i].T) for i in range(len(label_seg))]
            else:
                train_samples = [(fbank_seg[i].T, utt_id) for i in range(len(fbank_seg))]
        # print(train_samples[0])
        return train_samples

    def __getitem__(self, index):

        return self.train_samples[index]


    def __len__(self):
        return len(self.train_samples)

    def __repr__(self):
        print("repr")
        return "fmt_str"

class SeqFeatDataSet(data.Dataset):
    def __init__(self, config):
        print("init")
        self._config = config
        self.seg_len = config['data_config']['seg_len']
        self.seg_shift = config['data_config']['seg_shift']
        self.feat_buff = None
        self.label_buff = None
        self.utt_name = None
        self.aux_label = None
        self.train_samples = []
        self.load_ark_file(config)
    
    def load_ark_file(self,config):
        print("load")
        utt_id_wav,utt_data_wav,utt_label_wav,utt_aux_wav = self._load_streams(config["source_paths"], config['data_path'], is_speech=True, is_rir=False)
        print(len(utt_id_wav),len(utt_data_wav),len(utt_label_wav),len(utt_aux_wav))
        for i in range(len(utt_id_wav)):
            for j in range(len(utt_data_wav[i])):
                self.train_samples.append([utt_data_wav[i][j],utt_id_wav[i][j],utt_label_wav[i][j].T,utt_aux_wav[i][j].T])

    def _load_streams(self, source_list, data_path, is_speech=True, is_rir=False):
        '''
            utt_id_wav:list(source)+list(wav id)
            utt_data_wav:list(source)+list(wav)+list(time)+list(feat)
            utt_label_wav:list(source)+list(wav)+list(1)+list(label)
        '''
        utt_id_wav = []
        utt_data_wav = []
        utt_label_wav = []
        utt_aux_wav=[]
        label_paths = []
        aux_path= []
        for i in range(len(source_list)):
            corpus_type = source_list[i]['type']
            corpus_wav_path = data_path+source_list[i]['wav']            
            if 'label' in source_list[i]:
                label_paths.append(data_path+source_list[i]['label'])
            else:
                corpus_label_path = None
            if 'aux_label' in source_list[i]:
                aux_path.append(data_path+source_list[i]['aux_label'])
            else:
                corpus_label_path = None
            print("%s::_load_streams: loading %s from %s..." % (self.__class__.__name__, corpus_type, corpus_wav_path))
            utt_name, feat_data = zip(*read_mat_scp(corpus_wav_path))
            utt_id_wav.append(utt_name)
            utt_data_wav.append(feat_data)
        
        for i in range(len(label_paths)):
            given_utt_id = set(utt_id_wav[i])
            print(label_paths[i],"given utt:",len(utt_id_wav[i]))
            lines = my_cat(label_paths[i])
            lines.sort()        # each lines start with utterance ID, hence effectively sort the labels with utterance ID.
            curr_utt_id_label = [i.split(" ")[0] for i in lines]
            selected_utt_id = set(curr_utt_id_label) & given_utt_id
            print("select utt:",len(selected_utt_id))
            if len(selected_utt_id)!=len(given_utt_id):
                print(given_utt_id-selected_utt_id,"is not exist in label file")
            # selected_utt_id = set(curr_utt_id_label)
            selected_label_list = get_label(lines,selected_utt_id)
            utt_label_wav.append(selected_label_list)

        for i in range(len(aux_path)):
            given_utt_id = set(utt_id_wav[i])
            print(aux_path[i],"given aux utt:",len(utt_id_wav[i]))
            lines = my_cat(aux_path[i])
            lines.sort()        # each lines start with utterance ID, hence effectively sort the labels with utterance ID.
            curr_utt_id_label = [i.split(" ")[0] for i in lines]
            selected_utt_id = set(curr_utt_id_label) & given_utt_id
            print("select aux utt:",len(selected_utt_id))
            if len(selected_utt_id)!=len(given_utt_id):
                print(given_utt_id-selected_utt_id,"is not exist in label file")
            selected_label_list = get_label(lines,selected_utt_id)          
            utt_aux_wav.append(selected_label_list)

        return utt_id_wav,utt_data_wav,utt_label_wav,utt_aux_wav

    def __getitem__(self, index):

        return self.train_samples[index]

    def __len__(self):
        return len(self.train_samples)

    def __repr__(self):
        print("repr")
        return "fmt_str"


def get_label(label_lines, selected_utt_id):
    selected_label_list = []
    for line in label_lines:
        tmp = line.split(" ")
        utt_id = tmp[0]
        if utt_id in selected_utt_id:
            tmp_label = np.asarray([int(j) for j in tmp[1:] if len(j)>0])[np.newaxis,:]
            selected_label_list.append(tmp_label)
    return selected_label_list


