import numpy as np 
import torch
import yaml
from data.f_dataset import FeatDataSet 
from data.dataloader import ChunkDataloader
train_config = "./example/librispeech/configs/ce.yaml"
data_config = "./example/librispeech/configs/data.yaml"

if __name__ == "__main__":
    print('main test')
    with open(train_config) as f:
        config = yaml.safe_load(f)

    with open(data_config) as f:
        data = yaml.safe_load(f)
        config["source_paths"] = [j for i, j in data['clean_source'].items()]
        if 'dir_noise' in data:
            config["dir_noise_paths"] = [j for i, j in data['dir_noise'].items()]
        if 'rir' in data:
            config["rir_paths"] = [j for i, j in data['rir'].items()]
    config['data_path'] = ""
    
    print(config)
    trainset = FeatDataSet(config)
    train_dataloader = ChunkDataloader(trainset,
                                       batch_size=10,
                                       distributed=False,
                                       num_workers=1)
    for i, data in enumerate(train_dataloader, 0):
        feat = data["x"]
        label = data["y"]
        print(i,"times,x:",feat.size(),"y:",label.size())
