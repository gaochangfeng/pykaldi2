import torch
from utils.dynamic_import import dynamic_import
#import importlib

def GenerateModel(model_config):
    print(model_config['name'])
    if 'name' not in model_config:
        raise ValueError("model_config do not has name")
    model_class = dynamic_import(model_config['name'])
    model_config.pop('name')
    model = model_class(**model_config)
    return model
