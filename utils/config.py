import os
import json
from omegaconf import OmegaConf

def save_config(config, path):
    with open(os.path.join(path, 'config.yaml'), 'w', encoding='utf-8') as file:
        OmegaConf.save(config=config, f=file.name)
    return

def load_config(filepath):
    return OmegaConf.load(os.path.join(filepath))

def load_config_json(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data

# def get_train_config(config_file):
#     if config_file is None:
#         config_file = 'default.json'
#     with open(os.path.join('configs', config_file)) as json_file:
#         data = json.load(json_file)
#     return data