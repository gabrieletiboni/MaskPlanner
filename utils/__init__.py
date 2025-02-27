from functools import wraps
import random
import socket
import string
import os
import numpy as np
from omegaconf import OmegaConf
import torch
import math
import wandb

from scipy.spatial.transform import Rotation as Rot

from utils.args import pformat_dict, to_dict
from utils.config import save_config

def get_random_string(n=5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

def get_output_dir(config):
    """Returns output_dir
        
        Priority:
            config.output_dir >
            $WORKDIR >
            './runs'
    """
    if config.output_dir is not None:
        return config.output_dir
    elif os.environ.get("WORKDIR") is not None:
        return os.environ.get("WORKDIR")
    else:
        # Default
        return 'runs'

def handle_cmap_input(cmap):
    if cmap == 'gt':
        # cmap = 'autumn'
        cmap = ['#D84315', '#F57C00', '#FFB300', '#FFEB3B']
    elif cmap == 'pred':
        # cmap = 'cool'
        # cmap = ['#00ACC1', '#03A9F4', '#42A5F5', '#7986CB']
        # cmap = ['#4CAF50', '#00BCD4', '#2196F3', '#673AB7']
        # cmap = ['#16A085', '#27AE60', '#2980B9', '#8E44AD']
        cmap = ['#2C3E50', '#16A085', '#27AE60', '#2980B9', '#8E44AD']   # official before 23.02.24

        ### 23.02.24
        # cmap='tab20'  # 'tab20', 'turbo', 'Dark2' (best thus far), 'Paired', 'hsv' (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
        # cmap = 'winter'

    return cmap

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def set_seed(seed):
    if seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

def create_dir(path):
    try:
        os.mkdir(os.path.join(path))
    except OSError as error:
        pass

def create_dirs(path):
    try:
        os.makedirs(os.path.join(path))
    except OSError as error:
        pass

def get_root_of_dir(dirname, roots):
    """Find root where dirname belongs to"""
    for root in roots:
        if os.path.isdir(os.path.join(root, dirname)):
            return root       
    return None


class FakeRot():
    """Mimics Rot object from scipy, to apply rotations in terms of normals
    (2D pose representation for points)"""
    def __init__(self, normals):
        self.normals = normals

    def apply(self, *args, **kwargs):
        return self.normals
        

def orient_in(extra_data):
    """Whether each output pose includes orientations.
    
    Returns the specific orient representation as well.
    """
    valid = ['orientquat', 'orientrotvec', 'orientnorm']
    for v in valid:
        if v in extra_data:
            return True, v
    
    return False, None


def rot_from_representation(orient_repr, arr):
    if orient_repr == 'orientquat':
        return Rot.from_quat(arr)
    elif orient_repr == 'orientrotvec':
        return Rot.from_rotvec(arr)
    elif orient_repr == 'orientnorm':
        return FakeRot(arr)

def new_run(f):
    @wraps(f)
    def new_f(*args, **kwargs):
        conf = args[0]

        if 'render' in conf:
            conf = OmegaConf.load(os.path.join(conf.render, 'config.yaml'))
            conf.only_render = True
            run_name = os.path.basename(conf.run_dir)
            save_dir = conf.run_dir
        else:
            conf.only_render = False
            random_str = get_random_string(5)
            set_seed(conf.seed)

            run_name = random_str+('_'+conf.name if conf.name is not None else '')+'-S'+str(conf.seed)
            output_dir = get_output_dir(conf)
            save_dir = os.path.join((output_dir if not conf.debug else 'debug_runs'), run_name)
            create_dirs(save_dir)
            conf.run_dir = save_dir
            save_config(conf, save_dir)



        print('\n ===== RUN NAME:', run_name, f' ({save_dir}) =====')
        print(pformat_dict(conf, indent=0))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")
        conf.device = str(device)

        if not conf.only_render:
            wandb.init(config=to_dict(conf),
                    project="dev-paintnet",
                    name=run_name,
                    group=conf.group,
                    save_code=True,
                    notes=conf.notes,
                    mode=('online' if not conf.debug else 'disabled'))
            
            wandb.config.path = save_dir
            wandb.config.hostname = socket.gethostname()
        
        try:
            ret = f(conf, *args[1:], **kwargs)
        except KeyboardInterrupt:
            pass
        if not conf.only_render:
            wandb.finish()

        return ret

    return new_f