"""Argument parser

    Priority of arguments:
        CLI parameters > <config-file>.yaml > default.yaml

    Examples:
        python script.py config=[conf1,conf2]  # conf1.yaml must exist in CONFIG_PATH/
"""
import os
import psutil
# from collections import Mapping
from collections.abc import Mapping
from omegaconf import OmegaConf, ListConfig

CONFIG_PATH = 'configs'
DEFAULT_CONFIG = 'default.yaml'
PARAMS_AS_LIST = ['exp', 'loss', 'eval_metrics', 'extra_data', 'augmentations', 'dataset']  # Params to interpret as lists

def add_extension(config_file):
    assert type(config_file) == str
    filename, _ = os.path.splitext(config_file)  # Returns filename and extension
    return filename+".yaml"

def pformat_dict(d, indent=0):
    fstr = ""
    for key, value in d.items():
        fstr += '\n' + '  ' * indent + str(key) + ":"
        if isinstance(value, Mapping):
            fstr += pformat_dict(value, indent+1)
        else:
            fstr += ' ' + str(value)
    return fstr

def to_dict(args):
    return OmegaConf.to_container(args)

def as_list(arg):
    if isinstance(arg, str):
        return [arg]
    elif "_content" in dir(arg) and isinstance(arg._content, list):
        return arg
    elif isinstance(arg, list) or isinstance(arg, ListConfig):
        return arg
    else:
        raise ValueError()

def pars_as_list(args, keys):
    for key in keys:
        try:
            if key in args:
                args[key] = as_list(args[key])
            else:
                print(f"Warning! This parameter was not found in config: {key}")
        except ValueError:
            print(f"Warning! This parameter was neither a string nor a list: {key}={args[key]}")
    return args


def load_args(root = None):
    if root is None:
        conf_path = os.path.join(CONFIG_PATH)
    else:
        conf_path = os.path.join(root)  # Config path

    conf_args = {}
    cli_args = OmegaConf.from_cli()  # Read the cli args

    if 'cpu' in cli_args:
        c_start = cli_args.cpu[0]
        c_end = cli_args.cpu[1] + 1
        p = psutil.Process()
        p.cpu_affinity(list(range(c_start, c_end)))

    auto_wandb_group=''

    if 'config' in cli_args and cli_args.config:  # Read configs (potentially as a list of configs)
        cli_args.config = [cli_args.config] if type(cli_args.config) == str else cli_args.config
        for i in range(len(cli_args.config)):
            config_name = cli_args.config[i]
            
            # Handle alias configurations
            if is_alias(config_name):
                list_of_configs = from_alias_to_configs(config_name)
                for config_name in list_of_configs:
                    auto_wandb_group += config_name[0].upper()+config_name[1:]+'_'
                    config_name = add_extension(config_name)  # Add .yaml extension if not present
                    file_args = OmegaConf.load(os.path.join(conf_path, config_name))
                    conf_args = OmegaConf.merge(conf_args, file_args)

            else:
                auto_wandb_group += config_name[0].upper()+config_name[1:]+'_'
                config_name = add_extension(config_name)  # Add .yaml extension if not present
                file_args = OmegaConf.load(os.path.join(conf_path, config_name))
                conf_args = OmegaConf.merge(conf_args, file_args)


    conf_args = OmegaConf.merge(conf_args, cli_args)  # Merge cli args into config ones (potentially overwriting config args)

    conf_args['auto_wandb_group'] = auto_wandb_group[:-1]

    # check if default is specified in the config file
    if ('default' not in conf_args or conf_args.default) and 'render' not in conf_args:
        print("Using default config file")
        default_args = OmegaConf.load(os.path.join(conf_path, DEFAULT_CONFIG))  # Default config file
        conf_args = OmegaConf.merge(default_args, conf_args) 
    conf_args = pars_as_list(conf_args, PARAMS_AS_LIST)  # Convert parameters to lists if they are strings
    # args = OmegaConf.to_object(args)  # Recursively convert your OmegaConf object to a plain python object (ListConfig to python list) -> args.dataset would no longer work, in favor of args['dataset']

    ### HARD CONSTRAINTS on args
    return conf_args

# args = load_args(os.path.dirname(os.path.dirname(__file__))) # Retrocompatibility



def is_alias(config_name):
    return config_name in config_aliases()

def from_alias_to_configs(config_name):
    assert is_alias(config_name)
    return list(config_aliases()[config_name])

def config_aliases():
    """Aliases for configuration files
        
        Each alias should point to more than a single config file,
        otherwise there's not really a point in using aliases.
    """
    aliases = {
        'maskplanner': ['asymm_chamfer_v9', 'delayMasksLoss', 'traj_sampling_v2', 'sched_v9'],
        'segmentWise': ['stable_v1', 'delayMasksLoss', 'traj_sampling_v2', 'sched_v9'],
        'pointWise':   ['lambda1', 'delayMasksLoss', 'traj_sampling_v2', 'sched_v9']
    }

    return aliases