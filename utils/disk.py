import pdb
import pickle
import omegaconf
import os
import socket
import csv

import numpy as np
import point_cloud_utils as pcu
import torch
from scipy.spatial.transform import Rotation as Rot

from utils import orient_in


def get_dataset_downscale_factor(category):
    """Returns precomputed average max vertex distance
    over given category's train set."""
    mean_max_distance = {
            'containers-v2': 884.1423249856435,
            'cuboids-v1': 888.7967305471634,
            'cuboids-v2': 889.6556509728579,
            'cuboids-small-v2': 885.8284752276212,
            'cuboids-discrete-v1': 881.05007396,
            'cuboids-discrete-xfixed-v1': 873.877203026212,
            'cuboids-large-v1': 888.0597387021147,
            'shelves-v1': 905.4091900499023,
            'shelves-v2': 424.2046732264433,
            'cuboids-v1-windows-v1-shelves-v1': 947.2448614376127,   # Joint-category training
            'windows-v1-shelves-v1-containers-v2': 969.337674913636,  # Joint-category training
            'cuboids-v1-shelves-v1-containers-v2': 895.6137144950626,  # Joint-category training
            'cuboids-v1-windows-v1-containers-v2': 961.3291445923128,  # Joint-category training
            'cuboids-v1-windows-v1-shelves-v1-containers-v2': 940.7008946944458,  # Joint-category training
            'cuboids-v2-windows-v2-shelves-v2-containers-v2': 779.2320060197117,  # Joint-category training
            'cuboids-v2-windows-v2-shelves-v2': 776.1721217165386,  # Joint-category training
            'windows-v1': 1157.9744613449216,
            'windows-v2': 1014.656040950315,
            'realtime_windows-v1': 1027.2274259059286
    }
    if category not in mean_max_distance:
        return None
    else:
        return mean_max_distance[category]


def get_auxiliary_pretrained_custom_path(dataset, version : int):
    """Return path of auxiliary pretrained model for SoPs prediction
        and object features
    """
    category = get_dataset_name(dataset)

    if version == 1:
        pretrained_custom_paths = {
                'cuboids-v2': 'pretrained_models/4T4BI-S42',
                'windows-v2': 'pretrained_models/0WCLK-S42',
                'shelves-v2': 'pretrained_models/TZOV8-S42',
                'containers-v2': 'pretrained_models/CN000-S42'
        }
    elif version == 2:
        pretrained_custom_paths = {
                'cuboids-v2': 'pretrained_models/sop_cuboids-v2/36FNJ-S46',
                'windows-v2': 'pretrained_models/sop_windows-v2/I66C6-S48',
                'shelves-v2': 'pretrained_models/sop_shelves-v2/39LI2-S46',
                'containers-v2': 'pretrained_models/sop_containers/VDGKO-S47',
                'realtime_windows-v1': 'pretrained_models/sop_windows-v2/I66C6-S48'
        }

    return pretrained_custom_paths[category]


def get_paint_feedback_root(category):
    return os.environ.get(f"PAINT_FEEDBACK_ROOT")


def get_dataset_name(dataset):
    """Return a single string even for datasets that are
        defined as list of multiple categories
    """
    # assert isinstance(dataset, list) or isinstance(dataset, omegaconf.listconfig.ListConfig), 'config.dataset is expected to be of type list.'  # Handles joint category training
    if not isinstance(dataset, list) and not isinstance(dataset, omegaconf.listconfig.ListConfig):
        dataset = list(dataset)
    return '-'.join(dataset)


def get_dataset_root():
    """Returns PaintNet root according to env variable"""
    assert os.environ.get("PAINTNET_ROOT") is not None, "Set PAINTNET_ROOT environment variable to localize the paintnet dataset root."
    assert os.path.isdir(os.environ.get("PAINTNET_ROOT")), f'Dataset root path was set but does not exist on current system. Path: {os.environ.get("PAINTNET_ROOT")}'
    return os.environ.get("PAINTNET_ROOT")


def get_dataset_meshes_path(category):
    """Returns dir with meshes of corresponding dataset.
        
        Mainly used to render results locally, without
        having to store the full dataset.
    """
    PAINTNET_ROOT = get_dataset_root()
    if 'gabriele-hp-pavilion' in socket.gethostname():
        suffixes = ['', '_meshonly']
        
        for suffix in suffixes:
            if os.path.isdir(os.path.join(PAINTNET_ROOT, f'{category}', '{category}{suffix}')):
                return os.path.join(PAINTNET_ROOT, f'{category}', f'{category}{suffix}')

        raise ValueError("No local dataset found for category " + str(category))

    else:
        assert os.path.join(PAINTNET_ROOT, category), f'Current dataset category {category} does not exist on your system.'
        return os.path.join(PAINTNET_ROOT, category)


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


def get_test_results_save_dir_name(config, cli_args):
    """Save test results on a separate directory to avoid
        overwriting the training results
    """
    target_suffix = '' if cli_args.target is None else '_'+str(cli_args.target)

    if cli_args.model not in ['best', 'last'] or cli_args.target is not None:
        return os.path.join(cli_args.run, 'test', f'{cli_args.model}{target_suffix}')
    else:
        return os.path.join(cli_args.run, 'test')


def get_dataset_path(category):
    """Returns dir path where files are stored.

    category : str
               e.g. cuboids-v1, windows-v1, shelves-v1, containers-v2
    """
    PAINTNET_ROOT = get_dataset_root()
    if 'gabriele-hp-pavilion' in socket.gethostname():
        suffixes = ['', '_small']

        for suffix in suffixes:
            if os.path.isdir(os.path.join(PAINTNET_ROOT, f'{category}/{category}{suffix}')):
                return os.path.join(PAINTNET_ROOT, f'{category}/{category}{suffix}')

        raise ValueError(f'No local dataset found for category {category}. Full path searched for is: {os.path.join(PAINTNET_ROOT, f"{category}/{category}[suffix]/")}') 
    else:
        assert os.path.join(PAINTNET_ROOT, category), f'Current dataset category {category} does not exist on your system.'
        return os.path.join(PAINTNET_ROOT, category)


def get_dataset_paths(categories):
    """Same as get_dataset_path, but handles multiple categories for joint training"""
    if isinstance(categories, list) or isinstance(categories, omegaconf.listconfig.ListConfig):
        roots = []
        for cat in categories:
            roots.append(get_dataset_path(cat))
        return roots
    else:
        return [get_dataset_path(categories)]


def get_dataset_meshes_paths(categories):
    """Same as get_dataset_meshes_path, but handles multiple categories for joint training"""
    if isinstance(categories, list) or isinstance(categories, omegaconf.listconfig.ListConfig):
        roots = []
        for cat in categories:
            roots.append(get_dataset_meshes_path(cat))
        return roots
    else:
        return [get_dataset_meshes_path(categories)]
    

def read_traj_file(filename, extra_data=[], weight_orient=1.):
    """Returns trajectory as nd-array (T, <3,6>)
    given traj filename in .txt"""
    points = []
    stroke_ids = []
    stroke_id_index = 6
    cols_to_read = [0, 1, 2]
    orientations, orient_repr = orient_in(extra_data)
    if orientations:
        cols_to_read += [3, 4, 5]

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
    
        for cols in reader:
            cols_float = np.array(cols, dtype='float64')[cols_to_read]
            stroke_id = int(np.array(cols, dtype='float64')[stroke_id_index])

            if orientations:
                if orient_repr == 'orientquat':
                    quats = weight_orient*Rot.from_euler('yzx', [cols_float[4], cols_float[5], cols_float[3]], degrees=True).as_quat()
                    points.append(np.concatenate((cols_float[:3], quats)))
                elif orient_repr == 'orientrotvec':
                    rotvec = weight_orient*Rot.from_euler('yzx', [cols_float[4], cols_float[5], cols_float[3]], degrees=True).as_rotvec()
                    points.append(np.concatenate((cols_float[:3], rotvec)))
                elif orient_repr == 'orientnorm':
                    rot = Rot.from_euler('yzx', [cols_float[4], cols_float[5], cols_float[3]], degrees=True)
                    e1 = np.array([1,0,0])
                    normals = weight_orient*rot.apply(e1)
                    points.append(np.concatenate((cols_float[:3], normals)))
            else:
                points.append(cols_float)

            stroke_ids.append(stroke_id)

    return np.array(points), np.array(stroke_ids)


def load_stroke_npy(filename, extra_data=[], weight_orient=1.):
    stroke = np.load(filename)
    orientations, orient_repr = orient_in(extra_data)
    stroke_orient = stroke[:, [4, 5, 3]]
    stroke = stroke[:, :3]
    if orientations:
        if orient_repr == 'orientquat':
            stroke_orient = weight_orient*Rot.from_euler('yzx', stroke_orient, degrees=True).as_quat()
        elif orient_repr == 'orientrotvec':
            stroke_orient = weight_orient*Rot.from_euler('yzx', stroke_orient, degrees=True).as_rotvec()
        elif orient_repr == 'orientnorm':
            rot = Rot.from_euler('yzx', stroke_orient, degrees=True)
            e1 = np.array([1,0,0])
            stroke_orient = weight_orient*rot.apply(e1)
        stroke = np.concatenate((stroke, stroke_orient), axis=1)
    return stroke


def save_traj_file(traj, filepath, kind='normals'):
    """Save trajectory as k-dim sequence of vectors,
    with k=3 (x,y,z), or k=6 (+ orientations), or k=7 (+ strokeId)

        traj : (N, k) array, 
        kind : str
                 'normals': orientations dims are interpreted as a 3D normal vector
                 'euler': orientations dims are interpreted as Euler angles
    """
    assert traj.ndim == 2 and (traj.shape[-1]==3 or traj.shape[-1]==6 or traj.shape[-1]==7), f"Trajectory is not formatted correctly: {traj.shape} - {traj}"
    assert kind in {'normals', 'euler'}

    if torch.is_tensor(traj):
        traj = traj.cpu().detach().numpy()
    
    k = traj.shape[-1]

    if kind == 'normals':
        header = ['X','Y','Z','W1','W2','W3','strokeId']
        if k > 3:
            assert np.allclose(np.linalg.norm(traj[:, 3:6], axis=-1), 1, atol=0.02)  # sanity check on normals being normals

    elif kind == 'euler':    
        header = ['X','Y','Z','A','B','C','strokeId']

    header = header[:k]
    with open(os.path.join(filepath), 'w', encoding='utf-8') as file:
        print(";".join(header), file=file)
        for cols in traj:
            print(";".join(map(str, cols)), file=file)
    return


def read_mesh_as_pointcloud(filename, return_more=False):
    v, f = pcu.load_mesh_vf(os.path.join(filename))

    if return_more:
        centroid = np.mean(v, axis=0)
        v_centered = v - centroid
        max_distance = np.max(np.sqrt(np.sum(v_centered ** 2, axis=1)))

    f_i, bc = pcu.sample_mesh_poisson_disk(v, f, 10000, 0.5)  # Num of points (not guaranteed), radius for poisson sampling
    points = pcu.interpolate_barycentric_coords(f, f_i, bc, v)

    if return_more:
        return points, centroid, max_distance
    
    return points


def load_object(filepath):
    return pickle.load(open(filepath, 'rb'))


def save_object(obj, save_dir, filename):
    with open(os.path.join(save_dir, f'{filename}.pkl'), 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)