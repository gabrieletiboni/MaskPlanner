import pdb
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import point_cloud_utils as pcu
import torch
from scipy.interpolate import Akima1DInterpolator

from . import orient_in
from .disk import get_dataset_downscale_factor


def get_max_distance(meshpath):
    """Returns max distance from mean of given mesh filename"""
    v, f = pcu.load_mesh_vf(os.path.join(meshpath))
    centroid = np.mean(v, axis=0)
    v = v - centroid
    m = np.max(np.sqrt(np.sum(v ** 2, axis=1)))
    return m


def get_mean_mesh(meshpath):
    v, f = pcu.load_mesh_vf(os.path.join(meshpath))
    centroid = np.mean(v, axis=0)
    return centroid


def center_pair(point_cloud, traj, meshpath, centroid=None):
    assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 3
    assert centroid is not None or meshpath is not None
    if meshpath is not None:
        centroid = get_mean_mesh(meshpath) # np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    traj[:, :3] -= centroid
    return point_cloud, traj


def center_traj(traj, meshpath, centroid=None):
    if meshpath is not None:
        centroid = get_mean_mesh(meshpath) # np.mean(point_cloud, axis=0)
    traj[:, :3] -= centroid
    return traj


def denormalize_traj(traj, meshpath, config, normalization='per-dataset', dataset=None, custom_data_scale_factor=None):
    """Transform traj back to its original space.
        Used for saving predictions or ground truth in the format expected
        by the offline simulator.
        
        In practice: scale, shift, and adjust orientation normals.
    """
    assert normalization=='per-dataset', 'Not yet implemented for per-mesh normalization.'
    assert traj.ndim == 2 and traj.shape[-1] == 6, 'traj is expected to be (N, 6) array'
    assert 'orientnorm' in config['extra_data'], 'Not yet implemented for other orientation types'

    if normalization == 'per-dataset':
        assert dataset is not None

    centroid = get_mean_mesh(meshpath)

    if custom_data_scale_factor is not None:
        scale = custom_data_scale_factor
    else:
        scale = get_dataset_downscale_factor(dataset)

    traj[:, :3] *= scale  # re-scale
    traj[:, :3] += centroid  # shift

    traj[:, 3:6] /= config['weight_orient']  # orientation normals back to norm = 1

    return traj


def normalize_pc(pc):
    """Normalizes point-cloud such that furthest point 
    from origin is equal to 1 and mean is centered
    around zero
    
    pc : (N, 3) array

    returns (N, 3) array
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def is_padded(traj):
    return np.any(np.where(((traj[:,0] == -100) & (traj[:,1] == -100) & (traj[:,2] == -100))))

def is_padded_v2(traj):
    raise NotImplementedError()

def add_padding(traj, traj_points, lmbda, overlapping=0, extra_data=[]):
    # assert traj.shape[-1] == get_dim_traj_points(extra_data)*lmbda, f'traj shape: {traj.shape} vs. expected shape: N,{get_dim_traj_points(extra_data)*lmbda}'
    if overlapping == 0:
        num_fake_points = ((traj_points//lmbda) - traj.shape[0])
    else:
        max_subsequences = (traj_points-lmbda)//(lmbda-overlapping) + 1
        num_fake_points = (max_subsequences - traj.shape[0])
    return np.pad(traj, pad_width=((0, num_fake_points),(0,0)), constant_values=-100)  # Pad last values with -100


def remove_padding(traj, extra_data=[]):
    assert (traj.ndim == 2 or traj.ndim == 3) and traj.shape[-1] == get_dim_traj_points(extra_data), f'Make sure to reshape the traj correctly before removing padding. ndim:{traj.ndim} | shape:{traj.shape}'
    if is_padded(traj):  # Check if it's actually padded
        first_padding_index = np.where( ((traj[:,0] == -100) & (traj[:,1] == -100) & (traj[:,2] == -100)) )[0][0]
        traj = traj[:first_padding_index, :].copy()
    
    return traj

def remove_padding_v2(traj, stroke_ids):
    """v2: it can filter out vectors that are positioned anywhere
        in the trajectory, not just at the end.
        NOTE: this requires changing the stroke_ids vector too.
    """    
    assert stroke_ids.shape[:] == traj.shape[:-1]

    fake_mask = np.all((traj == -100), axis=-1)  # True for fake vectors

    stroke_ids = stroke_ids[~fake_mask].copy()
    traj = traj[~fake_mask].copy()

    return traj, stroke_ids


def from_seq_to_pc(traj, extra_data, stroke_ids=None):
    """From lambda-sequences (or strokes) to point-cloud of poses
    
    traj: (n_sequences, lambda_points*outdim) 
    """
    assert traj.ndim == 2, traj.ndim
    expected_outdim = get_dim_traj_points(extra_data)

    # Skip if traj is already in point-wise format
    if traj.shape[-1] == expected_outdim:
        if stroke_ids is not None:
            return traj, stroke_ids
        return traj

    lambda_points = int(traj.shape[-1]//expected_outdim)

    traj = traj.reshape(-1, expected_outdim)
    traj = remove_padding(traj, extra_data)

    if stroke_ids is not None:
        stroke_ids = stroke_ids[:traj.shape[0]//lambda_points, None] # remove padding from curr_stroke_ids
        stroke_ids = np.repeat(stroke_ids, lambda_points) # curr_stroke_ids from sequence to point
        return traj, stroke_ids
    return traj


def from_seq_to_pc_v2(traj, stroke_ids, config):
    """From lambda-sequences (or strokes) to point-cloud of poses
        v2:
            - config instead of extra_data
            - lambda_points is not inferred, to prevent errors.
            - stroke_ids is mandatory to prevent errors.
            - remove_padding_v2 is used.
    traj: (n_sequences, lambda_points*outdim) 
    """
    assert traj.ndim == 2, traj.ndim
    extra_data, lambda_points = config['extra_data'], config['lambda_points']
    outdim = get_dim_traj_points(extra_data)

    # Skip if traj is already in point-wise format
    if traj.shape[-1] == outdim:
        return traj, stroke_ids

    traj, stroke_ids = remove_padding_v2(traj, stroke_ids=stroke_ids)
    traj = traj.reshape(-1, outdim)

    stroke_ids = stroke_ids[:traj.shape[0]//lambda_points, None] # remove padding from curr_stroke_ids (probably not needed)
    stroke_ids = np.repeat(stroke_ids, lambda_points) # curr_stroke_ids from sequence to point

    return traj, stroke_ids


def from_pc_to_seq(traj, traj_points, lambda_points, overlapping, extra_data, padding=True):
    """From point-cloud of poses to lambda-sequences (or strokes)"""
    expected_outdim = get_dim_traj_points(extra_data)
    assert traj.ndim == 2 and traj.shape[-1] == expected_outdim

    traj = traj.reshape(-1, expected_outdim*lambda_points)
    if padding:
        traj = add_padding(traj, traj_points=traj_points, lmbda=lambda_points, overlapping=overlapping, extra_data=extra_data)
    return traj


def resample_strokes_at_equal_spaced_points(traj, stroke_ids, distance, interpolate=True, equal_in_3d_space=False):
    """Given the input points, interpolate them and re-sample the signal
        with dynamic time intervals such that consecutive points are
        equally spaced by `distance`

        equal_in_3d_space : bool,
                            if True, distance computed w.r.t. (x,y,z) only, without normal
    """
    assert traj.ndim == 2
    assert stroke_ids.ndim == 1

    N, D = traj.shape

    out_traj = np.empty((0, D))
    out_stroke_ids = np.empty((0))

    for i in np.unique(stroke_ids):
        if i == -1:  # -1 is assigned to fake vectors 
            continue

        curr_stroke = traj[stroke_ids == i]
        
        if interpolate:
            # interpolate input points
            curr_stroke_sampled = resample_at_equal_spaced_points(curr_stroke, distance, equal_in_3d_space=equal_in_3d_space)
        else:
            # return a subsample of the input points
            curr_stroke_sampled = subsample_at_equal_spaced_points(curr_stroke, min_distance=distance, equal_in_3d_space=equal_in_3d_space)

        curr_stroke_ids = np.repeat(i, curr_stroke_sampled.shape[0])

        out_traj = np.append(out_traj, curr_stroke_sampled, axis=0)
        out_stroke_ids = np.append(out_stroke_ids, curr_stroke_ids, axis=0)

    return out_traj, out_stroke_ids


def subsample_at_equal_spaced_points(stroke, min_distance, equal_in_3d_space=False):
    """As in resample_at_equal_spaced_points, but no interpolation is done.
        Therefore, a subset of the input points in stroke is returned, such that
        distance among points is AT LEAST `min_distance`
    """
    assert stroke.ndim == 2
    N, D = stroke.shape

    out_stroke = np.empty((0, D))

    last_point = stroke[0, :].copy()
    out_stroke = np.append(out_stroke, last_point[None, :], axis=0)
    for i, point in enumerate(stroke[1:]):

        if equal_in_3d_space:
            distance = np.linalg.norm(point[:3] - last_point[:3])
        else:
            distance = np.linalg.norm(point - last_point)

        if distance > min_distance:
            last_point = point.copy()
            out_stroke = np.append(out_stroke, last_point[None, :], axis=0)

    return out_stroke


def resample_at_equal_spaced_points(stroke, distance, equal_in_3d_space=False):
    """Given the input points of a single stroke, interpolate them and re-sample
       the signal with dynamic time intervals such that consecutive points
       are equally spaced.
    """
    assert stroke.ndim == 2
    N, D = stroke.shape

    if equal_in_3d_space:
        raise NotImplementedError()

    out_stroke = np.empty((0, D))

    times = np.arange(0, N)
    curve = Akima1DInterpolator(times, stroke)

    dt = 0.2  # discrete time interval to check whether a new point should be sampled
    last_point = stroke[0, :].copy()  # curve(times[0])
    out_stroke = np.append(out_stroke, last_point[None, :], axis=0)
    curr_t = 0.1
    while True:
        # Exit when the stroke finishes
        if curr_t >= times[-1]:
            break

        curr_point = curve(curr_t)

        d = np.linalg.norm(curr_point - last_point)
        if d > distance:
            last_point = curr_point.copy()
            out_stroke = np.append(out_stroke, last_point[None, :], axis=0)

        curr_t += dt

    return out_stroke


def get_sequences_of_lambda_points(traj, stroke_ids, lmbda, dirname, overlapping=0, extra_data=[], padding=True):
    """Merge consecutive points in traj in a single sequence of lmbda points

    Input:
        traj: (N, 3)
        stroke_ids (N,)
        lmbda: int
        overlapping : int
    Output:
        new_traj: (~(N//lmbda), lmbda)
                  first size is not constant, due to multiple strokes in the same traj.
                  consecutive points are per-stroke, so not all consecutive sequences can be made.
                  In general, new_traj.shape[0] <= N//lmbda if overlapping = 0. Otherwise,
                  the maximum number of sub-sequences is (N-lmbda)/(lmbda-overlapping) + 1 
    """
    outdim = get_dim_traj_points(extra_data)
    assert traj.ndim == 2 and traj.shape[-1] == outdim
    
    N, D = traj.shape

    n_strokes = int(stroke_ids[-1]+1)
    count = 0
    new_stroke_count = 0
    first_time = True
    warning_skipped_strokes = 0

    start_idx = 0
    for stroke_id in range(n_strokes):
        if stroke_id == n_strokes-1:  # if last one
            end_idx = N - 1
        else:
            end_idx = np.argmax(stroke_ids == (stroke_id+1)) - 1   # index of last point in current stroke (search for next stroke's starting point)

        stroke_length = end_idx+1 - start_idx
        curr_stroke = traj[start_idx:start_idx+stroke_length]

        if stroke_length >= lmbda:
            """Index of starting sequences (last idx is used as :last_idx to get final point of sequence)
            (stroke_length+1 because <stop> is never included, this way we can potentially include the index of point in the next stroke to use as :ar[-1])
            """

            if overlapping == 0:
                ar = np.arange(0, stroke_length+1, step=lmbda)
                
                remainder = (stroke_length%lmbda)
                centered_stroke = curr_stroke[(remainder//2) : ar[-1] + (remainder//2)]  # Pad this stroke
                new_traj_piece = centered_stroke.reshape((-1, lmbda*outdim))  # Join lambda points in the same dimension


            else:  # Handle mini-sequence that overlap
                overlapped_repetitions = int(  (stroke_length - lmbda) / (lmbda - overlapping)  )
                assert  int(  (stroke_length - lmbda) / (lmbda - overlapping)  ) ==  (stroke_length - lmbda) // (lmbda - overlapping)
                
                eff_length = overlapped_repetitions*(lmbda - overlapping) + lmbda
                remainder = stroke_length%eff_length

                ol_length = lmbda - overlapping  # overlapped_length

                new_traj_piece = np.array([ curr_stroke[(i*ol_length):(i*ol_length)+lmbda] for i in range(overlapped_repetitions+1) ])  # +1 for the starting non-overlapped sequence on the left, (overlapped_repetitions+1, lambda, outdim)
                assert new_traj_piece.ndim == 3
                new_traj_piece = new_traj_piece.reshape((overlapped_repetitions+1), lmbda*outdim)  # (overlapped_repetitions+1, lmbda*outdim)


            if first_time:
                new_traj = new_traj_piece.copy()
                new_stroke_ids = np.ones((new_traj_piece.shape[0]))*(new_stroke_count)
                first_time = False
            else:
                new_traj = np.append(new_traj, new_traj_piece, axis=-2)
                new_stroke_ids = np.append(new_stroke_ids, np.ones((new_traj_piece.shape[0]))*(new_stroke_count))

            new_stroke_count += 1

        else:
            # Cannot make sequences of points longer than the points present in this stroke
            # print(f'Warning! A stroke has been ignored and discarded due to a <lambda_points> value' \
            #       f'being higher than this stroke length. Lambda:{lmbda} | Stroke_length:{stroke_length}' \
            #       f'| Stroke_id:{stroke_id} | Dirname:{dirname}')
            warning_skipped_strokes += 1

        # print('\n\n=========')
        # print('Lmabda:', lmbda, '\n')
        # print('start_idx | end_idx:', start_idx, end_idx)
        # print('stroke length:', stroke_length)
        # print('Expected number of sequences:', stroke_length//lmbda)
        # print('Remainder:', remainder)
        # print('padded stroke length:', padded_stroke.shape)
        # print()
        # print('First sequence:\n', padded_stroke[0,:].reshape(-1, outdim))
        # print('Last sequence:\n', padded_stroke[-1,:].reshape(-1, outdim))

        start_idx = end_idx+1
        count += 1

    # print('=======\n======\n======')
    # print('Number of strokes handled:', count)
    # print('Final new traj:', new_traj.shape)
    # print('Final new stroke ids:', new_stroke_ids.shape)

    # print('new_traj.shape[0] | N//lmbda:', new_traj.shape[0], (N//lmbda))
    # print('new_traj.shape[0] <= N//lmbda --->', new_traj.shape[0] <= N//lmbda)
    # unique, counts = np.unique(new_stroke_ids, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    if overlapping == 0:
        assert new_traj.shape[0] <= N//lmbda
    else:
        assert new_traj.shape[0] <= (N-lmbda)//(lmbda-overlapping) + 1 
    assert count == n_strokes
    assert new_traj.shape[-1] == (lmbda*outdim)

    # Pad last values with -100
    if padding:
        new_traj = add_padding(new_traj, N, lmbda, overlapping, extra_data=extra_data)
        new_stroke_ids = np.append(new_stroke_ids, -1*np.ones(new_traj.shape[0] - new_stroke_ids.shape[0]))  # pad stroke ids too

    if warning_skipped_strokes > 0:
        print(f'Warning! Skipped {warning_skipped_strokes} strokes in {dirname} as having length < {lmbda}')

    return new_traj, new_stroke_ids


def reshape_stroke_to_segments(stroke, lambda_points, overlapping):
    assert stroke.ndim == 2
    outdim = stroke.shape[-1]
    if overlapping == 0:
        new_stroke = stroke[:stroke.shape[0]//lambda_points*lambda_points].copy().reshape(-1, lambda_points, outdim)
    else:
        new_stroke = np.lib.stride_tricks.sliding_window_view(stroke, lambda_points, axis=0)[::(lambda_points-overlapping), :].copy()
        new_stroke = np.transpose(new_stroke, (0, 2, 1))

    return new_stroke


def get_traj_feature_index(feat, extra_data):
    if feat == None:
        return None

    if len(extra_data) == 0:
        indexes = {
            'pos': [0, 1, 2],
            'vel': None,
            'orientquat': None,
            'orientrotvec': None,
            'orientnorm': None
        }
    elif 'vel' in extra_data and len(extra_data) == 1:  # Vel only
        indexes = {
            'pos': [0, 1, 2],
            'vel': [3, 4, 5],
            'orientquat': None,
            'orientrotvec': None,
            'orientnorm': None
        }
    elif 'orientquat' in extra_data and len(extra_data) == 1:  # Orient only
        indexes = {
            'pos': [0, 1, 2],
            'vel': None,
            'orientquat': [3, 4, 5, 6],
            'orientrotvec': None,
            'orientnorm': None
        }
    elif 'orientrotvec' in extra_data and len(extra_data) == 1:  # Orient only
        indexes = {
            'pos': [0, 1, 2],
            'vel': None,
            'orientquat': None,
            'orientrotvec': [3, 4, 5],
            'orientnorm': None
        }
    elif 'orientnorm' in extra_data and len(extra_data) == 1:  # Orient only
        indexes = {
            'pos': [0, 1, 2],
            'vel': None,
            'orientquat': None,
            'orientrotvec': None,
            'orientnorm': [3, 4, 5]
        }
    else:
        raise ValueError('Other combinations of extra_data are not supported yet.')

    return indexes[feat]


def get_dim_traj_points(extra_data):
    """Returns dimensionality of each output pose"""
    if len(extra_data) == 0:
        return 3
    elif 'vel' in extra_data and len(extra_data) == 1:  # Vel only
        return 6
    elif 'orientquat' in extra_data and len(extra_data) == 1:  # Orient only
        return 7
    elif 'orientrotvec' in extra_data and len(extra_data) == 1: # Orient only
        return 6
    elif 'orientnorm' in extra_data and len(extra_data) == 1: # Orient only
        return 6
    else:
        raise ValueError('Other combinations of extra_data are not supported yet.')


def get_dim_orient_traj_points(extra_data):
    """Returns dimensionality of current orientation representation"""
    if not orient_in(extra_data)[0]:
        return 0

    dims = {
        'orientquat': 4,
        'orientnorm': 3,
        'orientrotvec': 3
    }
    for k, v in dims.items():
        if k in extra_data:
            return dims[k]
    raise ValueError(f'Unexpected error: code flow should not get here. Inspect it otherwise. extra_data: {extra_data}')


def get_velocities(traj, stroke_ids):
    """Returns per-point translational velocities.
    The last point of each stroke has zero velocity"""
    vels = np.zeros((traj.shape))

    vels[:-1,:] = (traj[1:, :] - traj[:-1, :]) # / 0.004*100 

    n_strokes = stroke_ids[-1]+1
    for stroke_id in range(1, n_strokes):  # Set to zero velocities at stroke changes
        ending_index = np.argmax(stroke_ids == stroke_id) - 1   # index of last point in current stroke
        vels[ending_index] = 0
    return vels


def downsample_strokes(traj, stroke_ids, stroke_points):
    """Downsample each stroke to stroke_points"""
    new_traj = []
    new_stroke_ids = []

    valid_strokes = np.where(np.unique(stroke_ids, return_counts=True)[1] > stroke_points)[0]  # filter strokes shorter than stroke_points

    c = 0
    for i in valid_strokes:
        curr_length = stroke_ids[stroke_ids == i].shape[0]
        starting_index = np.argmax(stroke_ids == i)

        choice = np.round_(np.linspace(0, (curr_length-1), num=stroke_points)).astype(int)
        choice += starting_index

        new_traj.append(np.copy(traj[choice, :]))
        new_stroke_ids.append(np.ones(choice.shape[0])*c)
        # new_stroke_ids.append(np.copy(stroke_ids[choice]))
        # stroke_ids = stroke_ids[choice]

        c += 1

    new_traj = np.array(new_traj)
    new_stroke_ids = np.array(new_stroke_ids)

    return new_traj, new_stroke_ids


def get_3dbbox(points):
    """Returns (non-minimum volume / non-oriented) 3D bounding box"""
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(points[:, 0]), np.max(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 1]), np.min(points[:, 2]), np.max(points[:, 2])

    return xmin, xmax, ymin, ymax, zmin, zmax

def get_center_of_3dbbox(box):
    xmin, xmax, ymin, ymax, zmin, zmax = box
    return [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]


def get_sizes_of_3dbbox(box):
    xmin, xmax, ymin, ymax, zmin, zmax = box
    return [xmax-xmin, ymax-ymin, zmax-zmin]


def from_bbox_encoding_to_visual_format(encoded_bbox):
    """From 3D bounding box encoded as (center, width, height, depth)
    to (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    x, y, z, w, h, d = encoded_bbox

    xmin = x - w/2
    xmax = x + w/2

    ymin = y - h/2
    ymax = y + h/2

    zmin = z - d/2
    zmax = z + d/2

    return xmin, xmax, ymin, ymax, zmin, zmax


def mean_knn_distance(point_cloud, k=2, render=False, y_lengths=None):
    """Visualization function for computing k-NNs of point-cloud"""
    if point_cloud.ndim == 2:
        point_cloud = point_cloud[np.newaxis, :, :]

    if not torch.is_tensor(point_cloud):
        point_cloud = torch.tensor(point_cloud)

    B, _, _ = point_cloud.size()

    """
        mean k-nn distance histogram
    """
    distances = torch.cdist(point_cloud, point_cloud, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    top_dists, ind = distances.topk(k+1, largest=False, sorted=True, dim=-1)

    top_dists = top_dists[:, :, 1:]  # Remove self-distance
    top_dists = torch.maximum(top_dists, torch.tensor([1e-12]).to(top_dists.device))

    top_dists = torch.mean(top_dists, dim=-1) # B, traj_points

    if y_lengths is not None:
        mask = torch.arange(point_cloud.shape[1], device=point_cloud.device)[None] >= y_lengths[:, None]
        top_dists[mask] = 0.0 
        top_dists_per_batch = top_dists.sum(1) / y_lengths
    else:
        top_dists_per_batch = torch.mean(top_dists, dim=-1)  # B, 

    if render:
        for b in range(B):
            top_dists_b = top_dists[b, :]
            top_dists_b = top_dists_b.flatten()

            top_dists_b = top_dists_b.detach().cpu().numpy()
            sns.histplot(top_dists_b)  # binwidth=0.0001
            plt.show()

    return top_dists_per_batch