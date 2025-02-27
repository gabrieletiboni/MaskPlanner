from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pdb
import json
import os
import os.path
import random
from functools import reduce
from glob import glob
from copy import deepcopy

import omegaconf
import numpy as np
import torch
import torch.utils.data as data

from utils import orient_in
from utils.disk import get_dataset_downscale_factor, load_stroke_npy, read_mesh_as_pointcloud
from utils.pointcloud import get_dim_traj_points, get_max_distance, reshape_stroke_to_segments
from concatenation.utils import nearest_neighbor_graph

def bit_quantization(segments, mean_max_distance, bit=8):
    dynamic_range = 2 ** bit - 1
    discrete_interval = mean_max_distance / (dynamic_range)#dynamic_range
    offset = (dynamic_range) / 2
    segments = segments / discrete_interval + offset

    segments = np.clip(segments, 0, dynamic_range-1)
    return segments.astype(np.int32)

def reorder_segments(segments):
    indeces = np.lexsort(segments.T[::-1])[::-1]
    return segments[indeces], indeces

def reorder_strokes(strokes, sort_v_ids, pad_id=-1):
    # apply sorted vertice-id and sort in-face-triple values.
    
    segments_ids = []
    for f in strokes:
        f_ids = np.concatenate([np.where(sort_v_ids==v_idx)[0] for v_idx in f])
        # max_idx = np.argmax(f_ids)
        # sort_ids = np.arange(len(f_ids))
        # sort_ids = np.concatenate([
        #     sort_ids[max_idx:], sort_ids[:max_idx]
        # ])
        segments_ids.append(f_ids)
        
    # padding for lexical sorting.
    max_length = max([len(f) for f in segments_ids])
    pad_segments_ids = np.array([
        np.concatenate([f, np.array([pad_id]*(max_length-len(f)))]) 
        for f in segments_ids
    ])
    
    # lexical sort over face triples.
    indeces = np.lexsort(pad_segments_ids.T[::-1])[::-1]
    segments_ids = [segments_ids[idx] for idx in indeces]
    return segments_ids, indeces

class MultipathDataset(data.Dataset):
    def __init__(self,
                 roots: List[str] = [],
                 dataset: str = None,
                 load_pc=False,
                 pc_points=5120,
                 traj_points=500,
                 lambda_points=1,
                 overlapping=0,
                 split='train',
                 stroke_pred=False,
                 stroke_points=100,
                 extra_data=None,
                 weight_orient=1.,
                 cache_size=2000,
                 overfitting=None,
                 augmentations=None,
                 normalization='per-dataset',
                 data_scale_factor=None,
                 train_portion=None,
                 bit=8,
                 config={},
                 **kwargs):
        """
            Handles loading of strokes and their corresponding stroke_ids.
            Used for downstream task with strokes as input, such as encoding
            segments of strokes into a latent space for clustering.
            
            roots : list of str
                    paths of dataset categories (list as it supports joint-category training)
            dataset : str
                      dataset name (may be concatenation of names for joint-category training)
        """
        self.dataset = dataset
        assert not isinstance(self.dataset, list) and not isinstance(self.dataset, omegaconf.listconfig.ListConfig)  # Should be concatenated beforehand

        self.roots = roots
        self.pc_points = pc_points
        self.traj_points = traj_points
        self.lambda_points = lambda_points
        self.overlapping = overlapping
        self.normalization = normalization
        self.data_scale_factor = data_scale_factor
        self.stroke_pred = stroke_pred
        self.stroke_points = stroke_points
        self.cache = {}
        self.cache_size = cache_size
        self.pc_cache = {}
        self.traj_strokes_lengths = {}
        self.overfitting = overfitting
        self.overfitting_n_samples = config.overfitting_n_samples if 'overfitting_n_samples' in config else 1
        self.weight_orient = weight_orient
        self.outdim = get_dim_traj_points(extra_data)
        self.load_pc = load_pc
        self.split = split
        self.config = config
        
        self.bit = bit
        self.bit_quantization = 'bit_quantization' in config and config['bit_quantization']  # enabled for Polygen

        """
            Sanity checks
        """
        assert len(self.roots) > 0, "No data root specified"
        assert lambda_points > overlapping, 'Overlapping can not be equal or lower than lambda'
        assert overlapping >= 0, 'overlapping value can only be positive'
        assert train_portion is None or (float(train_portion) > 0 and float(train_portion) <= 1), 'train_portion must be in range (0, 1]'

        if extra_data is not None and not(set(extra_data) <= {'vel', 'orientquat', 'orientrotvec', 'orientnorm'}):
            raise ValueError('extra_data allowed entries are ("vel", "orientquat", "orientrotvec, orientnorm")')
        elif extra_data is None:
            extra_data = tuple()
        assert not ('vel' in extra_data and orient_in(extra_data)[0]), 'vel and orientations together are not yet compatible. You would need to fix the network output as well'
        self.extra_data = extra_data

        if augmentations is None:
            augmentations = []
        assert set(augmentations) <= {'gaussian_noise', 'general_noise'}, f'Some augmentation is not available: {augmentations}'
        self.augmentations = augmentations

        assert normalization in ['none', 'per-mesh', 'per-dataset'], f'Normalization type {normalization} is not valid.'
        if normalization == 'per-dataset':
            if self.data_scale_factor is not None:
                self.dataset_mean_max_distance = self.data_scale_factor
            else:
                self.dataset_mean_max_distance = get_dataset_downscale_factor(self.dataset)  # Use a precomputed value
                if self.dataset_mean_max_distance is None:  # Precomputed value not found, compute it
                    self.compute_dataset_mean_max_distance = []

        """
            Directories loading
        """
        assert split in ['train', 'test'], f'Split value {split} is not valid'
        mesh_folder_paths = []
        for root in self.roots:
            assert os.path.isdir(root), f"Dataset dir not found on system: {root}"
            with open(os.path.join(root, f'{split}_split.json'), 'r') as fp:
                mesh_folder_paths += [(root, str(mesh_name)) for mesh_name in json.load(fp)]

        self.datapath = []
        for parent, mesh_folder in mesh_folder_paths:
            mesh_filename = mesh_folder+'.obj'
            traj_folder = 'trajectory'
            assert os.path.exists(os.path.join(parent, mesh_folder, mesh_filename)), f"mesh file {mesh_filename} does not exist in dir: {os.path.join(parent, mesh_folder)}"
            assert os.path.exists(os.path.join(parent, mesh_folder, traj_folder)), f"traj folder {traj_folder} does not exist in dir: {os.path.join(parent, mesh_folder)}"

            if normalization == 'per-dataset' and self.dataset_mean_max_distance is None:
                self.compute_dataset_mean_max_distance.append(get_max_distance(os.path.join(parent, mesh_folder, mesh_filename)))

            mesh_path = os.path.join(parent, mesh_folder, mesh_filename)
            multipath_stroke_paths = [stroke_path for stroke_path in glob(os.path.join(parent, mesh_folder, traj_folder, "*.npy"))]
            
            # Fast reading of strokes shapes
            fps = [np.load(stroke_path, mmap_mode="r") for stroke_path in multipath_stroke_paths]
            self.traj_strokes_lengths[mesh_folder] = [stroke.shape[0] for stroke in fps]
            for f in fps:
                f._mmap.close()
            del fps

            self.datapath.append((mesh_path, multipath_stroke_paths, mesh_folder))

        if split == 'train' and train_portion is not None:  # Few-shot experiments with only a subset of the training set
            # assert len(self.datapath) >= fewshot_samples, 'You are trying to use a subset of the training set but training set contains fewer samples than requested. Is this desired?'
            random.shuffle(self.datapath)  # Shuffle self.datapath list of tuples
            tot_samples = len(self.datapath)
            self.datapath = self.datapath[:int(train_portion*tot_samples)]
            assert len(self.datapath) > 0, f'The training set has zero samples due to the a train_portion value of: {train_portion}. Initial number of training samples was: {tot_samples}'


        if normalization == 'per-dataset' and self.dataset_mean_max_distance is None:  # Compute dataset mean if it has not been computed yet
            self.dataset_mean_max_distance = np.mean(self.compute_dataset_mean_max_distance)
            print(f'Mean_max_distance computed on the fly for split {split.upper()} of dataset {self.dataset.upper()}: {self.dataset_mean_max_distance}')

    def __getitem__(self, index):
        if self.overfitting is not None:
            if self.overfitting_n_samples > 1:
                index = (index % self.overfitting_n_samples) + 42
            else:
                index = self.overfitting  # overwrite index to the same sample every time
        
        item = {}

        if index in self.cache:
            item = deepcopy(self.cache[index])  # deepcopy because augmentations may modify the item object
        else:
            mesh_path, multipath_stroke_paths, mesh_folder = self.datapath[index]
            if mesh_folder not in self.pc_cache:
                if self.load_pc:
                    point_cloud, pc_centroid, pc_max_distance = read_mesh_as_pointcloud(mesh_path, return_more=True)
                    
                    choice = np.random.choice(point_cloud.shape[0], self.pc_points, replace=False)  # Sub-sample point-cloud randomly
                    point_cloud = point_cloud[choice, :]

                    self.pc_cache[mesh_folder] = point_cloud, pc_centroid, pc_max_distance
                    point_cloud = point_cloud - pc_centroid # center point cloud

                    if self.normalization == 'per-dataset':
                        point_cloud /= self.dataset_mean_max_distance
                    elif self.normalization == 'per-mesh':
                        point_cloud /= pc_max_distance

                else:
                    pc_centroid_maxdistance = np.load(os.path.join(os.path.dirname(mesh_path), "pc_centroid_maxdistance.npz"), mmap_mode="r")
                    point_cloud = np.empty(())
                    pc_centroid, pc_max_distance = pc_centroid_maxdistance["pc_centroid"], pc_centroid_maxdistance["pc_max_distance"]

                if self.split == 'train':
                    self.pc_cache[mesh_folder] = point_cloud, pc_centroid, pc_max_distance
            else:
                point_cloud, pc_centroid, pc_max_distance = self.pc_cache[mesh_folder]
            

            # Iterate over all strokes of this sample
            segments = []
            stroke_ids = []
            num_strokes = 0
            for stroke_path in multipath_stroke_paths:
                stroke = load_stroke_npy(stroke_path, extra_data=self.extra_data, weight_orient=self.weight_orient)  # load single stroke
                
                # Sub-sample stroke points
                multi_path_points = sum(self.traj_strokes_lengths[mesh_folder])  # tot number of points across all strokes
                stroke_points = (self.traj_points*stroke.shape[0])//multi_path_points  # subsample: number of target points
                assert stroke_points >= self.lambda_points, f'subsampling this stroke would result in fewer points than lambda, hence no segment would come result out of this stroke. if you wish to filter out such strokes, implement this.'
                choice = np.round_(np.linspace(0, (stroke.shape[0]-1), num=stroke_points)).astype(int)  # Sub-sample stroke at equal intervals (up to rounding) for a total of <self.traj_points> points
                stroke = stroke[choice, :]  # subsample stroke. Tot number of subsample stroke points will be <= self.traj_points

                # Reshape stroke points into segments
                if self.lambda_points > 1:
                    stroke = reshape_stroke_to_segments(stroke, self.lambda_points, self.overlapping).reshape(-1, self.lambda_points*self.outdim)

                # Assign ID to all stroke points/segments
                single_stroke_ids = len(stroke_ids)*np.ones(stroke.shape[0])
                
                segments.append(stroke)
                stroke_ids.append(single_stroke_ids)
                num_strokes += 1

            segments = np.concatenate(segments)  # list of segments for all strokes of this sample
            stroke_ids = np.concatenate(stroke_ids)  # list of stroke ids for all segments of this sample


            # Normalize segments
            points = segments.reshape(-1, self.outdim) # from segments to points
            points[:, :3] -= pc_centroid
            if self.normalization == 'per-dataset':
                points[:, :3] /= self.dataset_mean_max_distance
            elif self.normalization == 'per-mesh':
                stroke[:, :3] /= pc_max_distance
            segments = points.reshape(-1, self.lambda_points*self.outdim)  # from points to segments


            # Permutate segments (sanity check that points/segments order is not exploited)
            perm_idx = np.random.permutation(segments.shape[0])
            antiperm_idx = np.argsort(perm_idx)
            segments = segments[perm_idx]
            stroke_ids = stroke_ids[perm_idx]
            multipath_indexes = []
            for s_id in range(len(multipath_stroke_paths)):
                stroke_mask = stroke_ids[antiperm_idx] == s_id
                tour_nodes = antiperm_idx[stroke_mask]
                multipath_indexes.append(tour_nodes)


            # Debug: one_hot_encoding_sample
            if self.config.one_hot_encoding_sample and self.overfitting and self.overfitting_n_samples > 1:
                # one_hot_encoding_sample = 
                n_samples = self.overfitting_n_samples
                one_hot_encoding_sample = torch.nn.functional.one_hot(torch.tensor(index-42) % n_samples, num_classes=n_samples)  # (n_samples,) one-hot of this index (sample)
            else:
                one_hot_encoding_sample = None


            # Create final sample
            item = {
                'segments': segments, # segments (N, lambda*outdim)
                'stroke_ids': stroke_ids, # strokes ids
                'antiperm_idx': antiperm_idx, # unshuffle ids
                'multipath_indexes': multipath_indexes, # antiperm_idx grouped by stroke
                'num_strokes': num_strokes,  # number of strokes
                'one_hot_encoding_sample': one_hot_encoding_sample,

                'point_cloud': point_cloud,
                'mesh_path': mesh_path,
                'mesh_folder': mesh_folder,
                'pc_max_distance': pc_max_distance
            }

            if len(self.cache) < self.cache_size:
                self.cache[index] = deepcopy(item)  # deepcopy because augmentations may modify the item object


        """
            Data augmentation
        """
        if len(self.augmentations) > 0:
            segments = item['segments']
            if 'gaussian_noise' in self.augmentations:
                segments = segments.reshape(segments.shape[0], self.lambda_points, self.outdim)
                noise = np.tile(np.random.normal(0, 0.03, size=(segments.shape[0], 1, 3)), (1, self.lambda_points, 1))
                segments[..., :3] = segments[..., :3] + noise
                segments = segments.reshape(segments.shape[0], -1)
                item['segments'] = segments
                item['gaussian_noise'] = noise
            elif 'general_noise' in self.augmentations:
                # Add noise to x,y,z and noise to the normal vector encoding the orientation
                assert 'orientnorm' in self.extra_data and self.outdim == 6, 'general noise is tailored for 6D poses with x,y,z and orientation normals'

                segments = segments.reshape(segments.shape[0], self.lambda_points, self.outdim)
                trasl_noise = np.random.normal(0, self.config.trasl_noise_stdev, size=(segments.shape[0], self.lambda_points, 3))
                orient_noise = np.random.normal(0, self.config.orient_noise_stdev, size=(segments.shape[0], self.lambda_points, 3))
                noise = np.concatenate((trasl_noise, orient_noise), axis=-1)
                segments = segments + noise

                # re-normalize orientation normals after injecting noise
                segments[:, :, 3:] /= np.linalg.norm(segments[:, :, 3:], axis=-1)[:, :, np.newaxis]
                # TODO: apply weight_orient after re-normalization! Probably as below:
                #       segments[:, :, 3:] *= self.weight_orient

                segments = segments.reshape(segments.shape[0], -1)
                item['segments'] = segments
                item['general_noise'] = noise



        """
            Bit quantization and custom segments ordering (used for Polygen)
        """
        if self.bit_quantization:
            polygen_segments = bit_quantization(segments, 2, bit=self.bit)
            polygen_segments, ids = reorder_segments(polygen_segments)
            multipath_indexes_reordered = multipath_indexes
            multipath_indexes_reordered, s_ids = reorder_strokes(multipath_indexes, ids)
            multipath_indexes = [multipath_indexes[idx] for idx in s_ids]

            item = {
                **item,
                'polygen_segments': polygen_segments, # segments after bit-quantization + ordering preprocessing (model input for polygen)
                'indexes': multipath_indexes_reordered, # indexes after bit-quantization + ordering preprocessing (prediction gt for polygen)

                # additional data
                'reordering_ids': ids, # reordering ids
                'multipath_indexes': multipath_indexes
            }

        return item

    def __len__(self):
        return len(self.datapath)
    


class MultipathCollateBatch(object):
    """Aggregates individual items returned by MultipathDataset into a mini-batch"""

    def __init__(self, config):
        self.config = config

        self.bit_quantization = 'bit_quantization' in config and config['bit_quantization']  # enabled for Polygen
        self.augmentations = config.augmentations

        self.uneven_num_segments = config.uneven_num_segments if 'uneven_num_segments' in config else None
        assert self.uneven_num_segments in {None, 'duplicate'}


    def create_even_batch_of_segments(self, segments, stroke_ids):
        """Handle the variable number of segments per sample
            
            segments: batch of segments [(N, lambda*outdim), ...]
            stroke_ids: batch of stroke ids [(N,), ...]
        """
        if self.uneven_num_segments == 'duplicate':
            """
                Duplicate a sub-sample of points/segments to obtain
                the desired, fixed total number of points/segments.

                Roughly speaking, this shouldn't affect pointnet and pointnet++
                architectures too much. Interestingly, segments predicted by PaintNet
                may also happen to be overlapping (as more segments than strictly needed
                are predicted), so it may not be a bad idea to duplicate some GT segments
                in the first place as a data augmentation technique.
            """
            traj_points, lambda_points, overlapping = self.config.traj_points, self.config.lambda_points, self.config.overlapping

            target_num_segments_per_sample = (traj_points-lambda_points)//(lambda_points-overlapping) + 1

            even_segments = []
            even_stroke_ids = []
            fake_segments_mask = []
            # Iterate over samples of this batch
            for sample_segments, sample_stroke_ids in zip(segments, stroke_ids):
                num_missing_segments = target_num_segments_per_sample - sample_segments.shape[0]

                if num_missing_segments > 0:
                    # Choose which segments to duplicate
                    duplicate_ids = np.random.choice(sample_segments.shape[0], num_missing_segments, replace=False)

                    # Create duplicates
                    duplicate_segments = np.copy(sample_segments[duplicate_ids, :])
                    duplicate_stroke_ids = np.copy(sample_stroke_ids[duplicate_ids])

                    # Save even segments
                    even_segments.append(np.concatenate((sample_segments, duplicate_segments)))
                    even_stroke_ids.append(np.concatenate((sample_stroke_ids, duplicate_stroke_ids)))
                    fake_segments_mask.append(np.concatenate((np.zeros(sample_segments.shape[0], dtype=bool), np.ones(num_missing_segments, dtype=bool))))
                else:
                    even_segments.append(sample_segments)
                    even_stroke_ids.append(sample_stroke_ids)
                    fake_segments_mask.append(np.zeros(sample_segments.shape[0], dtype=bool))

            return even_segments, even_stroke_ids, fake_segments_mask


    def __call__(self, data):
        """Prepare batch of data.

            `data` is a list of <object> of len `batch_size`,
            where <object> is whatever the data.Dataset class returns.

            In my case, I return the dict `item`.
        """
        segments = [d['segments'] for d in data]  # batch of segments [(N, lambda*outdim), ...]
        stroke_ids = [d['stroke_ids'] for d in data]  # batch of stroke ids
        num_strokes = [d['num_strokes'] for d in data]  # batch of number of strokes

        fake_segments_mask = None  # True for segments that have been added as padding to create even batches
        if self.uneven_num_segments is not None:
            segments, stroke_ids, fake_segments_mask = self.create_even_batch_of_segments(segments, stroke_ids)  # handle the variable number of segments per sample
            segments = torch.stack([torch.as_tensor(segment, dtype=torch.float) for segment in segments])  # stack batch of segments into tensor
            stroke_ids = torch.stack([torch.as_tensor(stroke_id, dtype=torch.int64) for stroke_id in stroke_ids])  # stack batch of stroke ids into tensor

        one_hot_encoding_sample = [d['one_hot_encoding_sample'] for d in data]  # batch of stroke ids
        if one_hot_encoding_sample[0] is not None:
            one_hot_encoding_sample = torch.stack(one_hot_encoding_sample)

        antiperm_idx = [d['antiperm_idx'] for d in data] # unshuffle ids
        multipath_indexes = [d['multipath_indexes'] for d in data]  # batch of antiperm_idx grouped by stroke

        point_cloud = torch.stack([torch.as_tensor(d['point_cloud'], dtype=torch.float) for d in data])  # batch of object point-clouds
        pc_max_distance = [d['pc_max_distance'] for d in data]
        mesh_path = [d['mesh_path'] for d in data]
        mesh_folder = [d['mesh_folder'] for d in data]

        batch = {
            'segments': segments,
            'stroke_ids': stroke_ids,
            'antiperm_idx': antiperm_idx,
            'multipath_indexes': multipath_indexes,
            'num_strokes': num_strokes,
            'one_hot_encoding_sample': one_hot_encoding_sample,
            'fake_segments_mask': fake_segments_mask,
            'point_cloud': point_cloud,
            'pc_max_distance': pc_max_distance,
            'mesh_path': mesh_path,
            'mesh_folder': mesh_folder
        }


        if len(self.augmentations) > 0:
            if 'gaussian_noise' in data[0]:
                noise = [d['gaussian_noise'] for d in data]
                batch['gaussian_noise'] = noise
            elif 'general_noise' in data[0]:
                noise = [d['general_noise'] for d in data]
                batch['general_noise'] = noise


        if self.bit_quantization:  # used for Polygen
            polygen_segments = [torch.as_tensor(d['polygen_segments'], dtype=torch.int64) for d in data] # segments (polygen)
            indexes = [[torch.as_tensor(idx, dtype=torch.int64) for idx in d['indexes']] for d in data] # indexes
            reordering_ids = [d['reordering_ids'] for d in data]

            batch = {
                **batch,

                'polygen_segments': polygen_segments,
                'indexes': indexes,
                'reordering_ids': reordering_ids,
            }

        return batch