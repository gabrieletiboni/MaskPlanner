from functools import reduce
from glob import glob
import json
import os
import os.path
import math
import random

import omegaconf
import torch
import torch.utils.data as data
import numpy as np
from scipy.spatial.transform import Rotation as Rot


from utils import orient_in, set_seed
from utils.disk import get_dataset_downscale_factor, get_dataset_paths, load_stroke_npy, read_mesh_as_pointcloud, read_traj_file
from utils.pointcloud import center_traj, get_dim_traj_points, get_max_distance, remove_padding, center_pair, downsample_strokes, get_sequences_of_lambda_points, get_velocities, get_traj_feature_index, add_padding, reshape_stroke_to_segments

from concatenation.utils import nearest_neighbor_graph


class StrokeDataset(data.Dataset):
    def __init__(self,
                 roots='',
                 dataset=None,
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
                 normalization='per-mesh',
                 data_scale_factor=None,
                 train_portion=None,
                 neighbors= 0.2,
                 knn_strat='percentage',
                 distance='euclid',
                 config={},
                 **kwargs):
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
        self.weight_orient = weight_orient
        self.outdim = get_dim_traj_points(extra_data)
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.distance = distance
        self.load_pc = load_pc
        self.split = split

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
        assert set(augmentations) <= {'gaussian_noise'}, f'Some augmentation is not available: {augmentations}'
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
            traj_strokes = [(mesh_path, stroke_path, mesh_folder) for stroke_path in glob(os.path.join(parent, mesh_folder, traj_folder, "*.npy"))]
            
            # Fast reading of strokes shapes
            self.traj_strokes_lengths[mesh_folder] = [np.load(stroke_path, mmap_mode="r").shape[0] for _, stroke_path, _ in traj_strokes]

            self.datapath += traj_strokes

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
            index = self.overfitting
        
        item = {}

        if index in self.cache:
            item = self.cache[index]
        else:
            mesh_path, stroke_path, mesh_folder = self.datapath[index]
            if mesh_folder not in self.pc_cache:
                if self.load_pc:
                    point_cloud, pc_centroid, pc_max_distance = read_mesh_as_pointcloud(mesh_path, return_more=True)
                    
                    choice = np.random.choice(point_cloud.shape[0], self.pc_points, replace=False)  # Sub-sample point-cloud randomly
                    point_cloud = point_cloud[choice, :]

                    self.pc_cache[mesh_folder] = point_cloud, pc_centroid, pc_max_distance
                    if self.normalization == 'per-dataset':
                        point_cloud /= self.dataset_mean_max_distance
                    elif self.normalization == 'per-mesh':
                        point_cloud /= pc_max_distance

                    point_cloud = point_cloud - pc_centroid # center point cloud
                else:
                    pc_centroid_maxdistance = np.load(os.path.join(os.path.dirname(mesh_path), "pc_centroid_maxdistance.npz"), mmap_mode="r")
                    point_cloud = np.empty(())
                    pc_centroid, pc_max_distance = pc_centroid_maxdistance["pc_centroid"], pc_centroid_maxdistance["pc_max_distance"]

                if self.split == 'train':
                    self.pc_cache[mesh_folder] = point_cloud, pc_centroid, pc_max_distance
            else:
                point_cloud, pc_centroid, pc_max_distance = self.pc_cache[mesh_folder]
            
            stroke = load_stroke_npy(stroke_path, extra_data=self.extra_data, weight_orient=self.weight_orient)

            # Preprocess stroke
            multi_path_points = sum(self.traj_strokes_lengths[mesh_folder])
            stroke_points = (self.traj_points*stroke.shape[0])//multi_path_points + self.lambda_points # +lambda_points to have at least one segment
            choice = np.round_(np.linspace(0, (stroke.shape[0]-1), num=stroke_points)).astype(int)  # Sub-sample stroke at equal intervals (up to rounding) for a total of <self.traj_points> points
            
            assert choice.shape[0] > 3, f'Choice shape is {choice.shape[0]} for stroke {stroke_path}'

            stroke = stroke[choice, :]
            stroke[:, :3] = stroke[:, :3] - pc_centroid # center stroke
            if self.normalization == 'per-dataset':
                stroke[:, :3] /= self.dataset_mean_max_distance
            elif self.normalization == 'per-mesh':
                stroke[:, :3] /= pc_max_distance

            if self.lambda_points > 1:
                stroke = reshape_stroke_to_segments(stroke, self.lambda_points, self.overlapping).reshape(-1, self.lambda_points*self.outdim)
        
            # Random permutation of segments
            perm_idx = np.random.permutation(stroke.shape[0])
            antiperm_idx = np.argsort(perm_idx)
            stroke = stroke[perm_idx]

            tour_nodes = antiperm_idx # sequence of nodes to visit
            tour_edges = np.zeros((stroke.shape[0], stroke.shape[0])) # e_ij = e_ji = 1 if two nodes are adjacent
            tour_edges[tour_nodes[:-1], tour_nodes[1:]] = 1
            tour_edges[tour_nodes[1:], tour_nodes[:-1]] = 1

            graph = nearest_neighbor_graph(stroke.reshape(-1, self.lambda_points, self.outdim), self.neighbors, self.knn_strat, self.distance)

            item = {
                'stroke': stroke,
                'graph': graph,
                'tour_edges': tour_edges,
                'tour_nodes': tour_nodes,
                'point_cloud': point_cloud,
                'mesh_folder': mesh_folder,
                'stroke_path': stroke_path,
            }
            if len(self.cache) < self.cache_size and self.split == 'train':
                self.cache[index] = item

        if len(self.augmentations) > 0:
            if 'gaussian_noise' in self.augmentations:
                stroke = item['stroke']
                graph = item['graph']
                stroke = stroke.reshape(stroke.shape[0], self.lambda_points, self.outdim)
                noise = np.tile(np.random.normal(0, 0.05, size=(stroke.shape[0], 1, 3)), (1, self.lambda_points, 1))
                stroke[..., :3] = stroke[..., :3] + noise
                graph = nearest_neighbor_graph(stroke, self.neighbors, self.knn_strat, self.distance)
                stroke = stroke.reshape(stroke.shape[0], -1)
                item['stroke'] = stroke
                item['graph'] = graph
                item['gaussian_noise'] = noise

        return item

    def __len__(self):
        return len(self.datapath)

    def get_item_by_mesh(self, mesh, stroke_path):
        mesh_folder_list= [(mesh, stroke_path) for _, stroke_path, mesh in self.datapath]
        index = mesh_folder_list.index((mesh, stroke_path))
        
        mesh_path, stroke_path, mesh_folder = self.datapath[index]
        
        return self[index]

    @staticmethod
    def _pad(x, len):
        stroke, graph, tour_edges, tour_nodes, point_cloud = x['stroke'], x['graph'], x['tour_edges'], x['tour_nodes'], x['point_cloud']
        if 'gaussian_noise' in x:
            noise = x['gaussian_noise']
        
        padding_size = len - stroke.shape[0]
        if padding_size > 0:
            stroke = np.pad(stroke, ((0, padding_size), (0, 0)), constant_values=-100)
            graph = np.pad(graph, (0, padding_size), constant_values=1) # constant value 1 since it's a negative adj matrix
            tour_edges = np.pad(tour_edges, (0, padding_size), constant_values=-100)
            tour_nodes = np.pad(tour_nodes, (0, padding_size), constant_values=-100)
            if 'gaussian_noise' in x:
                noise = np.pad(noise, ((0, padding_size), (0, 0), (0, 0)), constant_values=0)

        stroke = torch.as_tensor(stroke, dtype=torch.float)
        graph = torch.as_tensor(graph, dtype=torch.int8)
        tour_edges = torch.as_tensor(tour_edges, dtype=torch.int8)
        tour_nodes = torch.as_tensor(tour_nodes, dtype=torch.long)
        point_cloud = torch.as_tensor(point_cloud, dtype=torch.float)
        if 'gaussian_noise' in x:
                noise = torch.as_tensor(noise, dtype=torch.float)

        item = {**x,
            'stroke': stroke,
            'graph': graph,
            'tour_edges': tour_edges,
            'tour_nodes': tour_nodes,
            'point_cloud': point_cloud
        }
        if 'gaussian_noise' in x:
                item['gaussian_noise'] = noise
        return item
    
    @staticmethod
    def stack_strokes(data):
        max_len = reduce(lambda x, y: x if x > y else y, map(lambda x: x['stroke'].shape[0], data), 0)

        padded_data = [StrokeDataset._pad(x, max_len) for x in data]
        item_batch = {key: [x[key] for x in padded_data] for key in data[0]}

        for key in item_batch:
            if key not in {'mesh_folder', 'stroke_path'}:
                item_batch[key] = torch.stack(item_batch[key], dim=0)
        return item_batch
