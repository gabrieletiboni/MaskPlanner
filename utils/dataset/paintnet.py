import pdb
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


from utils import orient_in, set_seed, create_dirs
from utils.disk import get_dataset_downscale_factor, get_dataset_paths, load_stroke_npy, read_mesh_as_pointcloud, read_traj_file
from utils.pointcloud import center_traj, get_dim_traj_points, get_max_distance, remove_padding, center_pair, downsample_strokes, get_sequences_of_lambda_points, get_velocities, get_traj_feature_index, add_padding, reshape_stroke_to_segments


class PaintNetDataloader(data.Dataset):
    def __init__(self,
                 roots='',
                 dataset=None,
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
                 config={},
                 **kwargs):
        """
        roots : list of str
               list of paths for each dataset category
        dataset : str
                  dataset name (e.g. containers-v2, windows-v1, ...)
        pc_points : int
                  number of final subsampled points from the point-cloud (mesh)
        traj_points : int
                       number of final subsampled points from the trajectory
        split : str
                <train,test>
        stroke_pred : bool
                      Whether to return the trajectories as a point-cloud of strokes, rather than sequences of lambda points
        extra_data : tuple of strings
                     Whether to include velocities and/or orientations in trajectories.
                     e.g. ('vel',); ('vel', 'orientquat', 'orientrotvec', 'orientnorm');
        cache_size : int
                     number of obj-traj pairs to save in cache during training
        overfitting : int
                      if set, overfits to a single sample, whose index is <overfitting>
        lambda_points : int
                        instead of considering point-clouds (N,3), reshape data into lambda-sequences as (N/lambda, lambda)
                        Any remainder is filled up with padding values
        overlapping : int
                      number of overlapping points between subsequent lambda-sequences. overlapping > lambda_points
        normalization : str
                        normalization type. One of: None, 'per-mesh' (norm mesh to unit sphere),
                        'per-dataset' (scale based on max mesh point across dataset).
                        NOTE: all (mesh, traj) pairs are ALWAYS shifted to have mesh zero-mean.
        data_scale_factor : float
                            Manually sets the 'per-dataset' scale factor. Useful for few-shot experiments
                            where the pretrained network has been trained on a different scale factor.
        train_portion : int
                            if set, loads only <train_portion> samples from the training dataset
        """
        self.dataset = dataset
        if isinstance(self.dataset, list) or isinstance(self.dataset, omegaconf.listconfig.ListConfig):  # (kept for retro-compatibility) Handles joint category training
            self.dataset = '-'.join(self.dataset)

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
        self.overfitting = overfitting
        self.weight_orient = weight_orient
        self.config = config

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
        assert set(augmentations) <= {'rot', 'roty', 'rotx'}, f'Some augmentation is not available: {augmentations}'
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
        parents = []
        dir_samples = []
        self.multi_root = False if len(self.roots) == 1 else True
        for root in self.roots:
            assert os.path.isdir(root), f"Dataset dir not found on system: {root}"
            with open(os.path.join(root, f'{split}_split.json'), 'r') as f:
                new_dir_samples = [str(d) for d in json.load(f)]
                parents += [root for i in range(len(new_dir_samples))]
                dir_samples += new_dir_samples

        self.datapath = []
        for c, (parent, curr_dir) in enumerate(zip(parents, dir_samples)):
            if self.overfitting is not None:
                if c != self.overfitting:
                    self.datapath.append(tuple())
                    continue

            mesh_filename = curr_dir+'.obj'
            traj_filename = 'trajectory.txt'
            assert os.path.exists(os.path.join(parent, curr_dir, mesh_filename)), f"mesh file {mesh_filename} does not exist in dir: {os.path.join(parent, curr_dir)}"
            assert os.path.exists(os.path.join(parent, curr_dir, traj_filename)), f"traj file {traj_filename} does not exist in dir: {os.path.join(parent, curr_dir)}"

            if normalization == 'per-dataset' and self.dataset_mean_max_distance is None:
                # self.dataset_max_distance = max(self.dataset_max_distance, get_max_distance(os.path.join(parent, curr_dir, mesh_filename)))
                self.compute_dataset_mean_max_distance.append(get_max_distance(os.path.join(parent, curr_dir, mesh_filename)))

            mesh_file = os.path.join(parent, curr_dir, mesh_filename)
            traj_file = os.path.join(parent, curr_dir, traj_filename)
            self.datapath.append(  (mesh_file, traj_file, curr_dir)   )

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

        if index in self.cache:  # Retrieve from cache
            point_cloud, traj, traj_as_pc, stroke_ids, dirname = self.cache[index]
        else:  # Retrieve from filesystem
            mesh_file, traj_file, dirname = self.datapath[index]
            
            if not self._preprocessed_sample_exists(mesh_file, traj_file, self.extra_data, self.weight_orient):
                """Load sample from disk, preprocess it, and save it for faster loading."""
                print(f'Loading sample {os.path.basename(dirname)} for the first time. Next queries will be faster as the preprocessed version will now be saved on disk (make sure you have write rights).')
                point_cloud = read_mesh_as_pointcloud(mesh_file)
                traj, stroke_ids = read_traj_file(traj_file, extra_data=self.extra_data, weight_orient=self.weight_orient)
                point_cloud, traj = center_pair(point_cloud, traj, mesh_file)  # Shift to zero mean

                if self.normalization == 'per-dataset':
                    point_cloud /= self.dataset_mean_max_distance
                    traj[:, :3] /= self.dataset_mean_max_distance
                elif self.normalization == 'per-mesh':
                    # max_distance = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
                    max_distance = get_max_distance(mesh_file)
                    point_cloud /= max_distance
                    traj[:, :3] /= max_distance

                assert point_cloud.shape[0] >= self.pc_points
                choice = np.random.choice(point_cloud.shape[0], self.pc_points, replace=False)  # Sub-sample point-cloud randomly
                point_cloud = point_cloud[choice, :]

                preprocessed_dir = os.path.join(os.path.abspath(os.path.join(mesh_file, os.pardir)), 'paintnet_preprocessed_sample')
                create_dirs(preprocessed_dir)
                filename = self._get_preprocessed_sample_name()
                np.savez(os.path.join(preprocessed_dir, filename),
                         point_cloud=point_cloud,
                         traj=traj,
                         stroke_ids=stroke_ids)
            else:
                # Load already-preprocessed sample
                preprocessed_dir = os.path.join(os.path.abspath(os.path.join(mesh_file, os.pardir)), 'paintnet_preprocessed_sample')
                filename = self._get_preprocessed_sample_name()
                sample = np.load(os.path.join(preprocessed_dir, filename))
                point_cloud, traj, stroke_ids = sample['point_cloud'], sample['traj'], sample['stroke_ids']
                
            assert 'pc_online_subsampling' not in self.augmentations, 'temp sanity check: NotImplementedError'
            assert 'traj_with_equally_spaced_points' not in self.config, 'temp sanity check: NotImplementedError'
            
            
            if self.stroke_pred:
                traj, stroke_ids = downsample_strokes(traj, stroke_ids, self.stroke_points)  # returns each stroke, downsampled to `stroke_points` poses
                assert traj.shape[0] == 6, "Temp assert for cuboids only. Stroke number in GT samples is expected to be 6, ALWAYS"
                traj = traj.reshape(traj.shape[0], -1)   # from (6, stroke_points, outdim) to (6, stroke_points*outdim)

            else:
                choice = np.round_(np.linspace(0, (traj.shape[0]-1), num=self.traj_points)).astype(int)  # Sub-sample traj at equal intervals (up to rounding) for a total of <self.traj_points> points
                traj = traj[choice, :]
                stroke_ids = stroke_ids[choice]

                # reconstructing the full traj from lambda segments is cumbersome or sometimes impossible (some segments are not created)
                # save it for usage in e.g. PCD metrics computation
                traj_as_pc = traj.copy()

                if self.lambda_points > 1:
                    traj, stroke_ids = get_sequences_of_lambda_points(traj, stroke_ids, self.lambda_points, dirname, overlapping=self.overlapping, extra_data=self.extra_data)  # Note: stroke_ids and traj are padded
                    
            # plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
            # visualize_pc(point_cloud, plotter=plotter, index=(0,0))
            # visualize_mesh(os.path.join(self.root, dirname, dirname+'_norm.obj'), plotter=plotter, index=(0,0))
            # visualize_sequence_traj(traj, plotter=plotter, index=(0,0), extra_data=self.extra_data)
            # visualize_traj(traj, plotter=plotter, index=(0,0), extra_data=self.extra_data)
            # visualize_mesh_traj(os.path.join(self.root, dirname, dirname+'_norm.obj'), traj, extra_data=extra_data)
            # plotter.add_axes_at_origin()
            # plotter.show()

            # mean_knn = mean_knn_distance(traj[:, :3], k=1, render=True)
            # pdb.set_trace()

            if 'vel' in self.extra_data:  # Include velocities
                assert self.lambda_points == 1, 'The opposite needs to be thought through: does it make sense to compute velocities for sequences? ALSO. MAKE SURE PADDING IS TAKEN CARE OF OTHERWISE'
                traj_vel = get_velocities(traj, stroke_ids)
                traj = np.concatenate((traj, traj_vel), axis=-1)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_cloud, traj, traj_as_pc, stroke_ids, dirname)


        """
            Data processing after cache loading.
            Everythign that happens here is recomputed for the same sample on different epochs.
        """
        outdim = get_dim_traj_points(self.extra_data)

        if len(self.augmentations) > 0:
            if 'rot' in self.augmentations or 'roty' in self.augmentations or 'rotx' in self.augmentations:
                """3D Rotations have multiple representations, but can be described by a minimum of 3 parameters.
                For example, quaternions use 4 plus a constraint. In general, all rotations in 3D space can
                be broken down to a single rotation about some axis (Euler's theorem), hence be described by
                one parameterization of choice (quaternions, rotation matrix, euler angles).
                https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
                """
                

                point_cloud, traj, traj_as_pc = point_cloud.copy(), traj.copy(), traj_as_pc.copy()  # Not sure if needed, but this way I don't apply changes to the cache

                if 'roty' in self.augmentations:
                    alpha = np.random.uniform(-math.pi, math.pi)
                    rot = Rot.from_euler(seq="y", angles=alpha)
                elif 'rotx' in self.augmentations:
                    alpha = np.random.uniform(-math.pi, math.pi)
                    rot = Rot.from_euler(seq="x", angles=alpha)
                else:
                    rot = Rot.random()

                if self.lambda_points > 1:
                    traj = traj.reshape(-1, outdim)
                    traj = remove_padding(traj, extra_data=self.extra_data)

                    traj_as_pc = traj_as_pc.reshape(-1, outdim)
                    traj_as_pc = remove_padding(traj_as_pc, extra_data=self.extra_data)

                    # visualize_pc(point_cloud)
                    # visualize_traj(traj, extra_data=self.extra_data)
                    if orient_in(self.extra_data)[0]:
                        orient_indexes = get_traj_feature_index(orient_in(self.extra_data)[1], self.extra_data)
                        point_cloud, traj[:, :3], traj[:, orient_indexes] = rot.apply(point_cloud), rot.apply(traj[:, :3].copy()), rot.apply(traj[:, orient_indexes].copy())
                        traj_as_pc[:, :3], traj_as_pc[:, orient_indexes] = rot.apply(traj_as_pc[:, :3].copy()), rot.apply(traj_as_pc[:, orient_indexes].copy())
                    else:
                        point_cloud, traj[:, :3] = rot.apply(point_cloud), rot.apply(traj[:, :3].copy())
                        traj_as_pc[:, :3] = rot.apply(traj_as_pc[:, :3].copy())
                    # visualize_pc(point_cloud)
                    # visualize_traj(traj, extra_data=self.extra_data)

                    traj = traj.reshape(-1, outdim*self.lambda_points)
                    traj = add_padding(traj, traj_points=self.traj_points, lmbda=self.lambda_points, overlapping=self.overlapping, extra_data=self.extra_data)

                    traj_as_pc = traj_as_pc.reshape(-1, outdim*self.lambda_points)
                    traj_as_pc = add_padding(traj_as_pc, traj_points=self.traj_points, lmbda=self.lambda_points, overlapping=self.overlapping, extra_data=self.extra_data)
                else:
                    point_cloud, traj[:,:outdim] = rot.apply(point_cloud), rot.apply(traj[:,:outdim])
                    traj_as_pc[:,:outdim] = rot.apply(traj_as_pc[:,:outdim])

                    if 'vel' in self.extra_data:
                        raise NotImplementedError('This part is now deprecated. Fix it with the correct indexes in case orient_in() is True etc.')
                        traj[:,3:6] = rot.apply(traj[:,3:6])


        return point_cloud, traj, traj_as_pc, stroke_ids, dirname

    
    def __len__(self):
        return len(self.datapath)


    def _preprocessed_sample_exists(self, mesh_file, traj_file, extra_data, weight_orient):
        """Check whether the already-preprocessed sample exists in the dataset dir
            for faster loading.
        """
        target_dir = os.path.join(os.path.abspath(os.path.join(mesh_file, os.pardir)), 'paintnet_preprocessed_sample')
        target_filename = self._get_preprocessed_sample_name()
        return os.path.isfile(os.path.join(target_dir, target_filename))


    def _get_preprocessed_sample_name(self):
        """Return filename of preprocessed sample"""

        # Encoding for normals in the trajectory
        extras = "_".join(list(self.extra_data))
        weightOrient = str(self.weight_orient)

        # number of points subsampled for point cloud
        pc_points = str(self.pc_points)

        # normalization used for point-cloud and trajectory
        norm = str(self.normalization)

        if self.multi_root:  # temp check for joint-category training to avoid changing name for previously saved data
            if self.normalization == 'per-dataset':
                norm += str(round(self.dataset_mean_max_distance, 2))

        # Wether pc is subsample online or just once and saved
        pc_online_subsampling = 'True' if 'pc_online_subsampling' in self.augmentations else 'False'

        # Wether traj_sampling_v2 preprocess has been applied already
        # traj_sampling_v2_flag = '_TrajSamplingV2'+str(self.config['equal_spaced_points_distance']) if self.config['traj_with_equally_spaced_points'] else ''

        filename = "preprocessed_"+extras+ \
                   "_weightOrient"+weightOrient+ \
                   "_pcPoints"+pc_points+ \
                   "_normalization"+norm+ \
                   "_pcOnlineSub"+pc_online_subsampling+ \
                   ".npz"

        return filename