import time
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
from utils.pointcloud import mean_knn_distance, resample_strokes_at_equal_spaced_points, get_sizes_of_3dbbox, get_center_of_3dbbox, get_3dbbox, center_traj, get_dim_traj_points, get_max_distance, remove_padding, center_pair, downsample_strokes, get_sequences_of_lambda_points, get_velocities, get_traj_feature_index, add_padding, reshape_stroke_to_segments
from utils.visualize import *


class PaintNetODv1Dataloader(data.Dataset):
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
                 force_fresh_preprocess=False,
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

        force_fresh_preprocess : bool
                                 if set, using already preprocessed data from disk and preprocess it on the fly.
                                 The preprocessed data will not be saved on disk if this is True.
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
        self.load_extra_data = config['load_extra_data']
        self.overfitting = overfitting
        self.weight_orient = weight_orient
        self.config = config
        self.force_fresh_preprocess = force_fresh_preprocess

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
        self.outdim = get_dim_traj_points(extra_data)

        if augmentations is None:
            augmentations = []
        assert set(augmentations) <= {'pc_online_subsampling', 'general_noise'}, f'Some augmentation is not available: {augmentations}'
        self.augmentations = augmentations

        # Sanity check on provided load_extra_data items
        assert set(self.load_extra_data) <= {'stroke_masks', 'stroke_prototypes', 'segments_per_stroke', 'history_of_segments_per_stroke_v1', 'history_of_segments_per_stroke_v2'}, f'load_extra_data items must is not among the supported ones.'

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

        if index in self.cache:
            # Retrieve from RAM cache
            point_cloud, traj, traj_as_pc, stroke_ids, stroke_ids_as_pc, dirname, extra_items = self.cache[index]
        else:
            # Retrieve from filesystem
            mesh_file, traj_file, dirname = self.datapath[index]
            
            if not self._preprocessed_sample_exists(mesh_file, traj_file, self.extra_data, self.weight_orient) or self.force_fresh_preprocess:
                """(1) Load sample from disk,
                   (2) preprocess it,
                   (3) and save it for faster loading.
                """
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

                if 'pc_online_subsampling' not in self.augmentations:
                    # Sub-sample point-cloud only once, e.g. for deterministic results on test set
                    assert point_cloud.shape[0] >= self.pc_points
                    choice = np.random.choice(point_cloud.shape[0], self.pc_points, replace=False)  # Sub-sample point-cloud randomly
                    point_cloud = point_cloud[choice, :]

                # Anticipate traj_sampling_v2 of the trajectory to save time.
                if not self.stroke_pred and self.config['traj_with_equally_spaced_points']:
                    traj, stroke_ids = resample_strokes_at_equal_spaced_points(traj,
                                                                               stroke_ids,
                                                                               distance=self.config['equal_spaced_points_distance'],
                                                                               interpolate=False,
                                                                               equal_in_3d_space=self.config['equal_in_3d_space'])

                if not self.force_fresh_preprocess:
                    # Save processed data
                    print(f'Loading sample {os.path.basename(dirname)} for the first time. Next queries will be faster as the preprocessed version will now be saved on disk (make sure you have write rights).')
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
                

            """
                Data preprocessing BEFORE cache loading.
                Everything here happens only once, then it's saved in RAM cache.
            """
            if self.stroke_pred:
                """
                    Sample a fixed number of points for each stroke.
                    DEPRECATED: stroke_pred is deprecated. Only used for `mse_strokes` rollout_loss in strokeRollout task.
                """
                traj, stroke_ids = downsample_strokes(traj, stroke_ids, self.stroke_points)  # returns each stroke, downsampled to `stroke_points` poses
                assert traj.shape[0] == 6, "Temp assert for cuboids only. Stroke number in GT samples is expected to be 6, ALWAYS"
                traj = traj.reshape(traj.shape[0], -1)   # from (6, stroke_points, outdim) to (6, stroke_points*outdim)

                # convert traj and stroke_ids to single-point format
                n_strokes = traj.shape[0]
                traj = traj.reshape(n_strokes, self.stroke_points, self.outdim).reshape(n_strokes*self.stroke_points, self.outdim)  # (tot_num_points=n_strokes*stroke_points, outdim)
                stroke_ids = stroke_ids.reshape(-1)  # (tot_num_points=n_strokes*stroke_points)

                traj_as_pc = traj.copy()
                stroke_ids_as_pc = stroke_ids.copy()

            else:
                """
                    Sub-sample trajectory
                """
                if self.config['traj_with_equally_spaced_points']:
                    # Avoid re-interpolation of the input trajectory as it's already very dense. Simply sub-sample consecutive points with a min-distance threshold.
                    # traj ends up having a variable number of points

                    """
                        Commented out, it's now already done as a pre-processing step to save trajectories to disk that are pre-sampled.
                    """
                    pass
                    
                else:
                    choice = np.round_(np.linspace(0, (traj.shape[0]-1), num=self.traj_points)).astype(int)  # Sub-sample traj at equal intervals (up to rounding) for a total of <self.traj_points> points
                    traj = traj[choice, :]
                    stroke_ids = stroke_ids[choice]


                """
                    Create segments
                """
                if self.lambda_points > 1:
                    traj_as_pc = traj.copy()  # single-point format
                    stroke_ids_as_pc = stroke_ids.copy()  # single-point format, no padding

                    # traj and stroke_ids in segments format + padding (-100 values for traj vectors, -1 for stroke_ids)
                    traj, stroke_ids = get_sequences_of_lambda_points(traj, stroke_ids, self.lambda_points, dirname, overlapping=self.overlapping, extra_data=self.extra_data)  # Note: stroke_ids and traj are padded
                    # TODO: the above function may filter out strokes if not even a segment can be created out of it. In turn, this creates non-matching stroke_ids_as_pc and stroke_ids, and even traj and traj_as_pc (those points should be canceled out from traj_as_pc too!)
                else:
                    traj_as_pc = traj.copy()  # single-point format
                    stroke_ids_as_pc = stroke_ids.copy()  # single-point format, no padding


            """
                Visualizations (sanity checks)
            """
            # plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
            # visualize_pc(point_cloud, plotter=plotter, index=(0,0))
            # visualize_mesh_v2(dirname, plotter=plotter, index=(0,0), config=self.config)
            # visualize_sequence_traj(traj, plotter=plotter, index=(0,0), extra_data=self.extra_data)
            # visualize_traj(traj, plotter=plotter, index=(0,0), extra_data=self.extra_data)
            # visualize_mesh_traj(dirname, traj, plotter=plotter, index=(0,0), config=self.config, stroke_ids=stroke_ids)
            # plotter.add_axes_at_origin()
            # plotter.show()
            # mean_knn = mean_knn_distance(traj_as_pc[:, :3], k=1, render=True)  # some points may be closer than `equal_spaced_points_distance` due to strokes intersecting each other.


            """
                Compute GT stroke masks

                from: padded `stroke_ids` Tensor of dim [(traj_points-lambda)//(lambda-overlapping)+1,]
                to: padded `stroke_masks` as binary Tensor of dim [n_strokes, (traj_points-lambda)//(lambda-overlapping)+1]

                Note: padded segments are simply given the 0 value for all masks
            """
            stroke_masks = None
            if 'stroke_masks' in self.load_extra_data:
                stroke_masks = []
                for stroke_id in np.unique(stroke_ids_as_pc):
                    stroke_mask = (stroke_ids == stroke_id).astype(int)
                    stroke_masks.append(stroke_mask)
                stroke_masks = np.stack(stroke_masks)

            # Save num of strokes of this sample
            n_strokes = len(np.unique(stroke_ids_as_pc))
            assert -1 not in np.unique(stroke_ids_as_pc), 'fake stroke ids should not be here'

            # Include velocities to input trajectory data points
            if 'vel' in self.extra_data:
                # DEPRECATED
                assert self.lambda_points == 1, 'The opposite needs to be thought through: does it make sense to compute velocities for sequences? ALSO. MAKE SURE PADDING IS TAKEN CARE OF OTHERWISE'
                traj_vel = get_velocities(traj, stroke_ids)
                traj = np.concatenate((traj, traj_vel), axis=-1)

            extra_items = {
                'stroke_masks': stroke_masks,
                'n_strokes': n_strokes,
            }
            
            # Save item to RAM cache
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_cloud, traj, traj_as_pc, stroke_ids, stroke_ids_as_pc, dirname, extra_items)



        """
            Data processing AFTER cache loading.
            Everythign that happens here is recomputed for the same sample on different epochs.
        """

        # Load stroke prototypes
        stroke_prototypes = None
        if 'stroke_prototypes' in self.load_extra_data:
            stroke_prototypes, info, stroke_order_check = self._get_stroke_prototypes(traj_as_pc, stroke_ids_as_pc, stroke_prototype_kind=self.config['stroke_prototype_kind'])
            # # Visualize GT 3Dbboxes (stroke_prototype_kind='3d_bboxes')
            # plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
            # visualize_traj(traj, plotter=plotter, index=(0,0), extra_data=self.extra_data)
            # for bbox in info['canonical_3dbboxes']:
            #     visualize_box(bbox, plotter=plotter, index=(0,0), color='orange')
            # plotter.show()

            # # Visualize SoPs tokens (stroke_prototype_kind='start_of_path_token')
            # plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
            # for proto in stroke_prototypes:
            #     visualize_mesh_traj(dirname, proto.reshape(-1, self.outdim), config=self.config, plotter=plotter, index=(0,0))
            # plotter.show()


        # Load segments and points per stroke
        segments_per_stroke, max_num_segments, points_per_stroke = None, None, None
        if 'segments_per_stroke' in self.load_extra_data:
            # Save segments/points per stroke explicitly for strokeRollout task
            segments_per_stroke, stroke_order_check_2 = get_vectors_per_stroke(traj, stroke_ids)  # list of varying-length strokes as [(N1, outdim*lambda_points), (N2, outdim*lambda_points)]
            max_num_segments = np.max([stroke_segments.shape[0] for stroke_segments in segments_per_stroke])

            # Regardless of lambda and overlapping, also save the single non-overlapping sequence of stroke points.
            points_per_stroke, stroke_order_check_3 = get_vectors_per_stroke(traj_as_pc, stroke_ids_as_pc)  # list of varying-length strokes as [(N1, outdim*lambda_points), (N2, outdim*lambda_points)]

            assert np.array_equal(stroke_order_check_2, stroke_order_check_3), 'The order that we are fetching strokes must be the same to assure matching items at the same i-th index.'
        
        
        # Load histories of segments for autoregressive tasks
        segments_per_substroke, segments_per_init_substroke = None, None
        strokewise_history_batch, strokewise_target_batch, strokewise_stroke_ids_batch, strokewise_end_of_path_batch = None, None, None, None
        if 'history_of_segments_per_stroke_v1' in self.load_extra_data:
            # Return a random subsample of the stroke of length self.config['substroke_points']
            assert 'segments_per_stroke' in self.load_extra_data
            segments_per_substroke, segments_per_init_substroke = self._create_stack_of_history_batches_v1(segments_per_stroke,
                                                                                                           history_length_plus_one=self.config['substroke_points'])
        elif 'history_of_segments_per_stroke_v2' in self.load_extra_data:
            assert 'stroke_prototypes' in self.load_extra_data
            assert 'segments_per_stroke' in self.load_extra_data
            """
                V2
                    - substroke_points is the length of history, does not include the next token
                    - all the possible histories of all strokes are fetched for each sample, instead of a single history sample for each stroke. This requires a much smaller batch size, and maybe fewer epochs.
                    - handles stroke prototypes different than 3D bboxes
                    - handles noisy teacher forcing
            """
            (strokewise_history_batch, \
             strokewise_target_batch, \
             strokewise_stroke_ids_batch, \
             strokewise_end_of_path_batch) = self._create_stack_of_history_batches_v2(segments_per_stroke,
                                                                                      stroke_order_check_2,
                                                                                      self.config['substroke_points'])
            assert np.array_equal(stroke_order_check, stroke_order_check_2) and np.array_equal(stroke_order_check, stroke_order_check_3), 'The order that we are fetching strokes must be the same to assure matching items at the same i-th index.'
        


        """
            Data augmentation
        """
        if len(self.augmentations) > 0:
            
            if 'pc_online_subsampling' in self.augmentations:
                # subsample original dense point-cloud with different samples every time
                assert point_cloud.shape[0] >= self.pc_points
                choice = np.random.choice(point_cloud.shape[0], self.pc_points, replace=False)  # Sub-sample point-cloud randomly
                point_cloud = point_cloud[choice, :]


            if 'general_noise' in self.augmentations and self.config['sample_substroke_v2']:
                # teacher forcing with noisy history of segments for autoregressive_v2 task

                # from (K, history length, lambda*outdim) to (K, history length, lambda, outdim)
                strokewise_history_batch = strokewise_history_batch.reshape(strokewise_history_batch.shape[0],
                                                                            self.config['substroke_points'],
                                                                            self.lambda_points,
                                                                            self.outdim) 
                trasl_noise = np.random.normal(0, self.config['trasl_noise_stdev'], size=(strokewise_history_batch.shape[0], self.config['substroke_points'], self.lambda_points, 3))
                orient_noise = np.random.normal(0, self.config['orient_noise_stdev'], size=(strokewise_history_batch.shape[0], self.config['substroke_points'], self.lambda_points, 3))
                noise = np.concatenate((trasl_noise, orient_noise), axis=-1)

                strokewise_history_batch += noise
                strokewise_history_batch[:, :, :, 3:] /= np.linalg.norm(strokewise_history_batch[:, :, :, 3:], axis=-1)[:, :, :, np.newaxis]
                strokewise_history_batch[:, :, :, 3:] *= self.weight_orient

                # Back to (K, history length, lambda*outdim)
                strokewise_history_batch = strokewise_history_batch.reshape(strokewise_history_batch.shape[0],
                                                                            self.config['substroke_points'],
                                                                            -1)

                # # Visualize noisy histories and GT next segment
                # view_10_histories = np.random.choice(np.arange(strokewise_history_batch.shape[0]), 10, replace=False)
                # for view_ith in view_10_histories:
                #     plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
                #     visualize_mesh_traj(dirname, strokewise_history_batch[view_ith], config=self.config, plotter=plotter, index=(0,0))
                #     visualize_mesh_traj(dirname, strokewise_target_batch[view_ith], config=self.config, trajc='blue', plotter=plotter, index=(0,0))
                #     plotter.show()


        # Create final sample
        item = {
            'point_cloud': point_cloud,
            'traj': traj,
            'traj_as_pc': traj_as_pc,
            'segments_per_stroke': segments_per_stroke,  # could also be points_per_stroke, according to lambda
            'points_per_stroke': points_per_stroke,  # explicitly save points_per_stroke regardless of lambda
            'max_num_segments': max_num_segments,
            'stroke_ids': stroke_ids,
            'stroke_ids_as_pc': stroke_ids_as_pc,
            'stroke_masks': extra_items['stroke_masks'],
            'stroke_prototypes': stroke_prototypes,
            'dirname': dirname,
            'n_strokes': extra_items['n_strokes'],

            # Autoregressive_v1
            'segments_per_substroke': segments_per_substroke,  # could also be points_per_substroke, according to lambda
            'segments_per_init_substroke': segments_per_init_substroke,  # initial sequence of the stroke (used for inference). could also be points_per_init_substroke, according to lambda
            # Autoregressive_v2
            'strokewise_history_batch': strokewise_history_batch,
            'strokewise_target_batch': strokewise_target_batch,
            'strokewise_stroke_ids_batch': strokewise_stroke_ids_batch,
            'strokewise_end_of_path_batch': strokewise_end_of_path_batch
        }
        
        return item
    

    def __len__(self):
        return len(self.datapath)


    def _create_stack_of_history_batches_v1(self, segments_per_stroke, history_length_plus_one):
        """
            Autoregressive _v1
        """
        segments_per_substroke = []
        segments_per_init_substroke = []
        for stroke in segments_per_stroke:
            stroke_length, points_dim = stroke.shape
            assert stroke_length > history_length_plus_one, f'ERROR: cannot subsample a sequence of {history_length_plus_one} points/segments for a stroke that is {stroke_length} points/segments long.'

            end_token_id = np.random.choice(np.arange(stroke_length))  # sample point/segment index that will be the final token of the sequence

            if (end_token_id + 1) - history_length_plus_one >= 0:
                """No initial padding needed"""

                substroke = stroke[(end_token_id+1)-history_length_plus_one:(end_token_id+1)].copy()
            else:
                """Pad initial sequence with zeros, if end token is sampled at the beginning of the stroke."""
                valid_points = stroke[0:(end_token_id+1)].copy()

                # Create padding points
                num_missing_points = history_length_plus_one - (end_token_id+1)
                padding_points = np.zeros((num_missing_points, points_dim))
                substroke = np.concatenate((padding_points, valid_points), axis=0)

            segments_per_substroke.append(substroke)

            # Save first point/segment for rollout with known initial point/segment
            init_substroke = stroke[0:1].copy()
            padding_points = np.zeros((history_length_plus_one - 1, points_dim))
            init_substroke = np.concatenate((padding_points, init_substroke), axis=0)

            segments_per_init_substroke.append(init_substroke)
        
        return segments_per_substroke, segments_per_init_substroke


    def _create_stack_of_history_batches_v2(self, segments_per_stroke, path_ids, K, concat_histories_with_vector=None):
        """
        Constructs a batch of history sequences of length K from input paths and returns the corresponding path IDs.
        
        Args:
            segments_per_stroke (list of np.ndarray): A list of paths, each path being a numpy array of shape (N, D).
            path_ids (np.ndarray): A numpy array of shape (L,), containing the ID of each input path.
            K (int): The length of the history sequence to be generated.
            
        Returns:
            np.ndarray: A batch of history sequences, shape (total_batches, K, D).
            np.ndarray: A batch of target sequences, shape (total_batches, D).
            np.ndarray: An array of path IDs corresponding to each history, shape (total_batches,).
        """
        # List to collect all history sequences, their corresponding next items, and path IDs
        history_batch = []
        target_batch = []
        path_id_batch = []
        end_of_path_batch = []
        
        # Iterate through each path
        for path, path_id in zip(segments_per_stroke, path_ids):
            N, D = path.shape
            
            # Create histories with padding
            for i in range(N):
                # Create an empty history with zeros, of shape (K, D)
                history = np.zeros((K, D))
                
                # Determine the slice indices
                start_idx = max(0, i - K)
                end_idx = i
                
                # Fill the appropriate part of the history array
                if start_idx < end_idx:  # Ensure there's something to copy
                    history[-(end_idx - start_idx):] = path[start_idx:end_idx]
                
                # The next item to predict
                if i < N:
                    next_item = path[i]
                    history_batch.append(history)
                    target_batch.append(next_item)
                    path_id_batch.append(path_id)  # Append the path ID
                    end_of_path_batch.append(i == N-1)
                else:
                    raise ValueError('Why did it end up here? There should always be a corresponding target.')
        
        # Convert lists to numpy arrays
        history_batch = np.array(history_batch)
        target_batch = np.array(target_batch)
        path_id_batch = np.array(path_id_batch)
        end_of_path_batch = np.array(end_of_path_batch)
        
        return history_batch, target_batch, path_id_batch, end_of_path_batch


    def _get_stroke_prototypes(self, traj, stroke_ids, stroke_prototype_kind):
        """Returns an encoded prototype representation for all strokes."""
        prototypes = []
        infos = []

        unique_stroke_ids = np.unique(stroke_ids)

        tot_lengths = 0

        stroke_order_check = []  # temp: make sure order of path fetching is the same for different methods, as you want matching items in the same i-th position

        for i in unique_stroke_ids:
            if i == -1:  # -1 is assigned to fake vectors 
                continue

            curr_length = stroke_ids[stroke_ids == i].shape[0]
            starting_index = np.argmax(stroke_ids == i)

            stroke = np.copy(traj[starting_index:starting_index+curr_length, :])
            tot_lengths += stroke.shape[0]

            prototype, encod_info = self._get_stroke_encoding(stroke, kind=stroke_prototype_kind)
            prototypes.append(prototype)
            infos.append(encod_info)

            stroke_order_check.append(i)

        assert tot_lengths == stroke_ids.shape[0]

        return np.array(prototypes), infos, np.array(stroke_order_check, dtype=int)


    def _get_stroke_encoding(self, stroke, kind):
        """Return some encoding of each stroke to distinguish it
           among other strokes.
            
            stroke : (N, outdim)
            kind : str, prototype kind
        """

        if kind == '3d_bboxes':
            # Stroke as 3D bounding box (=6D encoding)
            points = stroke[:, :3]  # consider (x,y,z) only
            canonical_3dbbox = get_3dbbox(points)  # xmin, xmax, ymin, ymax, zmin, zmax
            center_3dbbox = get_center_of_3dbbox(canonical_3dbbox)  # [x,y,z]
            sizes_3dbbox = get_sizes_of_3dbbox(canonical_3dbbox)  # [sqrt(w), sqrt(h), sqrt(z)]
            # TODO: use config.sqrt_output_bbox_sizes to transform the sizes

            return np.array(center_3dbbox + sizes_3dbbox) , {'canonical_3dbbox': canonical_3dbbox}

        elif kind == 'start_of_path_token':
            # Stroke as concatenation of X initial poses.
            assert stroke.shape[-1] == self.outdim, 'stroke is expected in point-format, not segment-format. This is the convention for the start_of_path token length'
            n_starting_points = self.config['start_of_path_token_length']

            if stroke.shape[0] < n_starting_points:
                # Handle short strokes that are shorted than the desired n_starting_points.
                # Try considering a duplicated half-length sequence
                assert n_starting_points % 2 == 0
                assert stroke.shape[0] >= n_starting_points // 2, f'The current stroke is less than {n_starting_points//2} points long (i.e. its {stroke.shape[0]} points long), which is half of the stroke prototype length required already. We cannot create a prototype out of this stroke.'

                # consider half the initial points required
                points = stroke[:n_starting_points//2, :]

                # duplicate them
                points = np.repeat(points[None, ...], 2, axis=0).reshape(-1, points.shape[-1])
                points = points.reshape(-1)  # flatten
            else:
                points = stroke[:n_starting_points, :]
                points = points.reshape(-1)  # flatten

            return points, {}
        else:
            raise ValueError(f'stroke prototype kind {kind} is not valid.')


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

        # if self.multi_root:
        if self.multi_root or self.data_scale_factor is not None:  # temp check for joint-category training or explicitly defined factor to avoid changing name for previously saved data
            if self.normalization == 'per-dataset':
                norm += str(round(self.dataset_mean_max_distance, 2))

        # Wether pc is subsample online or just once and saved
        pc_online_subsampling = 'True' if 'pc_online_subsampling' in self.augmentations else 'False'

        # Traj sampling different than normal (e.g. traj_sampling_v2, ...)
        traj_sampling_flag = ''
        if self.config['traj_with_equally_spaced_points']:
            if self.config['equal_in_3d_space']:
                # v3
                traj_sampling_flag = '_TrajSamplingV3'+str(self.config['equal_spaced_points_distance'])
            else:
                # v2
                traj_sampling_flag = '_TrajSamplingV2'+str(self.config['equal_spaced_points_distance'])
            

        filename = "preprocessed_"+extras+ \
                   "_weightOrient"+weightOrient+ \
                   "_pcPoints"+pc_points+ \
                   "_normalization"+norm+ \
                   "_pcOnlineSub"+pc_online_subsampling+ \
                   traj_sampling_flag+ \
                   ".npz"

        return filename



class Paintnet_ODv1_CollateBatch(object):
    """Aggregates individual items returned by PaintNetODv1Dataloader into a mini-batch"""


    def __init__(self, config):
        self.config = config
        self.load_extra_data = config['load_extra_data']
        self.out_prototypes = config.out_prototypes

        if self.out_prototypes is not None:
            self.max_n_strokes = config.max_n_strokes  # max number of strokes across GT samples
            assert self.out_prototypes >= self.max_n_strokes, 'Less bboxes are predicted than the max number of bboxes present in GT. is this desired? If so, remove the assert' 

    def __call__(self, data):
        """Prepare batch of data.

            `data` is a list of <object> of len `batch_size`,
            where <object> is whatever the data.Dataset class returns.

            In my case, I return the dict `item`.
        """
        # Handle batch of object point-clouds
        point_cloud = torch.stack([torch.as_tensor(d['point_cloud'], dtype=torch.float) for d in data])

        # Handle batch of trajectories
        if self.config['traj_with_equally_spaced_points']:
            # Handle mini-batches that have a different number of GT traj samples
            max_n_segments = np.max([d['traj'].shape[0] for d in data])
            max_n_points = np.max([d['traj_as_pc'].shape[0] for d in data])

            traj = torch.stack([torch.as_tensor(self.add_fake_vectors_v2(d['traj'], total_needed=max_n_segments), dtype=torch.float) for d in data])
            traj_as_pc = torch.stack([torch.as_tensor(self.add_fake_vectors_v2(d['traj_as_pc'], total_needed=max_n_points), dtype=torch.float) for d in data])

            stroke_ids = torch.stack([torch.as_tensor(self.add_fake_values_v2(d['stroke_ids'], total_needed=max_n_segments, fake_value=-1), dtype=torch.float) for d in data])
            stroke_ids_as_pc = torch.stack([torch.as_tensor(self.add_fake_values_v2(d['stroke_ids_as_pc'], total_needed=max_n_points, fake_value=-1), dtype=torch.float) for d in data])

        else:
            traj = torch.stack([torch.as_tensor(d['traj'], dtype=torch.float) for d in data])  # batch of traj
            traj_as_pc = torch.stack([torch.as_tensor(d['traj_as_pc'], dtype=torch.float) for d in data])  # batch of traj_as_pc
            stroke_ids = torch.stack([torch.as_tensor(d['stroke_ids'], dtype=torch.float) for d in data])  # batch of stroke_ids
            stroke_ids_as_pc = torch.stack([torch.as_tensor(d['stroke_ids_as_pc'], dtype=torch.float) for d in data])  # batch of stroke_ids_as_pc


        stroke_prototypes = None
        if 'stroke_prototypes' in self.load_extra_data:
            # Pad and stack stroke prototypes (e.g. 3D bboxes)
            padded_stroke_prototypes = self.add_fake_vectors([d['stroke_prototypes'] for d in data], total_needed=self.max_n_strokes)
            stroke_prototypes = torch.stack([torch.as_tensor(padded_stroke_proto, dtype=torch.float) for padded_stroke_proto in padded_stroke_prototypes])  # batch of stroke_prototypes


        stacked_segments_per_stroke, unstacked_segments_per_stroke, batch_max_num_segments, stacked_points_per_stroke = None, None, None, None
        if 'segments_per_stroke' in self.load_extra_data:
            # Handle mini-batches that have dynamic padding for segments_per_stroke (it's hard to say a prior the max number of segments a stroke can have)
            # Get max num of segments per stroke within batch
            batch_max_num_segments = np.max([d['max_num_segments'] for d in data])
            batch_segments_per_stroke = [d['segments_per_stroke'] for d in data]
            batch_segments_per_stroke = [self.add_fake_vectors(segments_per_stroke, total_needed=batch_max_num_segments) for segments_per_stroke in batch_segments_per_stroke]

            # stack segments_per_stroke of all samples along batch dim
            stacked_segments_per_stroke = []
            unstacked_segments_per_stroke = []
            for segments_per_stroke in batch_segments_per_stroke:
                stacked_segments_per_stroke += segments_per_stroke
                unstacked_segments_per_stroke.append(torch.as_tensor(np.array(segments_per_stroke), dtype=torch.float))
            stacked_segments_per_stroke = torch.stack([torch.as_tensor(d, dtype=torch.float) for d in stacked_segments_per_stroke])

            # Handle mini-batches of points_per_stroke, just like segments_per_stroke
            if 'out_points_per_stroke' in self.config and self.config['out_points_per_stroke'] is not None:
                batch_points_per_stroke = [d['points_per_stroke'] for d in data]
                batch_points_per_stroke = [self.add_fake_vectors(points_per_stroke, total_needed=self.config.out_points_per_stroke) for points_per_stroke in batch_points_per_stroke]
                stacked_points_per_stroke = []
                for points_per_stroke in batch_points_per_stroke:
                    stacked_points_per_stroke += points_per_stroke
                stacked_points_per_stroke = torch.stack([torch.as_tensor(d, dtype=torch.float) for d in stacked_points_per_stroke])


        # Handle autoregressive substroke v1
        stacked_segments_per_substroke, stacked_segments_per_init_substroke = None, None
        if 'history_of_segments_per_stroke_v1' in self.load_extra_data:
            # Handle segments_per_SUBstroke (sampling of a substroke for autoregressive methods)
            # stack segments_per_SUBstroke of all samples along batch dim (used for autoregressive predictions)
            batch_segments_per_substroke = [d['segments_per_substroke'] for d in data]
            batch_segments_per_init_substroke = [d['segments_per_init_substroke'] for d in data]
            stacked_segments_per_substroke = []
            stacked_segments_per_init_substroke = []
            for segments_per_substroke, segments_per_init_substroke in zip(batch_segments_per_substroke, batch_segments_per_init_substroke):
                stacked_segments_per_substroke += segments_per_substroke
                stacked_segments_per_init_substroke += segments_per_init_substroke
            stacked_segments_per_substroke = torch.stack([torch.as_tensor(d, dtype=torch.float) for d in stacked_segments_per_substroke])
            stacked_segments_per_init_substroke = torch.stack([torch.as_tensor(d, dtype=torch.float) for d in stacked_segments_per_init_substroke])


        # Handle autoregressive substroke v2
        strokewise_history_batch, strokewise_target_batch, strokewise_stroke_ids_batch, strokewise_sample_ids_batch, strokewise_end_of_path_batch, = None, None, None, None, None
        if 'history_of_segments_per_stroke_v2' in self.load_extra_data:
            strokewise_history_batch = torch.cat([torch.as_tensor(d['strokewise_history_batch'], dtype=torch.float) for d in data], dim=0)  # batch of strokewise_history_batch
            strokewise_target_batch = torch.cat([torch.as_tensor(d['strokewise_target_batch'], dtype=torch.float) for d in data], dim=0)  # batch of strokewise_target_batch
            strokewise_stroke_ids_batch = torch.cat([torch.as_tensor(d['strokewise_stroke_ids_batch'], dtype=torch.int) for d in data], dim=0)  # batch of strokewise_stroke_ids_batch
            strokewise_sample_ids_batch = torch.cat([(torch.ones(len(d['strokewise_stroke_ids_batch']))*i).int() for i, d in enumerate(data)], dim=0)
            strokewise_end_of_path_batch = torch.cat([torch.as_tensor(d['strokewise_end_of_path_batch'], dtype=torch.int) for d in data], dim=0)  # batch of strokewise_end_of_path_batch


        stroke_masks = None
        if 'stroke_masks' in self.load_extra_data:
            stroke_masks = [torch.as_tensor(d['stroke_masks'], dtype=torch.int64) for d in data]


        dirname = [d['dirname'] for d in data]
        n_strokes = [d['n_strokes'] for d in data]


        batch = {
            'point_cloud': point_cloud,
            'traj': traj,
            'traj_as_pc': traj_as_pc,
            'stacked_segments_per_stroke': stacked_segments_per_stroke,
            'stacked_points_per_stroke': stacked_points_per_stroke,
            'unstacked_segments_per_stroke': unstacked_segments_per_stroke,
            'stacked_segments_per_substroke': stacked_segments_per_substroke,
            'stacked_segments_per_init_substroke': stacked_segments_per_init_substroke,
            'strokewise_history_batch': strokewise_history_batch,
            'strokewise_target_batch': strokewise_target_batch,
            'strokewise_stroke_ids_batch': strokewise_stroke_ids_batch,
            'strokewise_sample_ids_batch': strokewise_sample_ids_batch,
            'strokewise_end_of_path_batch': strokewise_end_of_path_batch,
            'max_num_segments': batch_max_num_segments,
            'stroke_ids': stroke_ids,
            'stroke_ids_as_pc': stroke_ids_as_pc,
            'stroke_masks': stroke_masks,
            'stroke_prototypes': stroke_prototypes,
            'dirname': dirname,
            'n_strokes': n_strokes
        }

        return batch


    def add_fake_vectors(self, list_of_vectors, total_needed):
        """Add fake vectors so that we always
        have a total of total_needed vectors.

            A fake value of -100 is used for filling
            the fake vectors

            list_of_vectors : list [(N1, D), (N2, D), (N3, D)]

            ---
            returns
                list of vectors : list [(total_needed, D), (total_needed, D), ...]
        """        
        fake_value = -100

        vectors_dims = np.array([vec.shape[-1] for vec in list_of_vectors])
        assert np.all(vectors_dims == vectors_dims[0]), 'some vectors have different dimensionality than others.'

        vectors_dim = vectors_dims[0]  # num of dims per vector

        out_list_of_vectors = []
        for vec in list_of_vectors:
            # vec : (N1, D)
            assert vec.ndim == 2

            num_of_real_vectors = vec.shape[0]
            num_of_fake_vectors = total_needed - num_of_real_vectors

            if num_of_fake_vectors > 0:
                fake_vectors = fake_value*np.ones((num_of_fake_vectors, vectors_dim))
                out_list_of_vectors.append(np.concatenate((vec, fake_vectors), axis=0))
            else:
                out_list_of_vectors.append(vec)

        return out_list_of_vectors


    def add_fake_vectors_v2(self, matrix, total_needed):
        """As add_fake_vectors, but takes in input one single sequence of vectors

            matrix: [N, D]

        Returns
            matrix: [total_needed, D]

        """
        assert matrix.ndim == 2
        fake_value = -100
        N, D = matrix.shape

        n_fakes = total_needed - N

        if n_fakes > 0:
            fake_vectors = fake_value*np.ones((n_fakes, D))
            return np.concatenate((matrix, fake_vectors), axis=0)
        else:
            return matrix


    def add_fake_values_v2(self, points, total_needed, fake_value=-100):
        """Add fake values at the end of `points` vector
            v2: follows a different convention from add_fake_vectors,
                as it doesn't take a list in input, but just a single vector.

        Params
            points: [N,]

        Returns
            points: [total_needed,]
        """
        assert points.ndim == 1

        n_fakes = total_needed - points.shape[0]

        if n_fakes > 0:
            return np.concatenate((points, np.repeat(fake_value, n_fakes)))
        else:
            return points


def from_points_to_fixedlength_strokes(traj, stroke_points):
    """From points format to strokes format,
        assuming fixed-length strokes, no padding and no stroke_ids are taking into account because of this

        traj: (B, N, outdim)
        stroke_ids: (B, N)
            
            N is aggregated number of points:
                - if stroke_pred: N=n_strokes*stroke_points
                - else:           N=traj_points
        ---
        returns
            traj: (B, n_strokes, stroke_points*outdim)
    """
    B, N, outdim = traj.shape
    strokes = traj.reshape(B, -1, stroke_points, outdim).reshape(B, -1, stroke_points*outdim)  # .reshape(-1, stroke_points, outdim)

    return strokes


def get_vectors_per_stroke(traj, stroke_ids):
    """From stacked segments/points, return list
     of varying-length strokes encoded as (num_vectors, D)

        traj: (N, D)
        stroke_ids: (N,)

    ---
    returns

        out_strokes: [(N1, D), (N2, D), ...], N = N1 + N2 + ...
                     note: padded vectors are also discarded,
                     so N >= N1 + N2 + ...
    """

    unique_stroke_ids = np.unique(stroke_ids)

    stroke_order_check = []  # temp: make sure order of path fetching is the same for different methods, as you want matching items in the same i-th position

    out_strokes = []
    for i in unique_stroke_ids:
        if i == -1:  # -1 is assigned to fake vectors
            continue

        curr_stroke = traj[stroke_ids == i].copy()
        out_strokes.append(curr_stroke)

        stroke_order_check.append(i)

    return out_strokes, np.array(stroke_order_check, dtype=int)