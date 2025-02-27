"""Auxiliary script to also apply the postprocessing steps to the predictions of all methods.
    Then, all predictions can be saved to file for later computations of the metrics.

    Example:
        python standalone/from_pred_to_offline_v2.py --run <run_dir>/test 
        python standalone/from_pred_to_offline_v2.py --run <run_dir>/test --postprocess --segments_filtering_only
        python standalone/from_pred_to_offline_v2.py --run <run_dir>/test --postprocess --no_interpolation

        FULL POSTPROCESS:
            python standalone/from_pred_to_offline_v2.py --run <run_dir>/test --postprocess
"""
import sys
import os
import argparse
import glob
import inspect
import numpy as np
import omegaconf
import pdb
import time

import torch

# Trick to import paintnet_utils from parent dir
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils import orient_in
from utils.config import load_config
from utils.disk import get_dataset_meshes_paths
from utils.postprocessing import from_strokewise_to_pointwise, process_stroke_segments, process_pred_stroke_masks_to_stroke_ids, postprocess_strokewise_predictions_into_strokes, postprocess_sop_predictions
from metrics_handler import MetricsHandler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, help='run path', required=True)
    parser.add_argument('--output_dir', default='postprocess_predictions', type=str, help='output dir')
    parser.add_argument('--split',      default='test', type=str, help='<train,test>')
    parser.add_argument('--force_overwrite', default=False, action='store_true', help='if set, overwrite previously saved .csv files.')

    parser.add_argument('--postprocess', default=False, action='store_true', help='if True, postprocess predictions')
    parser.add_argument('--segments_filtering_only', default=False, action='store_true', help='when postprocessing, only perform segments filtering.')
    parser.add_argument('--skip_segments_filtering', default=False, action='store_true', help='Skip segment filtering')
    parser.add_argument('--no_interpolation', default=False, action='store_true', help='when postprocessing, avoid the final interpolation + smoothing.')
    parser.add_argument('--custom_data_scale_factor', default=None, type=float, help='Denorm trajectories with a custom data scale factor (use when tested joint training models on target categories)')


    # Thresholds
    parser.add_argument('--mask_confidence_threshold', default=0.5, type=float, help='Mask confidence threshold for MaskPlanner predictions')
    parser.add_argument('--stroke_confidence_threshold', default=0.5, type=float, help='Confidence threshold for retaining the correct num of paths for the Path-wise and the Autoregressive baselines.')
    parser.add_argument('--point_confidence_threshold', default=0.5, type=float, help='Confidence threshold for discarding end-of-path points for the Path-wise and Autoregressive baselines.')

    return parser.parse_args()

args = parse_args()


def main():
    assert os.path.isdir(args.run)
    assert args.split in ['train', 'test'], 'split must be either train or test'

    filenames = glob.glob(os.path.join(args.run, '*_'+str(args.split)+'_*.npy'))
    assert type(filenames) == list and len(filenames) > 0, f'No results to be rendered exist in target directory: {args.run}'
    filenames = sorted(filenames)  # Order by glob.glob() is not by filename

    config = load_config(os.path.join(args.run, 'config.yaml'))

    assert config['normalization'] == 'per-dataset', 'Other types of norms are not supported for now'
    assert orient_in(config['extra_data'])[0], 'You cannot simulate trajectories that do not contain normals'

    metrics_handler = MetricsHandler(config=config, metrics=config.eval_metrics)

    # ITERATE OVER EACH BATCH
    for file in filenames:
        data = np.load(file, allow_pickle=True).item()

        dirnames = data.get('dirnames')
        batch_id = data.get('batch')
        suffix = data.get('suffix')
        traj = data.get('traj')
        stroke_ids = data.get('stroke_ids')

        traj_pred = data.get('traj_pred')
        B = len(traj_pred) # traj_pred.shape[0]


        # Postprocess MaskPlanner predictions
        if config['task_name'] == 'MaskPlanner':
            pred_stroke_masks = data.get('pred_stroke_masks')
            stroke_masks_scores = data.get('stroke_masks_scores')
            traj_as_pc = data.get('traj_as_pc')
            n_strokes = data.get('n_strokes')

            # From masks to stroke ids
            stroke_ids_pred = process_pred_stroke_masks_to_stroke_ids(pred_stroke_masks,
                                                                      confidence_scores=stroke_masks_scores,
                                                                      confidence_threshold=args.mask_confidence_threshold
                                                                      )

            if args.postprocess:
                # Post-processing on network predictions: filter, concatenate segments, smoothing, ...
                traj_pred_processed, stroke_ids_pred_processed = process_stroke_segments(traj=traj_pred,
                                                                                         stroke_ids=stroke_ids_pred,
                                                                                         config=config,
                                                                                         segments_filtering_only=args.segments_filtering_only,
                                                                                         no_interpolation=args.no_interpolation)

            if config['lambda_points'] == 1:
                assert not args.postprocess, 'Point-wise baseline cannot make use of postprocessing (no concatenation)'
                

        # Now you can save the data in the format you like.
        pdb.set_trace()
        

    return

if __name__ == '__main__':
    main()