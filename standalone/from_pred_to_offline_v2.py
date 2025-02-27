"""Convert predictions of a RUN into simulator (i.e. offline) format for getting spray painting feedback.
    Compatible with: train_maskplanner.py
    
    Note:
        it always outputs both the ground-truth and predictions together, in two separate sub-directories

    Example:
        python standalone/from_pred_to_offline_v2.py --output_dir predicted_programs --run <run_dir> 

        python standalone/from_pred_to_offline_v2.py --output_dir predicted_programs --run <run_dir> --postprocess --segments_filtering_only
    
        JOINT-CATEGORY TRAINING:
            python standalone/from_pred_to_offline_v2.py --output_dir predicted_programs --run <run_dir> --postprocess --segments_filtering_only --custom_data_scale_factor <VAL>
"""
import sys
import os
import argparse
import glob
import inspect
import numpy as np
import omegaconf
import pdb

from scipy.spatial.transform import Rotation as Rot
try:
    import pyvista as pv  
except ImportError:
    logging.warn('Unable to import pyvista package. visualizations won\'t be available. Run `pip install pyvista`')
    pass

# Trick to import paintnet_utils from parent dir
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils import orient_in, get_root_of_dir
from utils.config import load_config, load_config_json
from utils.disk import get_dataset_meshes_paths, get_dataset_downscale_factor, save_traj_file
from utils.pointcloud import get_dim_traj_points, get_mean_mesh, remove_padding, denormalize_traj, remove_padding_v2, from_seq_to_pc_v2
from utils.postprocessing import from_strokewise_to_pointwise, from_stroke_ids_to_masks, process_stroke_segments, process_pred_stroke_masks_to_stroke_ids, select_top_bboxes, handle_end_of_sequence, permute_and_align_stroke_ids_for_visualization
from utils.visualize import visualize_traj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, help='run path', required=True)
    parser.add_argument('--output_dir', default='predicted_programs', type=str, help='output dir')
    parser.add_argument('--split',      default='test', type=str, help='<train,test>')
    parser.add_argument('--force_overwrite', default=False, action='store_true', help='if set, overwrite previously saved .csv files.')

    parser.add_argument('--postprocess', default=False, action='store_true', help='if True, postprocess predictions')
    parser.add_argument('--segments_filtering_only', default=False, action='store_true', help='when postprocessing, only perform segments filtering.')
    parser.add_argument('--no_interpolation', default=False, action='store_true', help='when postprocessing, avoid the final interpolation + smoothing.')
    parser.add_argument('--custom_data_scale_factor', default=None, type=float, help='Denorm trajectories with a custom data scale factor (use when tested joint training models on target categories)')

    parser.add_argument('--debug', default=False, action='store_true', help='Avoid actually saving the trajectories if True')
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

    dataset_paths = get_dataset_meshes_paths(config['dataset'])
    if isinstance(config['dataset'], list) or isinstance(config['dataset'], omegaconf.listconfig.ListConfig):  # Handles joint category training
        category = '-'.join(config['dataset'])
    else:
        category = config['dataset']

    outdim = get_dim_traj_points(config['extra_data'])

    # Create target dirs
    gt_save_dir, pred_save_dir = get_output_dirnames(category, config)

    try:
        os.makedirs(os.path.join(args.output_dir))
    except OSError as error:
        pass
    try:
        os.makedirs(gt_save_dir)
        os.makedirs(pred_save_dir)
    except OSError as error:
        if not args.force_overwrite:
            raise ValueError(f"Output dir already exists at: {gt_save_dir} OR {pred_save_dir}")
    print('save dir (ground truth):', gt_save_dir)
    print('save dir (predictions):', pred_save_dir)


    # Iterate over predictions and turn them into offline format
    for file in filenames:
        data = np.load(file, allow_pickle=True).item()

        dirnames = data.get('dirnames')
        batch_id = data.get('batch')
        suffix = data.get('suffix')
        
        traj = data.get('traj')
        stroke_ids = data.get('stroke_ids')

        traj_pred = data.get('traj_pred')
        B = len(traj_pred) # traj_pred.shape[0]

        if config['task_name'] == 'MaskPlanner':
            pred_stroke_masks = data.get('pred_stroke_masks')
            stroke_masks_scores = data.get('stroke_masks_scores')

            # From masks to stroke ids
            stroke_ids_pred = process_pred_stroke_masks_to_stroke_ids(pred_stroke_masks, confidence_scores=stroke_masks_scores)

            if args.postprocess:
                # Post-processing on network predictions: filter, concatenate segments, smoothing, ...
                traj_pred, stroke_ids_pred = process_stroke_segments(traj=traj_pred, stroke_ids=stroke_ids_pred, config=config, segments_filtering_only=args.segments_filtering_only, no_interpolation=args.no_interpolation)



        # GT postprocessing
        if args.postprocess:
            traj, stroke_ids = process_stroke_segments(traj=traj, stroke_ids=stroke_ids, config=config, skip_segments_filtering=True, segments_filtering_only=args.segments_filtering_only, no_interpolation=args.no_interpolation)

        for b in range(B):
            curr_traj = traj[b].copy()
            curr_stroke_ids = stroke_ids[b].copy()
            curr_traj_pred = traj_pred[b].copy()
            curr_stroke_ids_pred = stroke_ids_pred[b].copy()

            # Remove padding in GT trajectories
            curr_traj, curr_stroke_ids = remove_padding_v2(curr_traj, curr_stroke_ids)

            # From segment-format to point-wise format
            curr_traj, curr_stroke_ids = from_seq_to_pc_v2(curr_traj, curr_stroke_ids, config=config)
            curr_traj_pred, curr_stroke_ids_pred = from_seq_to_pc_v2(curr_traj_pred, curr_stroke_ids_pred, config=config)

            # Sanity checks
            assert curr_traj.ndim == 2 and curr_traj.shape[-1] == outdim and curr_traj.shape[0] == curr_stroke_ids.shape[0], f'curr_traj.shape: {curr_traj.shape} \n curr_stroke_ids.shape: {curr_stroke_ids.shape}'
            assert curr_traj_pred.ndim == 2 and curr_traj_pred.shape[-1] == outdim and curr_traj_pred.shape[0] == curr_stroke_ids_pred.shape[0]

            # Denormalize trajectories for executing in original mesh space
            assert get_root_of_dir(dirnames[b], dataset_paths) is not None, f'{dirnames[b]} is not a subdir of any dir: {dataset_paths}'
            meshfile = os.path.join(get_root_of_dir(dirnames[b], dataset_paths), dirnames[b], (dirnames[b]+'.obj'))

            # todo: detect custom_data_scale_factor automatically by looking at config and checking for data_scale_factor is not None
            curr_traj = denormalize_traj(curr_traj, meshfile, config, normalization='per-dataset', dataset=category, custom_data_scale_factor=args.custom_data_scale_factor)
            curr_traj_pred = denormalize_traj(curr_traj_pred, meshfile, config, normalization='per-dataset', dataset=category, custom_data_scale_factor=args.custom_data_scale_factor)

            # Sanity shape checks
            # print('dirname:', dirnames[b])
            # print('curr traj:', curr_traj.shape)
            # print('curr traj pred:', curr_traj_pred.shape)
            # print('curr curr_stroke_ids:', curr_stroke_ids.shape)
            # print('curr curr_stroke_ids_pred:', curr_stroke_ids_pred.shape)

            # Visualization check
            # plotter = pv.Plotter(shape=(1, 2), window_size=(1920,1080))
            # visualize_traj(curr_traj, stroke_ids=curr_stroke_ids, trajc='orange', plotter=plotter, index=(0,0), extra_data=config['extra_data'], text='GT')
            # visualize_traj(curr_traj_pred, stroke_ids=curr_stroke_ids_pred, plotter=plotter, index=(0,1), extra_data=config['extra_data'], text='Pred')
            # plotter.show()

            # From normals to Euler angles (1 arbitrary DoF to define)
            for i in range(curr_traj.shape[0]):
                curr_traj[i, 3:] = from_normals_to_euler_angles(curr_traj[i, 3:].copy())

            for i in range(curr_traj_pred.shape[0]):
                curr_traj_pred[i, 3:] = from_normals_to_euler_angles(curr_traj_pred[i, 3:].copy())

            # Augment trajectories with 7th entry being the stroke id
            curr_traj = np.append(curr_traj, curr_stroke_ids[:, None], axis=-1)
            curr_traj_pred = np.append(curr_traj_pred, curr_stroke_ids_pred[:, None], axis=-1)

            #### Compute spatial length spanned by GT for a particular sample
            # n_strokes = data.get('n_strokes')
            # if n_strokes[b] == 41:  # or 
            # if dirnames[b] == 'box_h775_w500_d375.0_sh2.0_sv3.0'
            # if dirnames[b] == '232_cube_1000_719_1396':
            # if dirnames[b] == '101_wr1fr_1':
            #     print('total spatial length (mm):', compute_total_path_length(curr_traj))
            #     pdb.set_trace()

            if not args.debug:
                save_traj_file(curr_traj_pred, os.path.join(pred_save_dir, dirnames[b]+'.txt'), kind='euler')
                save_traj_file(curr_traj, os.path.join(gt_save_dir, dirnames[b]+'.txt'), kind='euler')

    return


def from_normals_to_euler_angles(target):
    """Computes the "best" euler angles given the current target normal vector.

    There's a free parameter going from compact normal repr. to full rigid body
    orientation repr., so the algorithm sets it to zero or some constant value by itself."""
    assert target.ndim == 1 and target.shape[0] == 3
    

    # source = np.eye(3, dtype=int)
    # x = target.reshape(-1)
    # u = np.array([1,0,0])
    # y = np.cross(target.reshape(-1), u)
    # y /= np.linalg.norm(y)
    # z = np.cross(target.reshape(-1), y)
    # z /= np.linalg.norm(z)
    # target = np.vstack((x,y,z))

    source = np.array([[1,0,0]])
    target = target.reshape(1, -1)

    """
        See more at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html
            e.g. "if a single vector is given for a and b, in which the shortest
            distance rotation that aligns b to a is returned."
    """
    rot = Rot.align_vectors(target, source)
    eulers = rot[0].as_euler('yzx', degrees=True)

    ordered_eulers = [eulers[2], eulers[0], eulers[1]]


    test_rot = Rot.from_euler('yzx', [ordered_eulers[1], ordered_eulers[2], ordered_eulers[0]], degrees=True)
    e1 = np.array([1,0,0])
    normal = test_rot.apply(e1)

    # print('target:', target.reshape(-1))
    # print('reconstruction:', normal)
    # print('is equal:', np.isclose(target.reshape(-1), normal))
    # print('Root MSE:', rot[1], 'Distance:', np.linalg.norm(target.reshape(-1)-normal))
    # print('Root MSE:', rot[1], 'Distance:', np.linalg.norm(rot[0].apply(e1) - target.reshape(-1)))

    if np.linalg.norm(rot[0].apply(e1) - target.reshape(-1)) > 0.00001:
    # if np.linalg.norm(rot[0].apply(e1) - target.reshape(-1)) > 0.25:
        raise ValueError('Norm higher than expected:', np.linalg.norm(rot[0].apply(e1) - target.reshape(-1)))

    return ordered_eulers


def compute_total_path_length(input_traj):
    total_length = 0.0
    # Loop through each unique path id (assumed in the last column)
    for pid in np.unique(input_traj[:, 6]):
        # Extract waypoints for the current path
        traj = input_traj[input_traj[:, 6] == pid]
        # If there's only one waypoint, skip since no distance to accumulate
        if traj.shape[0] < 2:
            continue
        # Compute differences between consecutive waypoints (columns 0,1,2 are x,y,z)
        diffs = np.diff(traj[:, :3], axis=0)
        # Compute Euclidean distances for these differences
        distances = np.linalg.norm(diffs, axis=1)
        # Sum distances for the current path and add to the total length
        total_length += np.sum(distances)
    
    return total_length


def get_output_dirnames(category, config):
    run_name = os.path.basename(args.run)

    suffix = str(run_name)+ \
             ('_postprocess' if args.postprocess else '') +\
             ('_SegmentsFilteringOnly' if args.postprocess and args.segments_filtering_only else '') +\
             ('_NoInterpolation' if args.postprocess and args.no_interpolation else '')

    gt_save_dir = os.path.join(args.output_dir, str(category)+'_GT_'+str(suffix)+'_Lambda'+str(config['lambda_points']))
    pred_save_dir = os.path.join(args.output_dir, str(category)+'_PRED_'+str(suffix))

    return gt_save_dir, pred_save_dir


if __name__ == '__main__':
    main()