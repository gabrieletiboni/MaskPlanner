"""Utility script to render results already saved as .npy locally (first batch only)

It renders:
    1. multiple side-by-side figures with GT vs predictions
    2. a single batch figure with the predictions only

    NOTE: it only renders the first batch of <train,test> splits

    Examples:
        python render_results.py --run runs/XXXXXX
        python render_results.py --run runs/XXXXXX --allbatches --save_n 12
        python render_results.py --run runs/XXXXXX --display [--sidebyside]
        
        (with post-processing concatenation)
            python render_results.py --run runs/XXXXXX --with_postprocess
            python render_results.py --run runs/XXXXXX --postprocess

        (with alignment of stroke colors between pred and gt)
            python render_results.py [...] --align_stroke_ids

"""
import pdb
import argparse
import glob
import logging
import os

import numpy as np
import omegaconf
from tqdm import tqdm
import matplotlib

from utils import create_dirs
from utils.config import load_config, load_config_json
from utils.disk import get_dataset_downscale_factor, get_dataset_meshes_paths
from utils.pointcloud import from_pc_to_seq, from_seq_to_pc, get_max_distance, remove_padding_v2
from utils.visualize import get_list_of_colors, remove_padding_from_vectors, visualize_sops, visualize_boxes, visualize_mesh_traj, visualize_mesh_traj_animated, visualize_traj, visualize_mesh_v2
from utils.postprocessing import from_strokewise_to_pointwise, from_stroke_ids_to_masks, process_stroke_segments, process_pred_stroke_masks_to_stroke_ids, select_top_bboxes, handle_end_of_sequence, permute_and_align_stroke_ids_for_visualization

try:
    import pyvista as pv  
except ImportError:
    logging.warn('Unable to import pyvista package. visualizations won\'t be available. Run `pip install pyvista`')
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',        default=None, type=str, help='Run directory')
    parser.add_argument('--save_n',     default=4, type=int, help='Save <save_n> side-by-side renders')
    parser.add_argument('--allbatches', default=False, action='store_true', help='If True, render all batches instead of just the first one')
    parser.add_argument('--split',      default=None, type=str, help='<train,test>')
    parser.add_argument('--nrows',      default=4, type=int, help='number of rows for batch visualization')
    parser.add_argument('--ncols',      default=8, type=int, help='number of cols for batch visualization')

    parser.add_argument('--display',    default=False, action='store_true', help='Display renders rather than saving them')
    parser.add_argument('--sidebyside', default=False, action='store_true', help='Whether to display sidebyside figure vs full batch figure')
    parser.add_argument('--concat',     default=False, action='store_true', help='Display stroke concatenation results [DEPRECATED]')
    parser.add_argument('--tag',        default='', action="store", help='Specify runs to render by tag')
    parser.add_argument('--video',      action="store_true", help='Create an animation of the strokes')
    parser.add_argument('--lines',      action="store_true", help='Whether to render lines that connect segments')

    parser.add_argument('--with_postprocess', default=False, action='store_true', help='if True, render results with post-processing concatenation as well.')
    parser.add_argument('--postprocess',      default=False, action='store_true', help='if True, ONLY render results with post-processing concatenation')

    parser.add_argument('--align_stroke_ids', default=False, action='store_true', help='If set, align predicted stroke ids so that close strokes are plot with the same color.')

    return parser.parse_args()

args = parse_args()


def main():
    assert args.run is not None and os.path.isdir(args.run), f'Run dir does not exist or is not valid: {args.run}'

    if os.path.isfile(os.path.join(args.run, 'config.json')):  # Retro-Compatibility
        config = load_config_json(os.path.join(args.run, 'config.json'))
    else:
        config = load_config(os.path.join(args.run, 'config.yaml'))

    if 'task_name' not in config:
        raise ValueError()

    # First load the configuration, then change the path
    if args.concat:
        args.run = os.path.join(args.run, 'concat')
        args.sidebyside = True

    assert config['normalization'] != 'none', 'rendered results are not compatible as of right now with norm=none, but its a easy fix if you wanna display the mesh.obj'

    batch_mask = 'batch*' if args.allbatches else 'batch0'
    # TODO handle tag_mask empty
    
    if args.split is None:  # Render 1st batch of both train and test
        filenames =  glob.glob(os.path.join(args.run, '*_train_'+batch_mask+"*"+args.tag+".npy"))
        filenames += glob.glob(os.path.join(args.run, '*_test_'+batch_mask+"*"+args.tag+".npy"))
    else:
        filenames = glob.glob(os.path.join(args.run, ('*_'+str(args.split))+'_'+batch_mask+"*"+args.tag+".npy"))

    if len(filenames) == 0:  # Retro-Compatibility
        filenames =  glob.glob(os.path.join(args.run, 'results_train_'+batch_mask))
        filenames += glob.glob(os.path.join(args.run, 'results_test_'+batch_mask))

    assert type(filenames) == list and len(filenames) > 0, f'No results to be rendered exist in target directory: {args.run}'

    dataset_paths = get_dataset_meshes_paths(config['dataset'])

    joint_training = False
    if isinstance(config['dataset'], list) or isinstance(config['dataset'], omegaconf.listconfig.ListConfig):  # Handles joint category training
        category = '-'.join(config['dataset'])
        joint_training = True
    else:
        category = config['dataset']

    if args.with_postprocess:
        postprocess_flags = [False, True]  # render both with and without post-processing
    elif args.postprocess:
        postprocess_flags = [True]  # render with post-processing only
    else:
        postprocess_flags = [False]  # render without post-processing

    for postprocess_flag in postprocess_flags:
        for j, file in enumerate(filenames):  # iterate over different batches, train/test, best/last
            basename = os.path.basename(file)
            model_prefix = basename.split('_')[0] if basename.split('_')[0] in ['best', 'last'] else 'unknown'  # prefix for output image file: best, last, unknown
            if args.tag == "":
                basename_noext = os.path.splitext(basename)[0]
                tag_index = basename_noext.find("tsp")
                tag_mask = basename_noext[tag_index:]
            else:
                tag_mask = args.tag if "tsp" in args.tag else f"tsp_{args.tag}"
            data = np.load(file, allow_pickle=True).item()

            if postprocess_flag:
                if config['lambda_points'] == 1:
                    continue

                save_dir = os.path.join(args.run, 'with_postprocess')
                if not args.display:
                    create_dirs(save_dir)
            else:
                save_dir = os.path.join(args.run)

            dirnames = data.get('dirnames')
            traj = data.get('traj')
            stroke_ids = data.get('stroke_ids')
            traj_pred = data.get('traj_pred')
            batch_id = data.get('batch')
            suffix = data.get('suffix')

            stroke_ids_pred = None
            max_n_strokes = None

            bbox_gt = None
            if 'bbox_gt' in data:
                # predicted 3d bboxes
                bbox_gt = data.get('bbox_gt')

                # remove padding from bbox_gt: from stacked array to list 
                bbox_gt = [remove_padding_from_vectors(bbox) for bbox in bbox_gt]

            bbox_pred = None

            if config['task_name'] == 'MaskPlanner':
                pred_stroke_masks = data.get('pred_stroke_masks')
                stroke_masks_scores = data.get('stroke_masks_scores')
                # seg_logits = data.get('seg_logits')
                
                processed_stroke_ids_pred = process_pred_stroke_masks_to_stroke_ids(pred_stroke_masks, confidence_scores=stroke_masks_scores)

                if args.align_stroke_ids:
                    # Get binary stroke masks corresponding to the resulting `processed_stroke_ids_pred`
                    processed_pred_stroke_masks = [from_stroke_ids_to_masks(b_processed_stroke_ids_pred) for b_processed_stroke_ids_pred in processed_stroke_ids_pred]

                    # Align GT stroke_ids 
                    processed_stroke_ids_pred, max_n_strokes = permute_and_align_stroke_ids_for_visualization(processed_pred_stroke_masks,
                                                                                               y_pred=traj_pred,
                                                                                               y=traj,
                                                                                               pred_stroke_ids=processed_stroke_ids_pred,
                                                                                               stroke_ids=stroke_ids)                    


                if postprocess_flag:
                    # Post-processing on network predictions: filter, concatenate segments, smoothing, ...
                    processed_traj_pred, processed_stroke_ids_pred = process_stroke_segments(traj=traj_pred, stroke_ids=processed_stroke_ids_pred, config=config)
                    processed_traj_gt, processed_stroke_ids_gt = process_stroke_segments(traj=traj, stroke_ids=stroke_ids, config=config, skip_segments_filtering=True)
                else:
                    processed_traj_pred = traj_pred.copy()
                    processed_traj_gt, processed_stroke_ids_gt = traj.copy(), stroke_ids.copy()
                

            tour_gt = None
            tour_pred = None

            if 'tour_gt' in data:
                tour_gt = data.get('tour_gt')
            if 'tour_pred' in data:
                tour_pred = data.get('tour_pred')
            if 'stroke_ids_pred' in data:
                stroke_ids_pred = data.get('stroke_ids_pred')

            B = traj.shape[0]  # Batch

            if 'lambda_points' not in config:  # Retro-Compatibility fix
                config['lambda_points'] = 1
            if 'stroke_pred' in config and config['stroke_pred'] == True:
                config['lambda_points'] = config['stroke_points']
                config['traj_points'] = config['n_strokes'] * config['stroke_points']

            ### Side-by-side figures
            if ((not args.display) or (args.display and args.sidebyside)):
                render_iter = 0
                for b in tqdm(range(min(B, args.save_n))):
                    plotter = pv.Plotter(shape=(2 if not args.concat else 3, 4), window_size=(1920,1080), title='Side-by-side', off_screen=(True if not args.display else False))

                    if args.video:
                        plotter.open_movie(filename=os.path.join(args.run, model_prefix+'_'+str(suffix)+'_batch'+str(batch_id)+'_sidebysidemulti_'+str(render_iter)+tag_mask+".mp4"), framerate=8)
                    
                    curr_traj = from_seq_to_pc(traj[b].copy(), extra_data=config['extra_data'])
                    curr_stroke_ids = stroke_ids[b].copy()

                    if config['lambda_points'] == 1:  # temp fix for lambda=1. Could also be done for any lambda just in case.
                        curr_traj, curr_stroke_ids = remove_padding_v2(curr_traj, curr_stroke_ids)

                    if traj_pred is not None:
                        curr_traj_pred = from_seq_to_pc(traj_pred[b].copy(), extra_data=config['extra_data'])
                    else:
                        # use GT traj as pred traj just for visualization
                        curr_traj_pred = curr_traj.copy()

                    if args.concat:
                        curr_tour_gt = tour_gt[b]

                    norm_meshfile = os.path.join(find_parent_of(dirnames[b], dataset_paths), dirnames[b], (dirnames[b]+'_norm.obj'))


                    if config['asymm_overlapping']:
                        # max overlapping in GT segments if asymm_overlapping is True
                        max_ol = config['lambda_points'] - 1
                        curr_traj = from_pc_to_seq(curr_traj, traj_points=curr_traj.shape[0], lambda_points=config['lambda_points'], overlapping=max_ol, extra_data=config['extra_data'])
                    else:
                        curr_traj = from_pc_to_seq(curr_traj, traj_points=curr_traj.shape[0], lambda_points=config['lambda_points'], overlapping=config['overlapping'], extra_data=config['extra_data'])
                    
                    curr_traj_pred_as_pc = curr_traj_pred.copy()
                    curr_traj_pred = from_pc_to_seq(curr_traj_pred, traj_points=curr_traj_pred.shape[0], lambda_points=config['lambda_points'], overlapping=config['overlapping'], extra_data=config['extra_data'])

                    if 'stroke_pred' in config and config['stroke_pred']:
                        stroke_ids_pred = np.repeat(np.arange(6), config['stroke_points'])

                    if 'shelves' in category:
                        camera_pos = [[-4,-4,0], [-5,1,1], [0,0,5], None]
                    else:
                        camera_pos = [[5,0,0], [0, 5, 1e-12], [0,0,5], None]  # second camera_pos was bugged. For some reason adding a small number instead of 0 fixed it.


                    if args.video:
                        render_continue = np.ones((len(camera_pos), 2)).astype(bool)
                        renderers = np.empty((len(camera_pos), 2), dtype=object)
                        
                        
                        for k, cam_pos in enumerate(camera_pos):
                            renderers[k] = [
                                visualize_mesh_traj_animated(norm_meshfile, curr_traj, plotter=plotter, index=(0, k), text='GT', trajc='orange', trajvel=('vel' in config['extra_data']), lambda_points=config['lambda_points'], camera=cam_pos, extra_data=config['extra_data'], stroke_ids=curr_stroke_ids, tour=None),
                                visualize_mesh_traj_animated(norm_meshfile, curr_traj, plotter=plotter, index=(1, k), text=f"GT with TSP ({tag_mask})", trajvel=('vel' in config['extra_data']), lambda_points=config['lambda_points'], camera=cam_pos, extra_data=config['extra_data'], stroke_ids=curr_stroke_ids, tour=curr_tour_gt),
                            # visualize_mesh_traj_animated(norm_meshfile, curr_traj_pred, plotter=plotter, index=(2, k), text='Prediction', trajvel=('vel' in config['extra_data']), lambda_points=config['lambda_points'], camera=cam_pos, extra_data=config['extra_data'], stroke_ids=stroke_ids_pred, tour=tour_pred)
                            ]

                        while render_continue.any():
                            for k in range(len(camera_pos)): 
                                for i, renderer in enumerate(renderers[k]):
                                    if render_continue[k, i]:
                                        try:
                                            next(renderer)
                                        except StopIteration:
                                            render_continue[k, i] = False
                            plotter.write_frame()
                    else:
                        for k, cam_pos in enumerate(camera_pos):

                            if config.task_name == 'MaskPlanner':
                                visualize_mesh_traj(dirnames[b],
                                                    processed_traj_gt[b],
                                                    config=config,
                                                    plotter=plotter,
                                                    index=(0, k),
                                                    text='GT',
                                                    trajc='orange',
                                                    camera=cam_pos,
                                                    stroke_ids=processed_stroke_ids_gt[b],
                                                    force_n_strokes=max_n_strokes[b] if max_n_strokes is not None else None)
                                visualize_mesh_traj(dirnames[b],
                                                    processed_traj_pred[b],
                                                    config=config,
                                                    plotter=plotter,
                                                    index=(1, k),
                                                    text='Prediction',
                                                    camera=cam_pos,
                                                    stroke_ids=processed_stroke_ids_pred[b],
                                                    force_n_strokes=max_n_strokes[b] if max_n_strokes is not None else None)
                            

                            # Visualization for all remaining tasks
                            else:
                                visualize_mesh_traj(dirnames[b], curr_traj, config=config, plotter=plotter, index=(0, k), text='GT', trajc='orange', camera=cam_pos)
                                visualize_mesh_traj(dirnames[b], curr_traj_pred, config=config, plotter=plotter, index=(1, k), text='Prediction', trajvel=('vel' in config['extra_data']), camera=cam_pos, stroke_ids=stroke_ids_pred, tour=tour_pred)
                                
                                # Alternatively display mesh with opacity
                                # visualize_mesh_v2(dirnames[b], config=config, plotter=plotter, index=(1,k), camera=cam_pos, opacity=.3)
                                # visualize_traj(curr_traj_pred_as_pc, extra_data=config['extra_data'], plotter=plotter, index=(1, k), text='Prediction', camera=cam_pos, trajc='lightblue')
                                
                            
                    if not args.video:
                        print(f"Writing: {os.path.join(save_dir, model_prefix+'_'+str(suffix)+'_batch'+str(batch_id)+'_sidebysidemulti_'+str(render_iter)+'_'+tag_mask+'.png')}")
                        plotter.show(screenshot=(os.path.join(save_dir, model_prefix+'_'+str(suffix)+'_batch'+str(batch_id)+'_sidebysidemulti_'+str(render_iter)+"_"+tag_mask+".png") if not args.display else None  ))
                    
                    render_iter += 1

            if args.video:
                plotter.close()
                exit()

            ### Full batch figure
            if 'shelves' in category:
                camera_pos = [-5,1,1]
            else:
                camera_pos = None

            if  not args.sidebyside and not args.concat:
                nrows = args.nrows
                ncols = args.ncols
                plotter = pv.Plotter(shape=(nrows, ncols), window_size=(1920,1080), title=str(basename), off_screen=(True if not args.display else False))
                for b in range(min(B,nrows*ncols)):

                    if config.task_name == 'MaskPlanner':
                        # curr_traj_pred = from_seq_to_pc(traj_pred[b].copy(), extra_data=config['extra_data'])
                        visualize_mesh_traj(dirnames[b], processed_traj_pred[b], config=config, plotter=plotter, index=(b//ncols, b%ncols), text='Prediction', camera=cam_pos, stroke_ids=processed_stroke_ids_pred[b])
                    
                    else:
                        # All other tasks
                        curr_traj = from_seq_to_pc(traj[b].copy(), extra_data=config['extra_data'])
                        if traj_pred is not None:
                            curr_traj_pred = from_seq_to_pc(traj_pred[b].copy(), extra_data=config['extra_data'])
                        else:
                            curr_traj_pred = curr_traj.copy()
                            
                        # norm_meshfile = os.path.join(find_parent_of(dirnames[b], dataset_paths), dirnames[b], (dirnames[b]+'_norm.obj'))
                        curr_traj_pred = from_pc_to_seq(curr_traj_pred, traj_points=curr_traj_pred.shape[0], lambda_points=config['lambda_points'], overlapping=config['overlapping'], extra_data=config['extra_data'])
                        
                        visualize_mesh_traj(dirnames[b], curr_traj_pred, config=config, plotter=plotter, index=(b//ncols, b%ncols), camera=camera_pos, stroke_ids=None)

                plotter.show(screenshot=(os.path.join(save_dir, model_prefix+'_'+str(suffix)+'_batch'+str(batch_id)+'.png') if not args.display else None))


def find_parent_of(dirname, parents):
    """Temp function to find parent dir of given dirname

    dirname: str
             dir to find parent of
    parents: str
           possible parent dirs
    """
    for parent in parents:
        if os.path.isdir(os.path.join(parent, dirname)):
            return parent

    raise ValueError('Parent not found.', parents, dirname)

if __name__ == '__main__':
    main()