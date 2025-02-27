import pdb
import random
import os
from tqdm import tqdm
from threadpoolctl import ThreadpoolController

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pyvista as pv
import torch

from . import orient_in, rot_from_representation
from .pointcloud import from_pc_to_seq, from_seq_to_pc, get_dim_traj_points, get_traj_feature_index, remove_padding, remove_padding_v2, get_mean_mesh, from_bbox_encoding_to_visual_format
from .disk import get_dataset_paths, get_dataset_name, get_dataset_downscale_factor
from . import get_root_of_dir

controller = ThreadpoolController()


def remove_padding_from_vectors(vectors):
    """From an array of vectors, 
        remove the fake vectors

        vectors : (N, D)
                   some of the N vectors are fake,
                   and filled with -100 values

        ---
        returns
            out_vectors : (M, D)
                          where M is the number of true vectors
    """
    assert vectors.ndim == 2
    fake_mask = np.all((vectors[:, :] == -100), axis=-1)  # True for fake vectors
    vectors = vectors[~fake_mask]
    return vectors 
    

def visualize_mesh(meshfile, plotter=None, index=None, text=None, camera=None):
    """Visualize mesh, given filename.obj"""
    show_plot = True if plotter is None else False
    if plotter is not None:
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    mesh_obj = pv.read(meshfile)
    plotter.add_mesh(mesh_obj)

    if camera is not None:
        plotter.set_position(camera)
    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show()
    return


def visualize_mesh_v2(dirname, config, plotter=None, index=None, text=None, camera=None, opacity=1.):
    """Visualize mesh, given dirname.
        
        The mesh contained in dirname is found and normalized
        according to the normalization specified.
    """
    show_plot = True if plotter is None else False
    if plotter is not None:
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    roots = get_dataset_paths(config.dataset)
    meshpath = os.path.join(get_root_of_dir(dirname, roots), dirname, dirname+'.obj')  # retrieve mesh path (unnormalized)
    mesh_obj = pv.read(meshpath)
    mesh_obj = normalize_pv_mesh(mesh_obj, meshpath=meshpath, normalization=config.normalization,  dataset_name=get_dataset_name(config.dataset))  # Normalized mesh
    
    plotter.add_mesh(mesh_obj, opacity=opacity)

    if camera is not None:
        plotter.set_position(camera)
    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show()
    return


def visualize_boxes(boxes, plotter, colors=[], **kwargs):
    """
        boxes: batch of box encoded format as the one used during training
               (x,y,z,w,h,d)
    """
    for i, bbox in enumerate(boxes):
        if len(colors) != 0:
            visualize_box(from_bbox_encoding_to_visual_format(bbox), plotter=plotter, color=colors[i], **kwargs)
        else:
            visualize_box(from_bbox_encoding_to_visual_format(bbox), plotter=plotter, **kwargs)
    return


def visualize_box(box, plotter=None, index=None, text=None, opacity=0.4, color='lightblue', camera=None):
    """
        box: tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    show_plot = True if plotter is None else False

    if plotter is not None:
        assert index is not None, 'index is None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    mesh = pv.Box(bounds=box)
    plotter.add_mesh(mesh, color=color, opacity=opacity, show_edges=True)

    if camera is not None:
        plotter.set_position(camera)
    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show_axes()
        plotter.show()
    return


def visualize_sops(sops, plotter=None, stroke_ids=None, colors=[], **kwargs):
    """
        sops: set of SoPs (start of path tokens) as concatenated points (outdim*length)
    """
    # Repeat the same color multiple times, if given as a string
    if isinstance(colors, str):
        colors = np.repeat(colors, len(sops))

    n_strokes = len(sops)

    for i, sop in enumerate(sops):
        curr_sop = np.array(to_numpy(sop))
        assert curr_sop.ndim == 1
        # Skip if it's fake (padding)
        if np.all(curr_sop == -100):
            continue

        visualize_sop(curr_sop,
                      plotter=plotter,
                      stroke_id=stroke_ids[i] if stroke_ids is not None else None,
                      color=colors[i] if len(colors) != 0 else None,
                      n_colors=n_strokes,
                      **kwargs)

    return


def visualize_sop(sop,
                  plotter=None,
                  index=None,
                  text=None,
                  opacity=1.,
                  stroke_id=None,
                  color='lightgreen',
                  camera=None,
                  line_width=10,
                  hide_bar=False,
                  n_colors=None,
                  force_cmap=None,
                  config={}):
    """Display SoPs as sequence of cylinders connecting each pair of consecutive points,
        plus a sphere at the beginning on the sequence.

        sop: sequence of concatenated points (outdim*length)
        stroke_id: int, overrides color if set. Useful to set the same color for objects with same stroke_ids
    """
    show_plot = True if plotter is None else False
    add_mesh_kwargs = {
            'cmap': 'Set1' if force_cmap is None else force_cmap
    } 

    if plotter is not None:
        assert index is not None, 'index is None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    outdim = get_dim_traj_points(config.extra_data)

    assert sop.ndim == 1 and sop.shape[0] % outdim == 0
    n_points = sop.shape[0] // outdim    

    sop_pointwise = sop.reshape(-1, outdim)
    points = sop_pointwise[:, :3]  # do not display normals

    # Add a sphere at the first point
    sphere_radius = 0.03
    # sphere_radius = 0.06
    start_sphere = pv.Sphere(radius=sphere_radius, center=points[0])
    plotter.add_mesh(start_sphere,
                     scalars=(np.ones(start_sphere.points.shape[0])*stroke_id if stroke_id is not None else None),  # color all mesh according to stroke_id
                     color=color,
                     show_scalar_bar=not hide_bar,
                     n_colors=n_colors,
                     clim=[0., n_colors],
                     **add_mesh_kwargs)  # You can change the color as desired

    # Iterate over pairs of consecutive points and create cylinders
    radius = 0.015
    # radius = 0.03
    for i in range(len(points) - 1):
        # Define the start and end points of the cylinder
        start_point = points[i]
        end_point = points[i + 1]

        # Calculate the direction vector and height of the cylinder
        direction = end_point - start_point
        height = np.linalg.norm(direction)

        # Create the cylinder
        cylinder = pv.Cylinder(center=(start_point + end_point) / 2,
                               direction=direction,
                               radius=radius,
                               height=height)

        # Add the cylinder to the plotter
        # plotter.add_mesh(cylinder, color=color)
        plotter.add_mesh(cylinder,
                         scalars=(np.ones(cylinder.points.shape[0])*stroke_id if stroke_id is not None else None),  # color all mesh according to stroke_id
                         color=color,
                         show_scalar_bar=not hide_bar,
                         n_colors=n_colors,
                         clim=[0., n_colors],
                         **add_mesh_kwargs)

    if camera is not None:
        plotter.set_position(camera)
    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show_axes()
        plotter.show()
    return


def visualize_traj(traj, stroke_ids=None, plotter=None, index=None, text=None, trajc='lightblue', extra_data=[], camera=None, opacity=1.0):
    expected_outdim = get_dim_traj_points(extra_data)
    if traj.shape[-1] != expected_outdim:
        # traj = from_seq_to_pc(traj, extra_data=extra_data)
        raise ValueError('Use visualize_sequence_traj to view lambda-sequences')

    show_plot = True if plotter is None else False

    if plotter is not None:
        assert index is not None, 'index is None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    if torch.is_tensor(traj):
        traj = traj.cpu().detach().numpy()
    traj_pc = pv.PolyData(traj[:, :3])
    plotter.add_mesh(traj_pc, scalars=stroke_ids, color=trajc, point_size=10.0, opacity=opacity, render_points_as_spheres=True)

    if get_traj_feature_index('vel', extra_data) is not None:
        indexes = get_traj_feature_index('vel', extra_data)
        plotter.add_arrows(traj[:, :3], traj[:, indexes], mag=1, color='green', opacity=0.8)

    if orient_in(extra_data)[0]:
        orient_repr = orient_in(extra_data)[1]
        indexes = get_traj_feature_index(orient_repr, extra_data)

        e1 = np.array([1, 0, 0])
        rots = rot_from_representation(orient_repr, traj[:, indexes])
        e1_rots = rots.apply(e1)
        down_scale_factor = 10
        e1_rots /= down_scale_factor
        plotter.add_arrows(traj[:, :3]-e1_rots, e1_rots, mag=1, color='red', opacity=0.8)

    if camera is not None:
        plotter.set_position(camera)
    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show_axes()
        plotter.show()
    return


def visualize_sequence_traj(traj, **args):
    """Visualize traj as groups of segments (lmbda > 1)

        traj: (N/lmbda, 3*lmbda) array
    """
    expected_outdim = get_dim_traj_points(args['extra_data'])
    assert traj.ndim == 2 and traj.shape[-1] > expected_outdim, f'Make sure to reshape the traj as a group of segments before visualizing it. ndim:{traj.ndim} | shape:{traj.shape}'

    lmbda = int(traj.shape[-1]/expected_outdim)

    traj = traj.reshape(-1, expected_outdim)
    traj = remove_padding(traj, args['extra_data'])
    traj = traj.reshape(-1, expected_outdim*lmbda)

    n_sequences = traj.shape[0]

    sequence_ids = np.repeat(np.arange(n_sequences), lmbda)  # (N,) new stroke_ids, one for each sequence

    traj = traj.reshape(-1, expected_outdim)
    visualize_complete_traj(traj, sequence_ids, **args)

    return


def visualize_complete_traj(traj, stroke_ids=None, plotter=None, index=None, text=None, extra_data=[], cpos=None):
    """Plot trajectory with strokes as different colours
    Input:
        traj: (N, 3) array with x-y-z
        stroke_ids: (N,) with stroke_ids
    """
    expected_outdim = get_dim_traj_points(extra_data)
    assert traj.ndim == 2 and traj.shape[-1] == expected_outdim

    show_plot = True if plotter is None else False

    if plotter is not None:
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1))
        plotter.subplot(0,0)

    if cpos is not None:
        plotter.camera_position = cpos

    pc = pv.PolyData(traj[:, :3])
    plotter.add_mesh(pc,
                     # scalars=np.multiply(5*stroke_ids, np.sin(30*stroke_ids)),
                     # scalars=list(map(lambda x: zlib.crc32(bytes(x)) % 400, stroke_ids)),  # (https://stackoverflow.com/questions/66055968/python-map-integer-to-another-random-integer-without-random)
                     scalars=stroke_ids,
                     # scalars=60*np.sin(2*math.pi*(1/12)*stroke_ids),
                     cmap="Set1",
                     #show_scalar_bar=False,
                     point_size=12.0,
                     opacity=1.0,
                     render_points_as_spheres=True,
                     scalar_bar_args={'title': f'strokes [{index}]'})

    if orient_in(extra_data)[0]:
        orient_repr = orient_in(extra_data)[1]
        indexes = get_traj_feature_index(orient_repr, extra_data)
        e1 = np.array([1, 0, 0])
        rots = rot_from_representation(orient_repr, traj[:, indexes])
        e1_rots = rots.apply(e1)
        down_scale_factor = 10
        e1_rots /= down_scale_factor
        plotter.add_arrows(traj[:, :3]-e1_rots, e1_rots, mag=1, color='red', opacity=0.8)

    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show()
    return


def visualize_centroid_traj(traj, stroke_ids=None, plotter=None, index=None, text=None, extra_data=[], cpos=None):
    """Plot trajectory with strokes as different colours
    Input:
        traj: (N, lambda_points, outdim) array
        stroke_ids: (N,) with stroke_ids
    """
    expected_outdim = get_dim_traj_points(extra_data)
    assert traj.ndim == 3 and traj.shape[-1] == expected_outdim

    show_plot = True if plotter is None else False

    if plotter is not None:
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1))
        plotter.subplot(0,0)

    if cpos is not None:
        plotter.camera_position = cpos
    traj_centroids = traj[:, :, :3].mean(axis=1)
    traj_out_vel = traj[:, -1, :3] - traj[:, -2, :3] 
    pc = pv.PolyData(traj_centroids[:, :3])
    plotter.add_mesh(pc,
                     scalars=stroke_ids,
                     # scalars=60*np.sin(2*math.pi*(1/12)*stroke_ids),
                     cmap="Set1",
                     #show_scalar_bar=False,
                     point_size=12.0,
                     opacity=1.0,
                     render_points_as_spheres=True,
                     scalar_bar_args={'title': f'strokes [{index}]'})
    plotter.add_arrows(traj_centroids, traj_out_vel, mag=2, color='green', opacity=0.8)

    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show()
    return


def visualize_complete_traj_tour(traj, stroke_ids, tour, plotter=None, index=None, text=None, extra_data=[], lines=False, cpos=None):
    """Plot trajectory with strokes as different colours
    Input:
        traj: (N, lambda_points, 3) array with x-y-z
        stroke_ids: (N,) with stroke_ids
        tour: (N)
    """
    expected_outdim = get_dim_traj_points(extra_data)
    assert traj.ndim == 3 and traj.shape[-1] == expected_outdim
    cmaps =  ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    get_cmap = lambda i: cmaps[i]  if i < len(cmaps) else random.choice(cmaps)
    show_plot = True if plotter is None else False

    if plotter is not None:
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    if cpos is not None:
        plotter.camera_position = cpos

    unique_stroke_ids = np.unique(stroke_ids)
    scalar_bar_title = f'strokes [{np.random.randint(0, 100)}]'
    for i, sid in enumerate(unique_stroke_ids):
        segment_idx = np.where(stroke_ids == sid)[0]
        stroke = traj[segment_idx].copy() # extract stroke from multi-stroke
        stroke = stroke[tour[segment_idx]] # order stroke according to tour
        scalars = np.arange(stroke.shape[0])
        scalars = (scalars + 1) / (scalars + 1).max() # normalize to 1

        pc = stroke.reshape(-1, expected_outdim)[:, :3]
        scalars = scalars.repeat(pc.shape[0]//stroke.shape[0])
        plotter.add_points(pc, point_size=10.0, render_points_as_spheres=True, scalars=scalars, cmap=get_cmap(i), scalar_bar_args={'title': scalar_bar_title})
        
        if lines:
            assert tour is not None
            stroke_subsampled = stroke[:, :, :3].reshape((-1, 3))
            s_len = stroke_subsampled.shape[0]
            if stroke_subsampled.shape[0] > 1:
                
                # stroke_subsampled = stroke_subsampled[np.dstack((np.arange(0, s_len, 2), np.arange(1, s_len, 2))).ravel()[1:]]
                
                stroke_subsampled = pv.lines_from_points(stroke_subsampled)
                plotter.add_mesh(stroke_subsampled, color="red", render_lines_as_tubes=True, line_width=7)

    plotter.update_scalar_bar_range([0, 1], name=scalar_bar_title)
    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show()
    return

def visualize_pc(pc,
                 plotter=None,
                 index=None,
                 text=None,
                 color=None,
                 camera=None,
                 camera_rot=None,
                 camera_zoom=None,
                 point_size=None,
                 add_points_kwargs={}):
    """Visualize point-cloud"""

    show_plot = True if plotter is None else False

    if plotter is not None:
        assert index is not None, 'index is None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    if torch.is_tensor(pc):
        pc = pc.cpu().detach().numpy()

    if camera is not None:
        plotter.set_position(camera)

    if camera_rot is not None:
        # camera_rot : tuple of azimuth, roll, elevation
        plotter.camera.azimuth, plotter.camera.roll, plotter.camera.elevation = camera_rot

    if camera_zoom is not None:
        plotter.camera.zoom(camera_zoom)

    # pc = pv.PolyData(pc[:, :3])
    # plotter.add_mesh(pc, point_size=6.0, opacity=1.0, render_points_as_spheres=True)

    plotter.add_points(pc[:, :3],
                       color=color,
                       point_size=20.0 if point_size is None else point_size,
                       opacity=1.0,
                       render_points_as_spheres=True,
                       **add_points_kwargs  
                       )

    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show_axes()
        plotter.show()

    return


def normalize_pv_mesh(mesh, meshpath, normalization='per-dataset', **kwargs):
    """normalize a mesh that has already been read through pv.read(meshpath)
        
        mesh: pv mesh object
    """
    assert normalization=='per-dataset' and 'dataset_name' in kwargs, 'Not yet implemented for per-mesh normalization.'

    centroid = get_mean_mesh(meshpath)
    scale = get_dataset_downscale_factor(kwargs.get('dataset_name'))
    mesh = mesh.translate(-centroid, inplace=False).scale(1/scale, inplace=False)
    return mesh


def visualize_mesh_traj_multiangle(save_dir=None, filename='', return_fig=False, **kwargs):
    """Wrapper for visualize_mesh_traj to show the object
        under four different camera view angles. 

        We also generate multiple rows for randomizing the stroke colors
        to disambiguate among strokes that are different but coloured the same.       
    """
    assert 'orbiting' not in kwargs or kwargs['orbiting'] == False, 'orbiting is not supported for multi-angle visualization'
    assert 'dirname' in kwargs and \
           'traj' in kwargs and \
           'config' in kwargs

    angles = 4  # number of different camera angles
    camera_positions = [[-0.4,1,0], [-0.5,0,-1], [0.5,0.5,0.5], [-1,-0.2,0]]

    stroke_ids = None
    if 'stroke_ids' in kwargs:
        n_rows = 2  # i.e. number of color permutations

        stroke_ids = kwargs['stroke_ids']
        del kwargs['stroke_ids']

        # create as many permutations as number of rows
        stroke_ids_permutations = []
        for k in range(n_rows):
            stroke_ids_permutations.append(randomize_labels_except_special(stroke_ids))
    else:
        n_rows = 1  # single row if stroke_ids are not given


    plotter = pv.Plotter(shape=(n_rows, angles), window_size=(1920,1088), off_screen=True if save_dir is not None else False)


    for n_figure in range(angles*n_rows):
        row = n_figure // angles
        col = n_figure % angles

        # Randomize stroke_ids for different views, so that
        # it's easier to tell similar-colored strokes apart
        if stroke_ids is not None:
            # stroke_ids = randomize_labels_except_special(stroke_ids)
            stroke_ids = stroke_ids_permutations[row]
            
        visualize_mesh_traj(plotter=plotter,
                            index=(row, col),
                            cpos=camera_positions[col],
                            point_size=18.0,  # smaller point_size as figures are now smaller
                            save_dir=None,
                            stroke_ids=stroke_ids,
                            **kwargs)
        
        plotter.camera.zoom(1.15)  # zoom in on active subplot as the individual figure is now smaller

    if save_dir is not None:
        plotter.screenshot(os.path.join(save_dir, filename if filename != '' else 'mesh_traj.png'))
        plotter.close()  # free up GPU memory after saving image
    elif not return_fig:
        plotter.show_axes()
        plotter.show()
    else:
        return plotter


def visualize_mesh_traj(dirname,
                        traj,
                        config,
                        stroke_ids=None,
                        plotter=None,
                        index=None,
                        text=None,
                        trajc='lightgreen',
                        trajvel=False,
                        camera=None,
                        camera_zoom=None,
                        cpos=None,
                        camera_rot=None,
                        cmap=None,
                        arrow_color=None,
                        tour=None,
                        lines=False,
                        save_dir=None,
                        filename='',
                        orbiting=False,
                        point_size=None,
                        hide_mesh=False,
                        hide_bar=False,
                        hide_arrows=False,
                        progressive_point_colors=False,
                        force_color=None,
                        force_cmap=None,
                        force_n_strokes=None,
                        mesh_only=False,
                        force_renormalization=False,
                        add_mesh_kwargs={},
                        add_points_kwargs={},
                        paint_coverage_kwargs={}):
    """Visualize mesh-traj pair

        dirname: str
                 name of dir where .obj mesh file is located, with same name as dir
        traj : (N,k) array
               set of segments/poses, according to lambda_points.
               traj is assumed to be ALREADY normalized according to
               config.normalization
    """
    curr_traj = to_numpy(traj).copy()
    stroke_ids = to_numpy(stroke_ids)

    lambda_points, overlapping, extra_data, normalization = config.lambda_points, config.overlapping, config.extra_data, config.normalization

    show_plot = True if plotter is None else False
    if plotter is not None:
        assert index is not None, 'index is None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1088), off_screen=True if save_dir is not None else False)
        plotter.subplot(0,0)

    roots = get_dataset_paths(config.dataset)
    dirname_parent_path = get_root_of_dir(dirname, roots)
    meshpath = os.path.join(dirname_parent_path, dirname, dirname+'.obj')  # retrieve mesh path (unnormalized)
    
    mesh_obj = pv.read(meshpath)  # Read mesh
    mesh_obj = normalize_pv_mesh(mesh_obj, meshpath=meshpath, normalization=normalization,  dataset_name=get_dataset_name(config.dataset))  # Normalized mesh

    if not hide_mesh:
        # Display mesh

        if len(paint_coverage_kwargs) != 0 and paint_coverage_kwargs['on'] == True:
            # Display mesh with coloured faces according to deposited paint
            mesh_faces = mesh_obj.faces.reshape(-1, 4)[:, 1:]
            n_faces = mesh_obj.n_faces  # = mesh_faces.shape[0]

            pred_thicknesses = paint_coverage_kwargs['vertices_thickness']
            gt_thicknesses = paint_coverage_kwargs['gt_vertices_thickness']
            
            gt_thicknesses = get_mesh_face_colors(vertices=mesh_obj.points,
                                                  faces=mesh_faces,
                                                  vertices_thickness=gt_thicknesses,
                                                  normalize_to_max=None,
                                                  clamp=None)


            pred_thicknesses = get_mesh_face_colors(vertices=mesh_obj.points,
                                                    faces=mesh_faces,
                                                    vertices_thickness=pred_thicknesses,
                                                    normalize_to_max=None,
                                                    clamp=None)


            visual_clamp = np.percentile(gt_thicknesses, paint_coverage_kwargs['percentile'])  # this should be set according to the compute_paint_coverage_script! or even in terms of absolute thickness!
            # visual_clamp = 7
            print('Visual clamp:', visual_clamp)

            push_uncovered_to_zero = False

            if not paint_coverage_kwargs['postprocessed'] and paint_coverage_kwargs['paint_correction_for_overlapping']:
                if config.lambda_points == 4:
                    pred_thicknesses /= 1.25
                elif config.lambda_points > 1:
                    raise ValueError()  # not expecting a lambda != than 4 for now.

            tot_pred_paint = np.sum(pred_thicknesses)
            tot_gt_paint = np.sum(gt_thicknesses)
            paint_factor = tot_pred_paint / tot_gt_paint
            print('tot paint pred/gt factor:', round(paint_factor, 3))

            display_thicknesses = pred_thicknesses if not paint_coverage_kwargs['gt_instead_of_pred'] else gt_thicknesses
            max_vertex_thickness = None
            clamp = None
            face_colors = display_thicknesses
            
            if clamp is not None:
                face_colors[face_colors > clamp] = clamp

            if max_vertex_thickness is not None:
                face_colors /= np.max(face_colors)
                face_colors *= max_vertex_thickness


            print(f'Max pred: {np.max(pred_thicknesses)} | Max gt: {np.max(gt_thicknesses)}')
            print('st dev:', np.std(face_colors))
            print('10-th perc:', np.percentile(gt_thicknesses, 10))
            print(f'Tot deposited paint: pred: {np.sum(pred_thicknesses)} | gt: {np.sum(gt_thicknesses)}')

            plotter.add_mesh(mesh_obj,
                             scalars=face_colors,  # color for each mesh face
                             # clim=[0., visual_clamp] if visual_clamp is not None else None,
                             clim=[0., visual_clamp] if visual_clamp is not None else None,
                             scalar_bar_args={'vertical': True},
                             below_color='#ececec',
                             show_scalar_bar=not hide_bar,
                             **add_mesh_kwargs)
        else:
            plotter.add_mesh(mesh_obj,
                             **add_mesh_kwargs)
        

    if camera is not None:
        plotter.set_position(camera)

    if cpos is not None:
        plotter.camera_position = cpos

    if camera_rot is not None:
        # camera_rot : tuple of azimuth, roll, elevation
        plotter.camera.azimuth, plotter.camera.roll, plotter.camera.elevation = camera_rot

    if text is not None:
        plotter.add_text(text)

    if camera_zoom is not None:
        plotter.camera.zoom(camera_zoom)

    if lambda_points > 1:
        # from segments to single poses for visualization purposes
        outdim = get_dim_traj_points(extra_data)
        assert curr_traj.shape[-1]%outdim == 0
        if stroke_ids is not None:
            curr_traj, stroke_ids = remove_padding_v2(curr_traj, stroke_ids=stroke_ids)
            curr_traj = curr_traj.reshape(-1, outdim)
            # duplicate stroke_ids lambda-times as segments are now rolled out into single poses
            if stroke_ids.shape[0] != curr_traj.shape[0]:
                stroke_ids = stroke_ids[:curr_traj.shape[0]//lambda_points, None]  # remove padding from stroke_ids
                stroke_ids = np.repeat(stroke_ids, lambda_points)  # stroke_ids from sequence to point

            if tour is not None:
                tour = tour[:curr_traj.shape[0]//lambda_points, None] # remove padding from stroke_ids
                tour = np.repeat(tour, lambda_points)
            
            assert (stroke_ids.shape[0] == curr_traj.shape[0]), f"{stroke_ids.shape}, {curr_traj.shape}"
        else:
            curr_traj = curr_traj.reshape(-1, outdim)
            curr_traj = remove_padding(curr_traj, extra_data)  # makes sense only if it's GT traj, but doesn't hurt


    # Force re-normalization (joint-training)
    # force_renormalization = True
    if force_renormalization:
        assert config.data_scale_factor is not None, f'config.data_scale_factor is expected to be not None.'
        renorm_to_category = 'containers-v2'  # temp for containers joint-training visualization
        curr_traj[:, :3] *= config.data_scale_factor  # denorm
        curr_traj[:, :3] /= get_dataset_downscale_factor(renorm_to_category)  # norm

    traj_pc = pv.PolyData(curr_traj[:, :3])

    if not mesh_only:
        if tour is not None:
            # Display within-stroke tour arrows
            curr_traj_subdivided = []
            if stroke_ids is not None:
                stroke_ids_unique = np.unique(stroke_ids)
                for i in stroke_ids_unique:
                    indexes = np.where(stroke_ids == i)[0]
                    curr_traj_subdivided.append(indexes) # create a list with points grouped by stroke id

            cmaps =  ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
            get_cmap = lambda i: cmaps[i]  if i < len(cmaps) else random.choice(cmaps)

            for i, curr_traj_stroke_indexes in tqdm(enumerate(curr_traj_subdivided), desc="Strokes"):
                tmp_traj_pc = traj_pc.extract_points(curr_traj_stroke_indexes)
                tour_length = tour[curr_traj_stroke_indexes]
                scalars = tour_length/tour_length.max() # normalize to 1

                plotter.add_mesh(tmp_traj_pc, scalars=scalars, point_size=20.0 if point_size is None else point_size, opacity=1.0, render_points_as_spheres=True, cmap=get_cmap(i))
                if lines:
                    segments_id = tour_length[np.sort(np.unique(tour_length, return_index=True)[1])] if tour is not None else np.arange(tour_length.shape[0]//lambda_points)

                    sorted_segment_indexes = np.argsort(segments_id)
                    tmp_traj_seg = from_pc_to_seq(tmp_traj_pc.points.copy(), None, lambda_points, overlapping, [], False)
                    tmp_traj_seg = tmp_traj_seg.reshape((tmp_traj_seg.shape[0], lambda_points, -1))
                    
                    # Alternative 1: draw lines connecting middle points of segments
                    tmp_traj_seg = tmp_traj_seg[:, lambda_points//2]
                    tmp_traj_seg = tmp_traj_seg[sorted_segment_indexes]
                    if tmp_traj_seg.shape[0] > 1:
                        tmp_traj_seg = tmp_traj_seg[np.dstack((np.arange(tmp_traj_seg.shape[0]), np.arange(tmp_traj_seg.shape[0]))).ravel()[1:-1]]
                        tmp_traj_seg = pv.lines_from_points(tmp_traj_seg)
                        plotter.add_mesh(tmp_traj_seg, color="red", render_lines_as_tubes=True, line_width=5)

        elif tour is None and stroke_ids is not None:
            # Display strokes as separate colors using stroke_ids
            for s_id in np.unique(stroke_ids):
                stroke = traj_pc.points[stroke_ids == s_id]
                scalars = stroke_ids[stroke_ids == s_id]
                cmap = "Set1"  # hsv, Paired, turbo, Set1 (for cuboids worked well), see all at: https://matplotlib.org/stable/users/explain/colors/colormaps.html
                if force_cmap is not None:
                    cmap = force_cmap

                unique_stroke_ids, counts = np.unique(stroke_ids, return_counts=True)
                n_strokes = len(unique_stroke_ids)

                if force_n_strokes is not None:
                    n_strokes = force_n_strokes  # consistent number of colors in the cmap. Used with --align_stroke_ids in render_results

                if progressive_point_colors:
                    # For "Reds":
                    # scalars = (np.arange(stroke.shape[0]) / (stroke.shape[0]*1.5)) + 0.33

                    # For "RgBu"
                    scalars = (np.arange(stroke.shape[0]) / (stroke.shape[0]))

                if force_color is not None:
                    scalars = None

                uid = index if index is not None else (0,0)
                show_scalar_bar = True if uid == (0,0) else None  # show one colormap only
                show_scalar_bar = False if hide_bar else show_scalar_bar  # overwrite display of bar if hide_bar is True

                plotter.add_points(stroke,
                                color=force_color,  # overwritten by scalars, if scalars is not None
                                scalars=scalars,
                                point_size=20.0 if point_size is None else point_size,
                                opacity=1.0,
                                render_points_as_spheres=True,
                                cmap=cmap,
                                show_scalar_bar=show_scalar_bar, 
                                n_colors=n_strokes if not progressive_point_colors else 256,
                                clim=[-0.1, n_strokes],  # -0.1 prevents stroke 0 to be same color as -1 
                                below_color='black',  # outliers encoded as -1 will be displayed in black color
                                scalar_bar_args={
                                                    'title': f'I={len(unique_stroke_ids)} [id:{uid}]',  # a unique title must be given to each subplot for the colormap to work
                                                    'position_x': 0.05,
                                                    'position_y': 0.01,
                                                    # 'width': 0.1,
                                                    # 'height': 0.8,
                                                    'n_labels': n_strokes,
                                                    'n_colors': n_strokes,
                                                    # 'interpolate_before_map': True,
                                                    'vertical': True,
                                                    'fmt': '%2.0f'
                                                },
                                **add_points_kwargs  
                                )

        else:
            # Display trajectory as point-cloud without color-encoding
            # cmap = handle_cmap_input(cmap)
            plotter.add_mesh(traj_pc,
                            color=trajc,
                            scalars=None,
                            point_size=14.0 if point_size is None else point_size,
                            opacity=1.0,
                            render_points_as_spheres=True,
                            cmap=None)

    if trajvel:
        # Display arrows of velocity for each pose
        assert 'vel' in extra_data, 'Cannot display traj velocity: trajectory does not contain velocities'
        plotter.add_arrows(curr_traj[:, :3], curr_traj[:, 3:6], mag=1, color='green', opacity=0.8)

    if orient_in(extra_data)[0] and not mesh_only and not hide_arrows:
        # Display orientation arrows
        orient_repr = orient_in(extra_data)[1]
        indexes = get_traj_feature_index(orient_repr, extra_data)

        e1 = np.array([1, 0, 0])
        rots = rot_from_representation(orient_repr, curr_traj[:, indexes])
        e1_rots = rots.apply(e1)
        down_scale_factor = 10
        e1_rots /= down_scale_factor

        if arrow_color is None:
            arrow_color = 'red'
            
        plotter.add_arrows(curr_traj[:, :3]-e1_rots, e1_rots, mag=1, color=arrow_color, opacity=0.35)

    if save_dir is not None:
        if orbiting:
            # Save a video of a camera orbiting around the mesh
            # See more at: (1) https://docs.pyvista.org/version/stable/examples/02-plot/orbit#orbiting-example
            # and (2) https://tutorial.pyvista.org/tutorial/03_figures/d_gif.html
            plotter.show(auto_close=False)
            path = plotter.generate_orbital_path(n_points=144, shift=1.0, factor=2.0, viewup=[0, 0, 1])
            plotter.open_movie(os.path.join(save_dir, filename if filename != '' else 'mesh_traj.mp4'))
            plotter.orbit_on_path(path, write_frames=True, viewup=[0, 1, 0], step=0.02)  # the viewup change allows to orbit from different perspectives
            plotter.close()
        else:
            plotter.screenshot(os.path.join(save_dir, filename if filename != '' else 'mesh_traj.png'))
            plotter.close()  # free up GPU memory after saving image
    elif show_plot:
        plotter.show_axes()
        plotter.show()
        
    return

def visualize_mesh_traj_animated(meshfile,
                        traj,
                        plotter=None,
                        index=None,
                        text=None,
                        trajc='lightblue',
                        trajvel=False,
                        lambda_points=1,
                        camera=None,
                        extra_data=[],
                        stroke_ids=None,
                        cmap=None,
                        arrow_color=None,
                        tour=None):
    """Visualize mesh-traj pair

        meshfile: str
                  mesh filename.objr
        traj : (N,k) array
        
        lambda_points: traj is set of sequences of lambda_points
    """
    curr_traj = traj.copy()
    show_plot = True if plotter is None else False

    if plotter is not None:
        assert index is not None, 'index is None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    mesh_obj = pv.read(meshfile)
    plotter.add_mesh(mesh_obj)
    
    
    if camera is not None:
        plotter.set_position(camera)

    if text is not None:
        plotter.add_text(text)

    if torch.is_tensor(curr_traj):
        curr_traj = curr_traj.cpu().detach().numpy()

    if lambda_points > 1:
        outdim = get_dim_traj_points(extra_data)
        assert curr_traj.shape[-1]%outdim == 0
        curr_traj = curr_traj.reshape(-1, outdim)
        curr_traj = remove_padding(curr_traj, extra_data)  # makes sense only if it's GT traj, but doesn't hurt
        if stroke_ids is not None:
            if stroke_ids.shape[0] != curr_traj.shape[0]:
                stroke_ids = stroke_ids[:curr_traj.shape[0]//lambda_points, None] # remove padding from stroke_ids
                stroke_ids = np.repeat(stroke_ids, lambda_points) # stroke_ids from sequence to point
            if tour is not None:
                tour = np.repeat(tour, lambda_points)
            assert (stroke_ids != -1).all()
            assert (stroke_ids.shape[0] == curr_traj.shape[0]), f"{stroke_ids.shape}, {curr_traj.shape}"

    traj_pc = pv.PolyData(curr_traj[:, :3])

    curr_traj_subdivided = []
    if stroke_ids is not None:
        stroke_ids_unique = np.unique(stroke_ids)
        for i in stroke_ids_unique:
            indexes = np.where(stroke_ids == i)[0]
            curr_traj_subdivided.append(indexes) # create a list with points grouped by stroke id

    cmaps =  ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    get_cmap = lambda i: cmaps[i]  if i < len(cmaps) else random.choice(cmaps)

    for i, curr_traj_stroke_indexes in tqdm(enumerate(curr_traj_subdivided), desc="Strokes"):
        plotter.subplot(*index)
        points_to_render = curr_traj[curr_traj_stroke_indexes, :3].copy()
        mesh = pv.PointSet(points_to_render.copy())
        mesh.points[:] = np.nan 
        scalars = tour[curr_traj_stroke_indexes] if tour is not None else np.arange(curr_traj_stroke_indexes.shape[0])
        scalars = scalars/scalars.max() # normalize to 1
        plotter.add_mesh(mesh, scalars=scalars, point_size=14.0, opacity=1.0, render_points_as_spheres=True, cmap=get_cmap(i))
        sorted_indexes = np.argsort(scalars)
        # for j in range(1, points_to_render.shape[0]//lambda_points+2):
        #     plotter.subplot(*index)
        #     render_index = min(j*lambda_points, points_to_render.shape[0])
        #     mesh.points[render_index-lambda_points:render_index] = points_to_render[render_index-lambda_points:render_index] 
        #     yield
        for j in range(1, points_to_render.shape[0]//lambda_points + 2):
            plotter.subplot(*index)
            render_index = min(j*lambda_points, points_to_render.shape[0])
            mesh.points[sorted_indexes[render_index-lambda_points:render_index]] = points_to_render[sorted_indexes[render_index-lambda_points:render_index]]
            yield

    if trajvel:
        assert 'vel' in extra_data, 'Cannot display traj velocity: trajectory does not contain velocities'
        plotter.add_arrows(curr_traj[:, :3], curr_traj[:, 3:6], mag=1, color='green', opacity=0.8)

    if orient_in(extra_data)[0]:
        orient_repr = orient_in(extra_data)[1]
        indexes = get_traj_feature_index(orient_repr, extra_data)

        e1 = np.array([1, 0, 0])
        rots = rot_from_representation(orient_repr, curr_traj[:, indexes])
        e1_rots = rots.apply(e1)
        down_scale_factor = 10
        e1_rots /= down_scale_factor

        if arrow_color is None:
            arrow_color = 'red'
            
        plotter.add_arrows(curr_traj[:, :3]-e1_rots, e1_rots, mag=1, color=arrow_color, opacity=0.8)

    if show_plot:
        plotter.show_axes()
        plotter.show()
    return


def visualize_latent_segments_batch(latent_segments, stroke_ids, save_dir=None, filename='', return_fig=False, batch_size=None):
    B, n_pts, outdim = latent_segments.shape

    if batch_size is not None:
        B = batch_size

    n_permutations = 3
    fig, ax = plt.subplots(nrows=B, ncols=n_permutations, figsize=(16,5*B))

    for b in range(B):
        visualize_latent_segments(latent_segments[b:b+1, :, :], stroke_ids[b:b+1, :],
                                  n_permutations=n_permutations,
                                  save_dir=None,
                                  filename='',
                                  return_fig=True,
                                  figax=(fig, ax),
                                  row=b)

    if return_fig:
        return fig
    else:
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, filename if filename != '' else 'latent_segments.png'), dpi=100)
            plt.close('all')
        else:
            plt.show()


@controller.wrap(limits=1, user_api='openmp')
def visualize_latent_segments(latent_segments, stroke_ids, n_permutations=3, save_dir=None, filename='', return_fig=False, figax=None, row=None):
    """Plot segments in learned latent space with contrastive loss
    
        n_permutations: number of color permutations to disambiguate among stroke colors
    """
    B, n_pts, outdim = latent_segments.shape

    latent_segments = to_numpy(latent_segments)
    stroke_ids = to_numpy(stroke_ids)

    latent_segments = np.copy(latent_segments[0, :, :])  # first element in batch

    x_norm = latent_segments / np.linalg.norm(latent_segments, axis=1)[:, np.newaxis]  # normalized
    # x = latent_segments

    if outdim > 2:
        # Apply T-SNE
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
        x_norm = tsne.fit_transform(x_norm)
        # x = tsne.fit_transform(x)

    cmap = 'Set1'

    if figax is None:
        fig, ax = plt.subplots(nrows=1, ncols=n_permutations, figsize=(16*n_permutations,5))

        for k in range(n_permutations):
            colors = randomize_labels_except_special(stroke_ids[0])  # random stroke_ids permutation
            ax[k].scatter(x_norm[:, 0], x_norm[:, 1], s=40, c=colors, alpha=0.6, cmap=cmap, marker='o')
            ax[k].set_title(f'Norm latent segments [color perm {k}]')
    else:
        fig, ax = figax

        for k in range(n_permutations):
            colors = randomize_labels_except_special(stroke_ids[0])  # random stroke_ids permutation
            ax[row, k].scatter(x_norm[:, 0], x_norm[:, 1], s=40, c=colors, alpha=0.6, cmap=cmap, marker='o')
            ax[row, k].set_title(f'Norm latent segments [color perm {k}]')
    
    plt.gcf().suptitle(f'# strokes = {len(np.unique(stroke_ids))}')

    if return_fig:
        return fig
    else:
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, filename if filename != '' else 'latent_segments.png'), dpi=300)
            plt.close('all')
        else:
            plt.show()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor


def get_mesh_face_colors(vertices,
                         faces,
                         vertices_thickness,
                         normalize_to_max=None,
                         clamp=None):
    """Given a list of `N*3` items `vertices_thickness`, compute the
        thickness of each of the N faces, given that each item corresponds
        to the face's vertex thickness, and are all concatenated together
    """
    assert vertices_thickness.shape[0] == faces.shape[0]*3

    n_faces = faces.shape[0]

    face_colors = []
    for i in range(n_faces):
        indexes = np.arange(i*3, i*3 + 3)  # triplet of vertices
        face_color = np.mean(vertices_thickness[indexes])  # mean of vertices thicknesses
        face_colors.append(face_color)

    face_colors = np.array(face_colors)

    # if normalize_to_max is not None:
    #     face_colors /= np.max(normalize_to_max)  # normalize in [0, normalize_to_max]
    # else:
    #     face_colors /= np.max(vertices_thickness)  # normalize in [0, 1]

    if clamp is not None:
        face_colors[face_colors > clamp] = clamp

    if normalize_to_max is not None:
        face_colors /= np.max(face_colors)
        face_colors *= normalize_to_max

    # if normalize_to_max is not None:
    #     face_colors[face_colors > normalize_to_max] = normalize_to_max

    return face_colors


def get_mesh_face_colors_OLD(vertices,
                             faces,
                             vertices_thickness,
                             gt_vertices_thickness=None):
    """Given a thickness paint for each vertex of the triangular mesh,
        return a color for each face as the mean thickness
        value of the 3 vertices of that face.
    """
    assert vertices_thickness.shape[0] == vertices.shape[0]

    max_thickness = np.max(vertices_thickness)

    print('min max:', np.min(vertices_thickness), np.max(vertices_thickness))

    face_colors = np.array([np.max(vertices_thickness[face]) for face in faces])
    face_colors /= max_thickness

    return face_colors


def randomize_labels_except_special(input_labels, special_label=-1):
    """Random labels ID permutation
        e.g. [0, 0, 1, 1, 2, 2, 2] -> [2, 2, 1, 1, 0, 0, 0]

        keep the special label -1 intact.
    """
    labels = to_numpy(input_labels)

    # Identify unique cluster labels excluding the special label
    unique_labels = np.unique([label for label in labels if label != special_label])
    clim = [np.min(unique_labels), np.max(unique_labels)]

    # Randomly shuffle the unique cluster labels
    np.random.shuffle(unique_labels)

    # Map the shuffled labels back to the original labels
    label_map = {old_label: new_label for old_label, new_label in zip(unique_labels, range(clim[0], clim[1]+1))}

    randomized_labels = [special_label if label == special_label else label_map[label] for label in labels]

    return np.array(randomized_labels)


def get_list_of_colors(n):
    return [(np.random.random(size=3) * 256).astype(int).tolist() for i in range(n)]