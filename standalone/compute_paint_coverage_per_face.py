"""Utility script to compute paint coverage metric given simulator feedback in .csv
    This version is "per_face", i.e. it uses simulator feedback without the "light mesh" checkbox.
    This means that output .csv files have N*3 items, where N is the number of faces. The list is
    the ordered list of faces with the value of the 3 vertex's thicknesses.

    Examples:
        python compute_paint_coverage_per_face.py --gt-run gt_run/ZZZZZZ --runs runs/XXXXXX  runs/YYYYYY
"""
import argparse
import glob
import os
import inspect
import sys
import pdb
import socket
import csv

import numpy as np
import pandas as pd
import seaborn as sns
import pyvista as pv

### Trick to import paintnet_utils from parent dir
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',        default=[], type=str, nargs='+', help='Runs of predictions with vertex thickness values')
    parser.add_argument('--gt-run',      default=None, type=str, help='Ground-truth run with vertex thickness values', required=True)
    parser.add_argument('--split',       default='test', type=str, help='<train,test>')
    parser.add_argument('--percentile', default=None, type=int, help='Explicitly set the percentile threshold.')

    return parser.parse_args()

args = parse_args()


def main():
    assert len(args.runs) > 0
    assert all([os.path.isdir(run) for run in args.runs]), f'Run dir does not exist or is not valid: {args.runs}'
    assert os.path.isdir(args.gt_run)

    n_scanned_feedbacks = 0
    percentile = 10 if args.percentile is None else args.percentile
    if args.percentile is None:
        print(f'WARNING! --percentile has not been set. Percentile value fallbacks to its default value of: {percentile}')

    # print('CARE! This script is currently in debug mode and many changes have been applied without a final confirmation. Dont trust its results for now')

    coverages = [[] for _ in range(len(args.runs))]

    gt_thick_values = []

    for item in os.listdir(args.gt_run):
        if os.path.isdir(os.path.join(args.gt_run, item)):
            raise ValueError('Why is there a dir?')

        gt_path = os.path.join(args.gt_run, item)

        thicknesses = get_thicknesses_values_per_face(gt_path)
        thick_values = thicknesses
        thick_vertices = np.arange(thicknesses.shape[0])

        n_vertices = thick_values.shape[0]
        
        nonzero_mask = np.logical_not(np.isclose(thick_values, 0))
        threshold = np.percentile(thick_values[nonzero_mask], percentile)
        gt_covered_mask = (thick_values >= threshold)
        and_mask = np.logical_and(nonzero_mask, gt_covered_mask)
        # and_mask = gt_covered_mask  # this should suffice
        covered_vertices = set(thick_vertices[and_mask].flatten())
        gt_n_covered = len(covered_vertices)

        # Save all gt thicknesses for global stats
        gt_thick_values += thick_values[and_mask].tolist()

        print('\n---')
        print(f'Mesh: {item} | N faces: {thicknesses.shape[0]} | N 0-thickness faces: {thick_values[~nonzero_mask].shape[0]} | {percentile}-th percentile threshold: {round(threshold, 2)}')
        print_boxplot(thicknesses, prefix='GT thicknesses')

        assert np.all(thick_values >= 0.), 'All thickness values are supposed to be positive.'

        # sns.histplot(thick_values)
        # plt.show()

        for i, run in enumerate(args.runs):
            i_path = os.path.join(run, item)
            assert os.path.isfile(i_path)

            i_thicknesses = get_thicknesses_values_per_face(i_path)

            i_thick_values = i_thicknesses
            i_thick_vertices = np.arange(i_thicknesses.shape[0])

            assert i_thick_vertices[and_mask].shape[0] == len(covered_vertices)
            assert np.all(i_thick_values >= 0.), 'All thickness values are supposed to be positive.'

            # Remove tail beyond X-th percentile
            # non_zero_i_thick_values = i_thick_values[i_thick_values > 0.001]
            # i_thick_values[i_thick_values > np.percentile(non_zero_i_thick_values, 80)] = np.percentile(non_zero_i_thick_values, 80)

            # Normalize to GT range
            # i_thick_values /= np.max(i_thick_values)
            # i_thick_values *= np.max(thick_values)

            i_thick_subset_values = i_thick_values[and_mask]  # among the faces actually covered in GT
            i_n_covered = i_thick_subset_values[i_thick_subset_values >= threshold].shape[0]

            print(f'> run {i} ---> covered faces: {i_n_covered}/{gt_n_covered} ({round(i_n_covered/gt_n_covered*100,2)}%)')
            # print_boxplot(thicknesses, prefix='Pred thicknesses')

            coverages[i].append(i_n_covered/gt_n_covered)

        n_scanned_feedbacks += 1


    # Compute global GT stats
    gt_thick_values = np.array(gt_thick_values)
    print('======== Global GT stats ========')
    print('Global GT thickness boxplot:', np.percentile(gt_thick_values, 25), np.percentile(gt_thick_values, 50), np.percentile(gt_thick_values, 75))
    print(f'Global GT thickness percentiles: 1-st={round(np.percentile(gt_thick_values, 1), 3)} | 2-nd={round(np.percentile(gt_thick_values, 2), 3)} | 5-th={round(np.percentile(gt_thick_values, 5), 3)}| 10-th={round(np.percentile(gt_thick_values, 10), 3)}')
    print('=================================')

    # Compute final coverage results
    coverages = np.array(coverages)
    mean_coverages = np.round(np.mean(coverages, axis=1)*100, 2)
    stdev_coverages = np.round(np.std(coverages, axis=1)*100, 2)

    print('\n\n======== FINAL RESULTS ========')
    print('RUNS ORDER:\n', args.runs)
    print(f'FINAL MEAN COVERAGES:\n {mean_coverages}%',)
    print(f'FINAL ST.DEV COVERAGES:\n {stdev_coverages}%')





def get_thicknesses_values_per_face(path):
    thicknesses_per_vertex_per_face = np.array(pd.read_csv(path, header=None, sep=';')[1])
    thicknesses_per_face = get_mesh_face_colors(thicknesses_per_vertex_per_face)

    return thicknesses_per_face


def get_mesh_face_colors(vertices_thickness,
                         normalize_to_max=None,
                         clamp=None):
    """Given a list of `N*3` items `vertices_thickness`, compute the
        thickness of each of the N faces, given that each item corresponds
        to the face's vertex thickness, and are all concatenated together
    """
    assert vertices_thickness.shape[0] % 3 == 0, 'the provided list should be without the `ligth_mesh` option, hence 3*N where N is the number of faces.'

    n_faces = vertices_thickness.shape[0] // 3

    face_colors = []
    for i in range(n_faces):
        indexes = np.arange(i*3, i*3 + 3)  # triplet of vertices
        face_color = np.mean(vertices_thickness[indexes])  # mean of vertices thicknesses
        face_colors.append(face_color)

    face_colors = np.array(face_colors)

    if clamp is not None:
        face_colors[face_colors > clamp] = clamp

    if normalize_to_max is not None:
        face_colors /= np.max(face_colors)
        face_colors *= normalize_to_max

    return face_colors


def print_boxplot(distribution, prefix='', round_dec=2):
    print(f"{prefix+' ' if prefix != '' else ''}boxplot [min, 25th, 50th, 75th, max]:",
          round(np.min(distribution), round_dec),
          round(np.percentile(distribution, 25), round_dec),
          round(np.percentile(distribution, 50), round_dec),
          round(np.percentile(distribution, 75), round_dec),
          round(np.max(distribution), round_dec))


if __name__ == '__main__':
    main()