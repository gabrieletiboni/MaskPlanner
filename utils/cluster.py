import pdb
import math

import torch
import numpy as np
import networkx as nx
from networkx.algorithms.tree import Edmonds

from utils.pointcloud import from_seq_to_pc, get_dim_traj_points


def concat_segments_of_stroke(traj, stroke_ids, config, verbose=0):
    """Given segments of a stroke, concatenate them based on proximity and alignment
        
        Use (1) minimum spanning tree + dag_longest_path to retrieve the final path
    
    Params
        traj: [N, D]
               N: num segments of this stroke, D: outdim
        stroke_ids: [N,]
    """
    outdim = get_dim_traj_points(config['extra_data'])
    assert traj.shape[1] == outdim*config['lambda_points']
    if not torch.is_tensor(traj):
        traj = torch.tensor(traj)
    n_segments = traj.shape[0]

    # Parameters -------
    vel_weight = 1.5
    radius = 0.2  # sparsification based on distance
    # sparsification = 0.5  # x-% sparsification from fully-connected graph (the lower the sparser)
    # k = int(n_segments * sparsification)  
    k = 5  # directly specified the knn instead of sparsification percentage (works better for strokes of varying length, as short strokes would suffer from sparsification)
    # ------------------

    k = min(n_segments, k)
    

    starting_points = traj[:, :outdim]
    ending_points = traj[:, -outdim:]
    inferred_vel_starting = vel_weight*(traj[:, outdim:outdim+3] - traj[:, :3])
    inferred_vel_ending = vel_weight*(traj[:, -outdim:-(outdim-3)] - traj[:, -(outdim*2):-(outdim*2-3)])
    starting_points = torch.cat((starting_points, inferred_vel_starting), dim=-1)
    ending_points = torch.cat((ending_points, inferred_vel_starting), dim=-1)
    
    # pc = from_seq_to_pc(traj, config['extra_data'])
    # mean_knn_distance(pc[:, :], k=1, render=True)

    distances = torch.cdist(ending_points, starting_points, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    top_dists, ind = distances.topk(k, largest=False, sorted=True, dim=-1)
    top_dists = top_dists.square()
    ind = ind.cpu().numpy()

    # Create graph
    seq_G = nx.DiGraph()
    seq_G.add_nodes_from(np.arange(ending_points.shape[0]))

    # Add edges to the graph according to k-nn
    for i, node_neighbours in enumerate(ind):  # iterate over nodes, node_neighbours: list of neighbours of node
        for kth, j in enumerate(node_neighbours):
            if i != j and top_dists[i, kth] < radius:  # do not concatenate ending and starting point of the same segment
                seq_G.add_edge(i, j, weight=top_dists[i, kth])

    # print('N nodes:', seq_G.number_of_nodes())
    # print('N edges:', seq_G.number_of_edges())
    # print('N strongly CCs:', nx.number_strongly_connected_components(seq_G))  # Should be = number_of_nodes()
    # print('N weakly CCs:', nx.number_weakly_connected_components(seq_G))  # Should be uqual to those of undi_seq_G
    # print('Is DAG?', nx.is_directed_acyclic_graph(seq_G))
    # print('---')

    # undi_seq_G = seq_G.to_undirected()

    # Find minimum UNDIRECTED spanning tree
    # undi_seq_G = nx.to_undirected(seq_G)
    # mst = nx.minimum_spanning_tree(undi_seq_G)
    # invert_weights(undi_seq_G)
    # pdb.set_trace()

    # longest_path = nx.dag_longest_path(mst)  # list of vertex indx
    # -------------------------------------

    # Find minimum directed spanning tree
    Edm = Edmonds(seq_G)
    directed_mst = Edm.find_optimum(kind='min', style='spanning arborescence')
    drop_weights(directed_mst)  # this I think is right
    # invert_weights(directed_mst)  # this I think is wrong
    # root = [n for n,d in directed_mst.in_degree() if d==0][0]
    # longest_path = find_longest_shortest_path(directed_mst, root)
    longest_path = nx.dag_longest_path(directed_mst)  # list of vertex indx
    # -----------------------------------

    # print('N nodes:', seq_G.number_of_nodes())
    # print('N edges:', seq_G.number_of_edges())
    # print('N nodes in longest path:', len(longest_path))

    if verbose > 0:
        if seq_G.number_of_nodes() != len(longest_path):
            print(f'this stroke discarded {(seq_G.number_of_nodes() - len(longest_path))}/{seq_G.number_of_nodes()} segments when concatenating!')

    stroke = traj[longest_path].clone().numpy()

    return stroke

def invert_weights(G):
    '''Drop the weights from a networkx weighted graph.'''
    for node, edges in nx.to_dict_of_dicts(G).items():
        for edge, attrs in edges.items():
            attrs['weight'] = 1/attrs['weight']

def drop_weights(G):
    '''Drop the weights from a networkx weighted graph.'''
    for node, edges in nx.to_dict_of_dicts(G).items():
        for edge, attrs in edges.items():
            attrs.pop('weight', None)

def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]