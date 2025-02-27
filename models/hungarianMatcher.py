"""
    Computes Hungarian matching

    Inspired by: https://github.com/facebookresearch/detr/blob/main/models/matcher.py
"""
import pdb
import time

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self):
        """Creates the matcher

        Params:

        """
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching with MSE loss

        Params:
            outputs: Tensor of dim [B, num_out_segments, outdim*lambda]
            targets: list of target gt_segments (len(targets) == batch_size), where each target is a Tensor of dim [num_gt_segments, outdim*lambda]

        Returns:
            A list of len batch_size, containing tuples of arrays (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        B, num_out_segments = outputs.shape[:2]

        # Stack along batch dimension for cost computation
        out_stacked_segments = outputs.flatten(0, 1)
        t_gt_segments = torch.cat([gt_segments for gt_segments in targets])

        cost_matrix = torch.cdist(out_stacked_segments, t_gt_segments, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')
        
        # Back to per-sample matrix
        sizes = [len(gt_segments) for gt_segments in targets]  # List[int] of size B with num of GT segments per sample
        cost_matrix = cost_matrix.view(B, num_out_segments, -1).cpu()
        per_sample_cost_matrix = cost_matrix.split(sizes, -1)  # list of size B of Tensors [B, out_num_segments, out_gt_segments] 

        indices = []
        for i, c in enumerate(per_sample_cost_matrix):
            # c[i]: Tensor of dim [out_num_segments, num_gt_segments for this sample]
            indices.append(linear_sum_assignment(c[i]))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
