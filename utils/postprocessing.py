import time
import sys
import pdb
import os
from threadpoolctl import ThreadpoolController
controller = ThreadpoolController()

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
try:
    from pytorch3d_chamfer import chamfer_distance 
except ImportError:
    print(f'Warning! Unable to import pytorch3d package.'\
          f'Chamfer distance with velocities won\'t be available.'\
          f'(Check troubleshooting.txt for info on how to install pytorch3d)')
    pass
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.pointcloud import from_seq_to_pc, get_dim_traj_points
from utils.cluster import concat_segments_of_stroke


def select_top_bboxes(batch_bboxes, threshold=0.05, confidence_logits=False):
    """
        filter out stroke proposals (as 3d bboxes)

        confidence_logits is False:
            iterative filter out boxes based on a distance threshold

        confidence_logits is True:
            use non-maximal suppression algorithm
    """
    out_batch_bboxes = []

    print('[Post-processing] 3D bboxes filtering:')
    for bboxes in batch_bboxes:
        
        filter_out = set()
        n_boxes = bboxes.shape[0]

        # pdb.set_trace()
        # pairwise_square_distances = pairwise_square_distances(torch.tensor(bboxes))

        tbboxes = torch.tensor(bboxes)
        distances = torch.cdist(tbboxes, tbboxes, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        # top_dists, ind = distances.topk(tbboxes.shape[0], largest=False, sorted=True, dim=-1)


        for i in range(n_boxes):

            if i in filter_out:
                continue

            filter_mask = distances[i, :] < threshold
            filter_mask[i] = False

            filter_indx = np.arange(n_boxes)[filter_mask].tolist()
            filter_out.update(filter_indx)

        all_idx = set(np.arange(n_boxes).tolist())
        retain_idx = list(all_idx - filter_out)
        retained_bboxes = bboxes[retain_idx]
        out_batch_bboxes.append(retained_bboxes)

        print(f'3D bboxes retained: {len(retain_idx)}/{n_boxes}')

    return out_batch_bboxes


def handle_end_of_sequence(sequence, confidence_logits, threshold):
    """
        
        sequence: (N, D) , sequence of vectors
        confidence_logits: (N, 1) , confidence score for each vector
    """
    high_confidence_mask = confidence_logits > threshold

    if np.all(high_confidence_mask):
        # All above confidence threshold
        return sequence
    else:
        first_low_conf_occurence = np.argmin(high_confidence_mask)
        return sequence[:first_low_conf_occurence, :]


def process_pred_stroke_masks_to_stroke_ids(pred_stroke_masks, confidence_scores, confidence_threshold=0.5):
    """Given mask scores, assign final stroke_ids to each segment.
        
        Post processing steps:
            1. Filtering
            2. Argmax (as in DETR)

    Params:
        pred_stroke_masks: [B, max_n_strokes, out_segments]
        confidence_scores: [B, max_n_strokes]
        confidence_threshold: threshold for a stroke mask to be considered valid.

    """
    assert pred_stroke_masks.ndim == 3, 'batch dimension is assumed to exist'
    assert confidence_scores.ndim == 2, 'batch dimension is asummed to exist'

    confidence_logits = torch.tensor(confidence_scores).sigmoid()
    prob_pred_stroke_masks = torch.tensor(pred_stroke_masks).sigmoid()

    ### TEMP STATS ON STROKE MASKS
    # max_n_strokes = 6  # assuming cuboids_v2
    # B = confidence_logits.shape[0]

    # n_strokes_retained = torch.sum(confidence_logits >= confidence_threshold) / B

    # print(f'Avg strokes retained per sample: {n_strokes_retained.item()} / {max_n_strokes}')
    # print(f'Avg confidence prob: {torch.mean(confidence_logits)}')
    # print(f'Avg segment prob: {torch.mean(prob_pred_stroke_masks)}')
    # sys.exit()
    ##############################


    """
        Filter out low-confidence masks (30% MaskFormer, 85% DETR)
    """
    filter_out_idx = torch.where(confidence_logits < confidence_threshold) 
    prob_pred_stroke_masks[filter_out_idx] = 0.  # temp: simply set the mask probs to 0. TODO: do this cleanly when post-processing is figured out.

    # prob_pred_stroke_masks[prob_pred_stroke_masks > sigmoid_threshold] = 1.
    # prob_pred_stroke_masks[prob_pred_stroke_masks < sigmoid_threshold] = 0.

    prob_pred_stroke_masks = prob_pred_stroke_masks.numpy()


    """
        Per-segment argmax
    """
    B, out_masks, out_segments = prob_pred_stroke_masks.shape

    stroke_ids_pred = np.zeros((B, out_segments)) - 1

    for b, b_prob_pred_stroke_masks in enumerate(prob_pred_stroke_masks):  # iterate over batch elements

        # Assign segments to stroke via argmax
        assoc_mask_id_per_segment = np.argmax(b_prob_pred_stroke_masks, axis=0)

        # Generate contiguous stroke_ids, i.e. from 0 on
        for i, stroke_id in enumerate(np.unique(assoc_mask_id_per_segment)):
            stroke_ids_pred[b, assoc_mask_id_per_segment == stroke_id] = i

    return stroke_ids_pred


@controller.wrap(limits=8, user_api='openmp')
def process_stroke_segments(traj,
                            stroke_ids,
                            config,
                            skip_segments_filtering=False,
                            segments_filtering_only=False,
                            no_interpolation=False,
                            fast_concat=False,
                            verbose=0):
    """Process predicted segments.

        1. Filter out segments too close to each other
        2. Concatenation: Directed MST -> Interpolate -> up-sample -> filter -> down-sample

        TODO: return discarded segments with stroke_id=-1 (both from filtering and concatenation)
    """


    """
        per-stroke filtering of overlapping segments
        
        [Possible improvement: use learned confidence for informed suppression (non-maximal suppression)]
    """
    filter_segments_percentage = 1  # 1. vs. 0.25 vs. 0.0001 # x-% of segments within a stroke are filtered out, if there are at least 1/x segments
    
    """
        Avoid filtering segments if they are not "close enough", using `filter_distance_threshold`.
        For traj_sampling_v2, you can set the distance threshold by reasoning on the allowed pose-to-pose distance, and say that
        if all point-wise distances of two segments are below config['equal_spaced_points_distance'] => you discard this segment.
        In other words, distance_threshold = sqrt(  (config['equal_spaced_points_distance']**2) * lambda ). See picture of 24_06_06 if in doubt.
    """
    filter_distance_threshold = 0.1  # 0.1
    out_traj = traj.copy()
    out_stroke_ids = stroke_ids.copy()
    if filter_segments_percentage != 0 and not skip_segments_filtering:
        print(f'[POST-PROCESSING] per-stroke filtering of overlapping segments. Filter at most {filter_segments_percentage*100}% segments out of each stroke.')
        for b, (b_traj, b_stroke_ids) in tqdm(enumerate(zip(traj, stroke_ids))):  # iterate over batch elements
            for b_stroke_id in np.unique(b_stroke_ids):  # iterate over strokes
                stroke = b_traj[b_stroke_ids == b_stroke_id]
                curr_stroke_ids = b_stroke_ids[b_stroke_ids == b_stroke_id]
                n_segments = stroke.shape[0]

                if n_segments > (1/filter_segments_percentage):
                    # Filter out x-% of segments that are too close to each other
                    to_filter = int(n_segments * filter_segments_percentage)

                    filtered = 0
                    filtered_stroke = stroke.copy()
                    filtered_stroke_ids = curr_stroke_ids.copy()

                    remained_stroke = stroke.copy()  # separate copy kept for code flexibility
                    list_removed_idx = []
                    while filtered < to_filter:
                        distances = torch.cdist(torch.tensor(remained_stroke), torch.tensor(remained_stroke), p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
                        top_dists, ind = distances.topk(2, largest=False, sorted=True, dim=-1)
                        
                        top_dists = top_dists[:, 1].numpy()  # remove self-distance
                        ind = ind[:, 1].numpy()  # remove self-index
                        
                        # Top_dists as np masked array: invalid elements are those that have already been filtered out.
                        # This way, the np.argmin below does not take these entries into account.
                        invalid = np.zeros(top_dists.shape[0], dtype=bool)
                        invalid[list_removed_idx] = True
                        masked_top_dists = np.ma.array(top_dists, mask=invalid)

                        # Stop filtering out segments if no pair is closest than the minimum threshold required for filtering.
                        if np.all(masked_top_dists > filter_distance_threshold):
                            break

                        remove_idx = np.argmin(masked_top_dists)

                        # Filter out segment by setting it to the padding value of -100, and stroke id of -1 
                        filtered_stroke[remove_idx] = -100  # padding value [COMMENT THIS OUT TO SEE THE FILTERED OUT SEGMENTS IN BLACK COLOR DURING VISUALIZATION]
                        filtered_stroke_ids[remove_idx] = -1  # padding value

                        remained_stroke[remove_idx] = -100
                        filtered += 1
                        list_removed_idx += [remove_idx]

                    out_traj[b, out_stroke_ids[b] == b_stroke_id] = filtered_stroke
                    out_stroke_ids[b, out_stroke_ids[b] == b_stroke_id] = filtered_stroke_ids

    traj = out_traj.copy()
    stroke_ids = out_stroke_ids.copy()

    if segments_filtering_only:
        return traj, stroke_ids


    """
        Concatenate segments within each stroke
    """
    print('[POST-PROCESSING] Concat segments within each stroke')
    out_traj = []
    out_stroke_ids = []
    Interpolation = Akima1DInterpolator
    upsample_factor = 5
    outdim = get_dim_traj_points(config['extra_data'])

    for b, (b_traj, b_stroke_ids) in tqdm(enumerate(zip(traj, stroke_ids))):  # iterate over batch elements
        b_out_traj = np.empty((0, outdim))
        b_out_stroke_ids = np.empty((0,))
        for b_stroke_id in np.unique(b_stroke_ids):  # iterate over strokes
            if b_stroke_id == -1:
                continue    

            curr_stroke = b_traj[b_stroke_ids == b_stroke_id]
            curr_stroke_ids = b_stroke_ids[b_stroke_ids == b_stroke_id]
            n_segments = curr_stroke.shape[0]

            curr_stroke = concat_segments_of_stroke(curr_stroke, curr_stroke_ids, config, verbose=verbose)

            # Remove poses that overlap among adjecent segments
            min_space_among_points_for_overlap = 0.05  # suggested to be equal to config['equal_spaced_points_distance']
            non_overlapping_points = find_non_overlapping_points_among_adjecent_segments(curr_stroke, distance=min_space_among_points_for_overlap, outdim=outdim)
            
            curr_stroke = from_seq_to_pc(curr_stroke, extra_data=config['extra_data'])  # this has to be done regardless of `find_non_overlapping_points_among_adjecent_segments`

            curr_stroke = curr_stroke[non_overlapping_points]
            
            if not fast_concat:
                if no_interpolation:
                    # if no_interpolation, then simply resample at the same distance used for traj_sampling_v2
                    min_space_among_points = config['equal_spaced_points_distance'] if 'equal_spaced_points_distance' in config and config['equal_spaced_points_distance'] is not None else 0.05
                    assert 'equal_spaced_points_distance' in config and config['equal_spaced_points_distance'] is not None, 'Temp sanity check. It\'s not mandatory, but I\'m expecting traj_sampling_v2 here.'
                    curr_stroke = resample_at_equal_spaced_points_except_last(curr_stroke, distance=min_space_among_points)
                
                else:
                    """
                        Filter stroke by constraining min distance among points

                        if not find_non_overlapping_points_among_adjecent_segments:
                            `min_space_among_points` should be at least (lambda-2) * config['equal_spaced_points_distance'].
                            This is so that two adjecent segments which overlap by the maximum amount still have last-first pose which are sufficiently further apart.
                        else:
                            `min_space_among_points` can be simply set equal to config['equal_spaced_points_distance'] or 2 * config['equal_spaced_points_distance']
                    """
                    min_space_among_points = 0.1  # 0.1
                    curr_stroke = resample_at_equal_spaced_points_except_last(curr_stroke, distance=min_space_among_points)  # retain for best pre-processing
                    # curr_stroke = subsample_with_min_distance_among_points(curr_stroke, min_distance=min_space_among_points)  # TODO: *_except_last, i.e. make sure last is retained regardless
                    # ---------------------------------------------------------

                
                    # Retain for best processing ----
                    times = np.arange(0, curr_stroke.shape[0])
                    curve = Interpolation(times, curr_stroke)
                    upsample_times = np.arange(0, curr_stroke.shape[0], step=1/upsample_factor)
                    curr_stroke = curve(upsample_times)
                    # Remove nans
                    mask = np.isnan(curr_stroke)
                    curr_stroke = curr_stroke[~mask[:, 0], :]
                    # -------------------------------


                    # Retain for best processing ----
                    sigma = 2
                    curr_stroke = apply_filter_to_each_dim(curr_stroke, kind='gaussian', sigma=sigma)
                    # -------------------------------
                

            curr_stroke_ids = np.repeat(b_stroke_id, curr_stroke.shape[0])

            b_out_traj = np.append(b_out_traj, curr_stroke, axis=0)
            b_out_stroke_ids = np.append(b_out_stroke_ids, curr_stroke_ids, axis=0)


        out_traj.append(b_out_traj)
        out_stroke_ids.append(b_out_stroke_ids)

    return out_traj, out_stroke_ids


def apply_filter_to_each_dim(signal, kind='gaussian', **kwargs):
    """Apply a filter to the signal, independently for each dimension

    Params:
        signal: [N, D] signal
    """
    assert signal.ndim == 2
    D = signal.shape[-1]

    out_signal = signal.copy()

    for dim in range(D):
        if kind == 'gaussian':
            out_signal[:, dim] = gaussian_filter1d(out_signal[:, dim], **kwargs)
        elif kind == 'median':
            window = kwargs['window']
            out_signal[window//2:-(window//2), dim] = medfilt(out_signal[window//2:-(window//2), dim], window)
        else:
            raise ValueError(f'Invalid kind of filtering: {kind}')

    return out_signal


def find_non_overlapping_points_among_adjecent_segments(stroke, distance, outdim):
    """
        Given the order of concatenated segments, the goal is to remove
        poses that are overlapping among two consecutive segments.
        Therefore, for each consecutive pair of (i, j) segments, we
        always trust the starting point of segment j, and potentially remove
        the last k poses of segment i depending on whichever is the first pose
        of segment i which comes closer than distance to the first pose of segment j.

        stroke : (N, lambda * outdim)
                 
    """
    N, D = stroke.shape
    lambda_points = D // outdim
    assert D % outdim == 0, 'unexpected error. stroke.shape[-1] should be lambda*outdim, i.e. expressed segment-wise'

    out_points_indexes = []

    for i in range(N-1):

        curr_segment = stroke[i].reshape(lambda_points, outdim)
        next_segment = stroke[i+1].reshape(lambda_points, outdim)

        # overlapping = [lambda_points-1, 0]  # by default, no overlapping. I.e., last point of curr_segment with first point of next_segment
        overlapping_found = False

        for j, curr_point in enumerate(curr_segment):
            d = np.linalg.norm(curr_point - next_segment[0])

            if d < distance:
                overlapping_found = True
                break

        if overlapping_found:
            out_points_indexes += (np.arange(j+1) + lambda_points*i).tolist()
        else:
            out_points_indexes += (np.arange(lambda_points) + lambda_points*i).tolist()


    out_points_indexes += (np.arange(lambda_points) + lambda_points*(N-1)).tolist()  # add full last segment

    return out_points_indexes


def resample_at_equal_spaced_points_except_last(stroke, distance):
    """Given the input points, interpolate them and re-sample
       the signal with dynamic time intervals such that consecutive points
       are equally spaced.

       *_except_last: explicitly point out that this function should not be used in the pre-processing
       phase, as it's more of a filtering 
    """
    assert stroke.ndim == 2
    N, D = stroke.shape

    out_points = np.empty((0, D))

    times = np.arange(0, N)
    curve = Akima1DInterpolator(times, stroke)

    last_point = stroke[0, :].copy()  # i.e. first point of the stroke added -> curve(times[0])
    out_points = np.append(out_points, last_point[None, :], axis=0)
    dt = 0.1  # discrete time interval to check whether a new point should be sampled
    curr_t = 0.1
    while True:
        # Exit when the stroke finishes
        if curr_t > times[-1] + 0.0005:  # beyond last one
            break

        curr_point = curve(curr_t)

        d = np.linalg.norm(curr_point - last_point)

        # TODO: handle last one differently, so that the threshold is only half of `distance`, and end if curr_t > times[-1] + dt + 0.0005
        if d > distance:
            last_point = curr_point.copy()
            out_points = np.append(out_points, last_point[None, :], axis=0)

        curr_t += dt

    out_points = np.append(out_points, stroke[-1, :].copy()[None, :], axis=0)  # add last point --- temporary until TODO above is resolved.

    return out_points


def subsample_with_min_distance_among_points(curr_stroke, min_distance):
    """Subsample the input points by ensuring that consecutive points are
        at least `min_distance` apart from each other
    """
    out_points = np.empty((0, 6))

    last_point = curr_stroke[0, :]
    out_points = np.append(out_points, last_point[None, :], axis=0)
    for i, point in enumerate(curr_stroke):
        if i == 0:
            continue

        distance = np.linalg.norm(point - last_point)

        if distance > min_distance:
            last_point = point.copy()
            out_points = np.append(out_points, last_point[None, :], axis=0)

    return out_points


def permute_and_align_stroke_ids_for_visualization(pred_stroke_masks, y_pred, y, pred_stroke_ids, stroke_ids):
    """Find a new stroke_ids permutation to match the order of GT stroke masks.

        In particular, you find the best match among predicted masks and GT masks (as projection to pred segments).
        Then, you use the matched masks to build new stroke_ids values.    
    """
    if torch.is_tensor(pred_stroke_masks):
        pred_stroke_masks = pred_stroke_masks.float().cuda()
    elif isinstance(pred_stroke_masks, list):
        pred_stroke_masks = [item.float().cuda() for item in pred_stroke_masks]    

    y_pred = torch.tensor(y_pred).cuda()
    y = torch.tensor(y).cuda()
    stroke_ids = torch.tensor(stroke_ids).cuda()
    pred_stroke_ids = torch.tensor(pred_stroke_ids).cuda()

    _, _, pred_to_gt_match, _ = chamfer_distance(y_pred, y, padded=True, return_matching=True)  # pred_to_gt_match: [B, num_pred_segments]

    # Assign stroke_ids to pred segments according to closest GT segment
    target_stroke_ids = stroke_ids.cuda().gather(dim=1, index=pred_to_gt_match)  # target_stroke_ids [B, out_segments]

    assert not torch.any(target_stroke_ids == -1), 'temp sanity check: no pred segment should be associated with the fake stroke id -1'

    # Create binary stroke masks from stroke ids,
    target_stroke_masks = [from_stroke_ids_to_masks(batch_target_stroke_ids).float()
                           for batch_target_stroke_ids in target_stroke_ids]  # list of size B [n_strokes[b], out_segments]

    new_stroke_ids, max_n_strokes = match_stroke_masks(target_stroke_masks=pred_stroke_masks,
                                                       pred_stroke_masks=target_stroke_masks,  # these are "inverted" so that pred stroke ids are mapped to GT stroke_ids, and not viceversa.
                                                       old_stroke_ids=pred_stroke_ids,
                                                       out_segments=y_pred.shape[1])

    new_stroke_ids = new_stroke_ids.detach().cpu().numpy()

    return new_stroke_ids, max_n_strokes


def match_stroke_masks(target_stroke_masks, pred_stroke_masks, old_stroke_ids, out_segments):
    """Match stroke masks with hungarian algorithm (target stroke id matched
       using closest GT segment to pred i-th segment), then map old_stroke_ids
       into new values to follow the predicted mask order.
    """
    smooth_targets = False
    stroke_mask_loss_kind = 'bce'

    # Temp sanity checks
    if not smooth_targets:
        assert torch.all(torch.stack([torch.all(b_target_stroke_mask.sum(0) == 1) for b_target_stroke_mask in target_stroke_masks])), 'temp sanity check: masks should be mutually exclusive across strokes, hence all equal to ones when summed.'

    """
        Find hungarian matching between pred_stroke_masks and target_stroke_masks
    """
    indices = []
    with torch.no_grad():
        for b, (b_pred_stroke_masks, b_target_stroke_masks) in enumerate(zip(pred_stroke_masks, target_stroke_masks)):  # iterate over batch elements
            # Compute cost matrix for this batch
            n_target_masks = b_target_stroke_masks.shape[0]
            n_pred_masks = b_pred_stroke_masks.shape[0]

            # all pairs in single-column format for loss computation ([n_pred_masks, n_target_masks] as [n_pred_masks*n_target_masks, 1])
            exp_b_pred_stroke_masks = b_pred_stroke_masks.repeat_interleave(n_target_masks, dim=0)
            exp_b_target_stroke_masks = b_target_stroke_masks.repeat(n_pred_masks, 1)
            
            bce = _compute_stroke_mask_loss(exp_b_pred_stroke_masks, exp_b_target_stroke_masks.float(), kind=stroke_mask_loss_kind)
            bce = bce.view(n_pred_masks, n_target_masks).cpu()  # cost matrix [n_pred_masks, n_target_masks]
            
            indices.append(linear_sum_assignment(bce))  # solve LAP

    indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  # list of size B, with elements (index_i, index_j). index_* is Tensor

    # Map old stroke_ids to new stroke_ids
    max_n_strokes = []
    new_stroke_ids = old_stroke_ids.clone().int()
    for b, (b_pred_indices, b_gt_indices) in enumerate(indices):  # batch dim
        n_target_strokes = target_stroke_masks[b].shape[0]
        n_pred_strokes = pred_stroke_masks[b].shape[0]

        # To map old stroke ids to new ones, use intermediate temp values to store the new value, i.e. +1 and *1000
        temp_values = []
        for b_pred_index, b_gt_index in zip(b_pred_indices, b_gt_indices):
            new_stroke_ids[b, new_stroke_ids[b] == b_gt_index] = (b_pred_index+1)*1000
            temp_values.append((b_pred_index+1)*1000)

        # Predicted strokes can be more or less than the GT.
        # When predicting more than the GT, make sure that any stroke id which is not mapped to any GT (i.e. exceeding) is remapped to a new id which does not exist in the GT.
        # (remind that target and pred are actually provided as inverted in the code.)
        if n_target_strokes > n_pred_strokes:
            n_exceeding_strokes = n_target_strokes - n_pred_strokes 
            n_remapped_strokes = 0

            current_idx = b_pred_indices.max() + 1
            for idx in b_pred_indices:  # b_pred_indices contain the new values that will be applied. Make sure that any previous cell with such values is remapped to a new id beyond these.
                if new_stroke_ids[b, new_stroke_ids[b] == idx].shape[0] > 0:
                    new_stroke_ids[b, new_stroke_ids[b] == idx] = current_idx
                    n_remapped_strokes += 1
                    current_idx += 1

            if n_remapped_strokes < n_exceeding_strokes:
                # there are remaining exceeding predicted stroke ids that have not been matched with an already existing stroke id.
                # remap these remaining ones contiguously.
                remaining_idx = set(range(n_target_strokes)) - set(b_pred_indices.tolist())

                for idx in remaining_idx:
                    if new_stroke_ids[b, new_stroke_ids[b] == idx].shape[0] > 0:
                        new_stroke_ids[b, new_stroke_ids[b] == idx] = current_idx
                        current_idx += 1

        # Go from temporary vale to actual value, i.e. divide by 1000 and subtract 1
        for temp_value in temp_values:
            new_stroke_ids[b, new_stroke_ids[b] == temp_value] = new_stroke_ids[b, new_stroke_ids[b] == temp_value] // 1000 - 1

        max_n_strokes.append(max(n_target_strokes, n_pred_strokes))

    return new_stroke_ids, max_n_strokes


def _compute_stroke_mask_loss(input, target, kind='bce'):
    """Compute loss on given stroke masks.
        No batch reduction is performed.
    """
    if kind == 'bce':
        return F.binary_cross_entropy_with_logits(input, target, reduction="none").sum(-1)
    elif kind == 'mse':
        return (input - target).square().sum(-1)
    else:
        raise NotImplementedError()


def _from_masks_to_stroke_ids(stroke_masks):
    """
        TODO
    """
    # assert stroke_masks.ndim == 2, 'expected batch dim'

    # B, n_pred_masks, out_segments = stroke_masks

    # stroke_ids = torch.zeros((B, out_segments))

    # for b, b_stroke_mask in enumerate(stroke_masks):
    #     stroke_ids = 
    raise NotImplementedError()


def from_stroke_ids_to_masks(stroke_ids, smooth_targets=False, nn_distance=None):
    """Returns n_strokes binary masks given the stroke_ids tensor
        
    Params:
        stroke_ids: Tensor of dim [N] with K unique values (stroke ids)
        smooth_targets: if set, real-valued masks in output instead of binary masks
        nn_distance: Tensor of dim [N] with distance to nearest GT segment 
        
        N: num of segments
    Returns:
        Tensor of dim [K, N] with binary stroke masks
    """
    assert stroke_ids.ndim == 1, 'a batch dimension is not expected'
    stroke_ids = to_tensor(stroke_ids)

    stroke_masks = []
    for stroke_id in torch.unique(stroke_ids):
        if stroke_id == -1:  # padding value for fake segments
            continue

        stroke_mask = (stroke_ids == stroke_id).int()

        if smooth_targets:
            segments_confidence = self._transform_segment_distance_to_confidence(nn_distance)
            segments_in_stroke_mask = stroke_mask == 1

            stroke_mask = stroke_mask.float()  # from binary to real-valued
            stroke_mask[segments_in_stroke_mask] = segments_confidence[segments_in_stroke_mask]

        stroke_masks.append(stroke_mask)
    return torch.stack(stroke_masks)


def to_tensor(arr):
    if not torch.is_tensor(arr):
        return torch.tensor(arr)
    else:
        return arr


def postprocess_sop_predictions(sop_pred, pred_sop_conf_scores, sop_conf_threshold=0.5, return_retained_idx=False):
    """SoP prediction inference time

        Sub sample SoPs pred using the learned confidence scores.
    """
    B, _, _ = sop_pred.shape
    # outdim = get_dim_traj_points(config.extra_data)

    sop_probs = pred_sop_conf_scores.sigmoid()
    
    # Sub sample SoPs
    sops, retained_idx = [], []
    for b in range(B):
        retained_sop_idx = torch.where(sop_probs[b] > sop_conf_threshold)
        retained_sops = sop_pred[b][retained_sop_idx]

        if retained_sops.shape[0] == 0:
            # No SoPs retained for this sample.
            print('\nWARNING! No stroke prototypes (SoPs) were retained for this sample. Is the confidence threshold too high?')
            sops.append([])
            retained_idx.append([])
        else:
            sops.append(retained_sops)
            retained_idx.append(retained_sop_idx)

    if return_retained_idx:
        return sops, retained_idx
    else:
        return sops


def postprocess_sample_autoregressive_predictions_into_strokes(paths, eop_logits, config, eop_conf_threshold=0.5):
    """
        Autoregressive_v2 inference time: from autoregressive network predictions to strokes

        In practice, use End-of-Path (EoP) confidence logits to sub select the stroke lengths.
        ---
            paths: Tensor of size [n_strokes, max_n_segments_per_stroke, outdim*lambda]
            eop_logits: Tensor of size [n_strokes, max_n_segments_per_stroke, 1]
                        confidence score indicating likelihood of segment to be an End-of-Path segment.
        ---
        Returns:
            paths_as_pc : Tensor of size [n_strokes, max_n_points_per_stroke*outdim]  padded along dim = 1
                          segments beyond the predicted length are padded with -100 values.
    """
    outdim = get_dim_traj_points(config.extra_data)

    eop_probs = eop_logits.sigmoid()
    eop_probs[:, -1, 0] = 1  # At least last one should be set as last (prevents argmax to yield wrong results).
    eop_probs[:, 0, 0] = 0 # At least one is retained.
    stroke_lengths = torch.argmax((eop_probs > eop_conf_threshold).int(), dim=1)  # find first segment whose EoP is above eop_conf_threshold

    retain_segments_mask = torch.arange(eop_probs.shape[1]).repeat((eop_probs.shape[0], 1))
    retain_segments_mask = retain_segments_mask <= stroke_lengths

    paths[~retain_segments_mask] = -100  # pad all discarded segments
    
    # [n_strokes, max_n_points_per_stroke, outdim]
    # paths_as_pc = paths.view(paths.shape[0], -1, outdim)

    # [n_strokes, max_n_points_per_stroke*outdim]
    paths_as_pc = paths.view(paths.shape[0], -1)

    return paths_as_pc


def postprocess_strokewise_predictions_into_strokes(strokes,
                                                    point_scores,
                                                    stroke_scores,
                                                    config,
                                                    stroke_conf_threshold=0.5,
                                                    point_conf_threshold=0.5,
                                                    device='cuda'):
    """StrokeWise inference time: from network predictions to strokes
        
        In practice, use stroke and point confidences to sub select
        strokes and their respective lengths.

        stroke_conf_threshold : confidence on wether stroke is real
        point_conf_threshold : confidence on whether the point is real (not beyond the learned length)
    """
    B, _, _ = strokes.shape
    outdim = get_dim_traj_points(config.extra_data)

    point_logits = point_scores.sigmoid()
    stroke_logits = stroke_scores.sigmoid()

    # Select strokes
    traj = []
    for b in range(B):
        # Sub-select predicted strokes above confidence threshold 
        retained_stroke_idx = torch.where(stroke_logits[b] > stroke_conf_threshold)
        retained_strokes = strokes[b][retained_stroke_idx]

        # Transform into Point-wise format [max_n_strokes, max_n_stroke_points, outdim] 
        retained_strokes = retained_strokes.reshape(retained_strokes.shape[0], -1, outdim)

        # Select lengths by stopping at first point below confidence threshold
        retained_point_logits = point_logits[b][retained_stroke_idx]
        
        # TEMP TEMP TEMP: THIS REQUIRES FIXING THE ARGMAX WHICH COULD YIELD ZERO-LENGTH STROKES IF ALL POINTS ARE BELOW point_conf_threshold.
        # pdb.set_trace()
        # retained_point_logits[-1] = 0  # or something like this.

        retained_lengths = torch.argmax((retained_point_logits < point_conf_threshold).int(), dim=-1)
        retained_lengths = retained_lengths.unsqueeze(-1).repeat((1, retained_point_logits.shape[-1]))

        retained_lengths_as_mask = torch.arange(retained_point_logits.shape[-1]).repeat((retained_point_logits.shape[0], 1)).to(device)
        retained_lengths_as_mask = retained_lengths_as_mask < retained_lengths

        # Mark all predicted points beyond the predicted length as `fake` points
        retained_strokes[~retained_lengths_as_mask] = -100

        # Back in stroke-wise format [retained_n_strokes, max_n_stroke_points*outdim]
        retained_strokes = retained_strokes.reshape(retained_strokes.shape[0], -1)
        traj.append(retained_strokes)

    return traj


def from_strokewise_to_pointwise(strokes, config, return_stroke_ids=True, remove_padding=True, device='cpu'):
    """from stroke-wise format to point-wise format.

        strokes: [N, max_n_stroke_points*outdim]
    """
    assert strokes.ndim == 2, 'batch dimension is not expected'
    N, _ = strokes.shape
    outdim = get_dim_traj_points(config.extra_data)

    strokes = to_tensor(strokes)
    strokes = strokes.to(device)

    strokes = strokes.clone().reshape(N, -1, outdim)  # [n_strokes, max_n_stroke_points, outdim]
    _, n_points_per_stroke, _ = strokes.shape


    # From stroke-wise format into point-wise
    strokes = strokes.reshape(N*n_points_per_stroke, outdim)


    # Generate associated point-wise stroke_ids
    if return_stroke_ids:
        stroke_ids = torch.arange(N)[:, None, None].repeat(1, n_points_per_stroke, 1).to(device)  # [n_strokes, max_n_stroke_points, 1]
        stroke_ids = stroke_ids.reshape(N*n_points_per_stroke)


    # Remove pending fake points in the input strokes
    if remove_padding: 
        fake_points_mask = torch.all(torch.isclose(strokes, torch.tensor(-100).float()), dim=-1)
        strokes = strokes[~fake_points_mask].clone()

        if return_stroke_ids:
            stroke_ids = stroke_ids[~fake_points_mask].clone()


    if return_stroke_ids:
        return strokes, stroke_ids
    else:
        return strokes


def remove_padding_from_tensors(tensors):
    """From an array of tensors, 
        remove the fake tensors

        tensors : (N, D)
                   some of the N tensors are fake,
                   and filled with -100 values

        ---
        returns
            out_vectors : (M, D)
                          where M is the number of true tensors
    """
    assert tensors.ndim == 2
    fake_mask = torch.all((tensors[:, :] == -100), axis=-1)  # True for fake tensors
    tensors = tensors[~fake_mask]
    return tensors