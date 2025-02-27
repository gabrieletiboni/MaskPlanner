"""Class for implementing and computing evaluation metrics"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pdb

import numpy as np
import torch
try:
    from pytorch3d_chamfer import chamfer_distance 
except ImportError:
    print(f'Warning! Unable to import pytorch3d package.'\
          f'Chamfer distance with velocities won\'t be available.'\
          f'(Check troubleshooting.txt for info on how to install pytorch3d)')
    pass
from sklearn.metrics.cluster import v_measure_score

from utils.pointcloud import get_dim_traj_points
from utils.postprocessing import remove_padding_from_tensors, postprocess_sop_predictions, process_pred_stroke_masks_to_stroke_ids


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor


class MetricsHandler():
    """Handle computation of evaluation metrics.

    E.g. compute pose-wise chamfer distance between
    predicted mini-sequences and ground-truth
    on the test set.
    """

    def __init__(self,
                 config,
                 metrics=[],
                 renormalize_output_config : Dict = {}
                ):
        """
        Parameters:
            metrics : list of str
                      metrics to be computed
        """
        super(MetricsHandler, self).__init__()
        self.metrics = metrics
        self.metrics_names = [
                    'pcd',
                    'chamfer_original',  # Unbalanced: comparison is done with entire original GT point-cloud (untrimmed due to lambda-sequences)
                    'stroke_chamfer',
                    'clustering_metrics',
                    'sop_metrics',
                    'sop_metrics_v2',
                    'stroke_masks_metrics',
                    'strokewise_num_of_strokes_metrics'
                  ]

        # more than a single value may be output by a single function, hence a tuple is expected
        self.output_metrics_names = [
                    ('point-wise chamfer distance',),
                    ('chamfer original',),
                    ('stroke chamfer distance',),
                    ('v_measure', 'adjusted_rand_score', 'avg_num_of_outliers'),
                    (
                        'avg_num_of_pred_sops',
                        'avg_num_of_gt_sops',
                        'avg_ratio_pred_over_gt_sops',
                        'avg_num_of_pred_sops_if_higher_threshold',
                        'avg_num_of_pred_sops_if_lower_threshold',
                        'avg_ratio_pred_over_gt_sops_if_higher_threshold',
                        'avg_ratio_pred_over_gt_sops_if_lower_threshold'
                    ),
                    (
                        'perc_correct_n_strokes',
                        'avg_num_of_pred_strokes',
                        'avg_num_of_gt_strokes',
                        'mean_absolute_error_NoP',
                        'avg_num_of_pred_strokes_if_higher_threshold',
                        'avg_num_of_pred_strokes_if_lower_threshold',
                        'mean_absolute_error_NoP_if_higher_threshold',
                        'mean_absolute_error_NoP_if_lower_threshold'
                    ),
                    (
                        'perc_correct_n_strokes',
                        'avg_num_of_pred_strokes',
                        'avg_num_of_gt_strokes',
                        'mean_absolute_error_NoP'
                    ),
                    (
                        'perc_correct_n_strokes',
                        'avg_num_of_pred_strokes',
                        'avg_num_of_gt_strokes',
                        'mean_absolute_error_NoP'
                    )
                  ]

        self.metric_functions = [
                    self.get_pcd,
                    self.get_chamfer_original,
                    self.get_stroke_chamfer,
                    self.get_clustering_metrics,
                    self.get_sop_metrics,
                    self.get_sop_metrics_v2,
                    self.stroke_masks_metrics,
                    self.strokewise_num_of_strokes_metrics
                  ]

        self.metric_index = {metric: i for i, metric in enumerate(self.metrics_names)}
        self.config = config

        # Handle renormalization of output trajectories to a different data_scale_factor for metric computation
        self.renormalize_output_config = renormalize_output_config
        self.renormalize_output = False
        if 'active' in self.renormalize_output_config and self.renormalize_output_config['active']:
            assert self.config['normalization'] == 'per-dataset'
            self.renormalize_output = True
        

    def get_eval_metric(self, metric, **kwargs):
        """Compute single metric"""
        assert metric in self.metrics_names, f"metric {metric} is not valid"
        metric = self.metric_functions[self.metric_index[metric]](**kwargs)
        return metric


    def compute(self, **kwargs):
        """Compute all metrics in self.metrics
        and returns them in a list"""
        if len(self.metrics) == 0:
            return 0
        else:
            metrics = []

            for metric in self.metrics:
                metrics += self._as_list(self.get_eval_metric(metric=metric, **kwargs))
            return np.array(metrics)


    # def summary_on_wandb(self, metric_values, wandb, suffix=''):
    #     """Log metrics on wandb as a summary"""
    #     assert len(metric_values) == len(self.metrics)

    #     for name, value in zip(self.metrics, metric_values):
    #             # wandb.log({str(name)+str(suffix): value})
    #             wandb.run.summary[f"{name}{suffix}"] = value


    def log_on_wandb(self, metric_values, wandb, epoch=None, suffix=''):
        """Log metrics on wandb"""
        if len(self.metrics) == 0:
            return

        else:
            assert self.tot_num_of_metrics() == len(metric_values)

            value_index = 0
            for name in self.metrics:
                index = self.metric_index[name]

                for k in range(self.num_of_metrics(name)): 
                    output_name = self.output_metrics_names[index][k]
                    value = metric_values[value_index]
                    if epoch is None:
                        wandb.log({str(output_name)+str(suffix): value})
                    else:
                        wandb.log({str(output_name)+str(suffix): value, "epoch": (epoch+1)})
                    
                    value_index += 1


    def pprint(self, metric_values, prefix=''):
        """Pretty print metric values"""
        if len(self.metrics) == 0:
            return

        else:
            assert self.tot_num_of_metrics() == len(metric_values)

            print(prefix)
            value_index = 0
            for name in self.metrics:
                index = self.metric_index[name]

                for k in range(self.num_of_metrics(name)): 
                    print(f"\t{self.output_metrics_names[index][k]}: {round(metric_values[value_index], 5)}")
                    value_index += 1


    def _as_list(self, item):
        """See item as a list"""
        return [to_numpy(item)] if not isinstance(item, list) else to_numpy(item)


    def tot_num_of_metrics(self):
        count = 0
        for name in self.metrics:
            count += len(self.output_metrics_names[self.metric_index[name]])
        return count


    def num_of_metrics(self, name):
        return len(self.output_metrics_names[self.metric_index[name]])


    def renormalize_traj(self, traj):
        """
            Renormalize trajectory according to a different
            data_scale_factor, as defined in self.renormalize_output_config

            traj : Tensor of size [N,D]
        """
        if not self.renormalize_output:
            return traj
        else:
            assert traj.shape[-1] == 6, 'point-wise format and orientnorm is assumed.'

            fake_mask = torch.all((traj[...] == -100), axis=-1)  # do not touch fake vectors
            traj[..., :3] = torch.where(~fake_mask.unsqueeze(-1), traj[..., :3] * self.renormalize_output_config['from'], traj[..., :3])
            traj[..., :3] = torch.where(~fake_mask.unsqueeze(-1), traj[..., :3] / self.renormalize_output_config['to'], traj[..., :3])

            return traj


    """
    
        EVALUATION METRICS

    """
    def get_pcd(self, y_pred, y, traj_as_pc=None, **kwargs):
        """Pose-wise Chamfer Distance between predictions and ground-truth poses"""
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if self.config['lambda_points'] > 1:
            y_pred = y_pred.reshape(B, -1, outdim)

            # Going from GT traj as segments to points is not ideal, because there is the overlapping parameter
            # and also end-of-stroke points may be thrown out. use the traj_as_pc instead.
            if traj_as_pc is None:
                raise ValueError('DEPRECATED: Going from GT traj as segments to points is not ideal. Use traj_as_pc instead.')
                # y = y.reshape(B, -1, outdim)
                # traj_as_pc = y.clone().detach()
        
        # Pred
        traj_pred_pc = y_pred.clone().detach()
        
        # GT
        if not traj_as_pc.is_cuda:
            traj_as_pc = traj_as_pc.to('cuda', dtype=torch.float)
        if not traj_pred_pc.is_cuda:
            traj_pred_pc = traj_pred_pc.to('cuda', dtype=torch.float)

        # if not traj_pred_pc.is_cuda:
        #     traj_pred_pc = traj_pred_pc.to('cuda')

        with torch.no_grad():
            if self.renormalize_output:
                traj_pred_pc, traj_as_pc = self.renormalize_traj(traj_pred_pc), self.renormalize_traj(traj_as_pc)
                
            chamfer = (10**4)*chamfer_distance(traj_pred_pc, traj_as_pc, padded=True)[0]

        traj_pred_pc = traj_pred_pc.cpu()
        traj_as_pc = traj_as_pc.cpu()

        return chamfer


    def get_chamfer_original(self, y_pred, y, traj_pc, **kwargs):
        """Chamfer between predictions and full, untrimmed ground truth traj_pc.

        trimming may happen because of splitting into lambda-sequences,
        but nevertheless it generally just skips a few poses."""
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if self.config['lambda_points'] > 1:
            y_pred = y_pred.reshape(B, -1, outdim)

        traj_pred_pc = torch.tensor(y_pred)

        print('effective points pred:', traj_pred_pc.shape[1])
        print('effective points GT original:', traj_pc.shape[1])

        chamfer = (10**4)*chamfer_distance(traj_pred_pc, traj_pc)[0]
        return chamfer


    def stroke_masks_metrics(self,
                             n_strokes,
                             pred_stroke_masks,
                             mask_scores,
                             confidence_threshold=0.5,
                             **kwargs):
        """
            Compute metrics on the predicted strokes (clusters of segments) by MaskPlanner

            - Percentage of samples with prediction of correct number of strokes
        """
        processed_stroke_ids_pred = process_pred_stroke_masks_to_stroke_ids(pred_stroke_masks.detach().cpu(), confidence_scores=mask_scores.detach().cpu(), confidence_threshold=confidence_threshold)

        n_strokes_pred = np.array([len(set(np.unique(stroke_ids_pred_b))) for stroke_ids_pred_b in processed_stroke_ids_pred]).astype(int)
        n_strokes = np.array(n_strokes).astype(int)

        perc_correct_n_strokes = np.mean((n_strokes == n_strokes_pred).astype(int))

        avg_num_of_pred_strokes = np.mean(n_strokes_pred)
        avg_num_of_gt_strokes = np.mean(n_strokes)

        mean_absolute_error_NoP = np.mean(np.abs(n_strokes_pred - n_strokes))

        return [perc_correct_n_strokes, avg_num_of_pred_strokes, avg_num_of_gt_strokes, mean_absolute_error_NoP]


    def strokewise_num_of_strokes_metrics(self,
                                          n_strokes,
                                          traj_pred,
                                          **kwargs):
        """
            Compute num-of-strokes metrics for strokeWise models.

            n_strokes: list of int, size B
            traj_pred: list of size B of Tensors [retained_n_strokes, max_n_stroke_points*outdim]
        """
        n_strokes_pred = np.array([traj_pred_b.shape[0] for traj_pred_b in traj_pred]).astype(int)
        n_strokes = np.array(n_strokes).astype(int)

        perc_correct_n_strokes = np.mean((n_strokes == n_strokes_pred).astype(int))

        avg_num_of_pred_strokes = np.mean(n_strokes_pred)
        avg_num_of_gt_strokes = np.mean(n_strokes)

        mean_absolute_error_NoP = np.mean(np.abs(n_strokes_pred - n_strokes))

        return [perc_correct_n_strokes, avg_num_of_pred_strokes, avg_num_of_gt_strokes, mean_absolute_error_NoP]


    def get_sop_metrics(self,
                        sop_pred,
                        processed_sop_pred,
                        sop_gt,
                        pred_sop_conf_scores,
                        sop_conf_threshold,
                        **kwargs):
        """Computes start-of-path (SoP) prediction metrics"""
        unpadded_sop_gt = [remove_padding_from_tensors(sop_gt_b) for sop_gt_b in sop_gt]

        num_of_pred_sops = [len(b_item) for b_item in processed_sop_pred]
        num_of_gt_sops = [len(b_item) for b_item in unpadded_sop_gt]

        avg_num_of_pred_sops = np.mean(num_of_pred_sops)
        avg_num_of_gt_sops = np.mean(num_of_gt_sops)

        avg_ratio_pred_over_gt_sops = np.mean([n_pred_sop/n_gt_sop for (n_pred_sop, n_gt_sop) in zip(num_of_pred_sops, num_of_gt_sops)])
        
        # With higher threshold
        higher_threshold = (sop_conf_threshold + 1) / 2
        processed_sop_pred_higher_t = postprocess_sop_predictions(sop_pred=sop_pred, pred_sop_conf_scores=pred_sop_conf_scores, sop_conf_threshold=higher_threshold)  # list of size B of Tensors [retained_n_sop, config.stroke_prototype_dim]
        num_of_pred_sops = [len(b_item) for b_item in processed_sop_pred_higher_t]
        avg_num_of_pred_sops_if_higher = np.mean(num_of_pred_sops)
        avg_ratio_pred_over_gt_sops_if_higher = np.mean([n_pred_sop/n_gt_sop for (n_pred_sop, n_gt_sop) in zip(num_of_pred_sops, num_of_gt_sops)])

        # With lower threshold
        lower_threshold = (sop_conf_threshold) / 2
        processed_sop_pred_lower_t = postprocess_sop_predictions(sop_pred=sop_pred, pred_sop_conf_scores=pred_sop_conf_scores, sop_conf_threshold=lower_threshold)  # list of size B of Tensors [retained_n_sop, config.stroke_prototype_dim]
        num_of_pred_sops = [len(b_item) for b_item in processed_sop_pred_lower_t]
        avg_num_of_pred_sops_if_lower = np.mean(num_of_pred_sops)
        avg_ratio_pred_over_gt_sops_if_lower = np.mean([n_pred_sop/n_gt_sop for (n_pred_sop, n_gt_sop) in zip(num_of_pred_sops, num_of_gt_sops)])

        sop_metrics = [avg_num_of_pred_sops,
                       avg_num_of_gt_sops,
                       avg_ratio_pred_over_gt_sops,
                       avg_num_of_pred_sops_if_higher,
                       avg_num_of_pred_sops_if_lower,
                       avg_ratio_pred_over_gt_sops_if_higher,
                       avg_ratio_pred_over_gt_sops_if_lower]
        
        return sop_metrics


    def get_sop_metrics_v2(self,
                           sop_pred,
                           processed_sop_pred,
                           sop_gt,
                           pred_sop_conf_scores,
                           sop_conf_threshold,
                           **kwargs):
        """Computes start-of-path (SoP) prediction metrics

            v2:
                - Avg ratio is deprecated (avg num of pred already carries enough information),
                  as it may be misleading (goods and bads can cancel out)
                - Accuracy of num of strokes
                - Mean absolute error
        """
        unpadded_sop_gt = [remove_padding_from_tensors(sop_gt_b) for sop_gt_b in sop_gt]

        num_of_pred_sops = np.array([len(b_item) for b_item in processed_sop_pred]).astype(int)
        num_of_gt_sops = np.array([len(b_item) for b_item in unpadded_sop_gt]).astype(int)

        avg_num_of_pred_sops = np.mean(num_of_pred_sops)
        avg_num_of_gt_sops = np.mean(num_of_gt_sops)

        perc_correct_n_strokes = np.mean((num_of_gt_sops == num_of_pred_sops).astype(int))
        
        mean_absolute_error_NoP = np.mean(np.abs(num_of_pred_sops - num_of_gt_sops))

        # deprecated
        # avg_ratio_pred_over_gt_sops = np.mean([n_pred_sop/n_gt_sop for (n_pred_sop, n_gt_sop) in zip(num_of_pred_sops, num_of_gt_sops)])
        
        # With higher threshold
        higher_threshold = (sop_conf_threshold + 1) / 2
        processed_sop_pred_higher_t = postprocess_sop_predictions(sop_pred=sop_pred, pred_sop_conf_scores=pred_sop_conf_scores, sop_conf_threshold=higher_threshold)  # list of size B of Tensors [retained_n_sop, config.stroke_prototype_dim]
        num_of_pred_sops = np.array([len(b_item) for b_item in processed_sop_pred_higher_t]).astype(int)
        avg_num_of_pred_sops_if_higher = np.mean(num_of_pred_sops)
        mean_absolute_error_NoP_if_higher = np.mean(np.abs(num_of_pred_sops - num_of_gt_sops))

        # With lower threshold
        lower_threshold = (sop_conf_threshold) / 2
        processed_sop_pred_lower_t = postprocess_sop_predictions(sop_pred=sop_pred, pred_sop_conf_scores=pred_sop_conf_scores, sop_conf_threshold=lower_threshold)  # list of size B of Tensors [retained_n_sop, config.stroke_prototype_dim]
        num_of_pred_sops = np.array([len(b_item) for b_item in processed_sop_pred_lower_t]).astype(int)
        avg_num_of_pred_sops_if_lower = np.mean(num_of_pred_sops)
        mean_absolute_error_NoP_if_lower = np.mean(np.abs(num_of_pred_sops - num_of_gt_sops))


        sop_metrics_v2 = [
                          perc_correct_n_strokes,
                          avg_num_of_pred_sops,
                          avg_num_of_gt_sops,
                          mean_absolute_error_NoP,
                          avg_num_of_pred_sops_if_higher,
                          avg_num_of_pred_sops_if_lower,
                          mean_absolute_error_NoP_if_higher,
                          mean_absolute_error_NoP_if_lower
                         ]

        return sop_metrics_v2


    def get_clustering_metrics(self, stroke_ids_pred, stroke_ids, clusterer, **kwargs):
        """Computes clustering and its evaluation metrics"""
        B, N = stroke_ids.shape

        clustering_metrics = clusterer.eval(labels_true=stroke_ids, labels_pred=stroke_ids_pred)

        return clustering_metrics


    def get_stroke_chamfer(self, y_pred, y, traj_pc, stroke_ids, **kwargs):
        """Debug: chamfer between predicted vectors and original strokes,
        with inner distance metric as an additional chamfer distance."""
        asymmetric = True
        print(f'---\nCAREFUL! Stroke-wise chamfer is with ASYMMETRIC={asymmetric}\n---')

        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        traj_pred = torch.tensor(y_pred)
        
        ##### 1° version
        chamfers = [0 for b in range(B)]
        for b in range(B):
            chamfer = 0

            n_pred_strokes = y_pred.shape[1]
            n_gt_strokes = stroke_ids[b, -1]+1
            unique, counts = np.unique(stroke_ids[b], return_counts=True)
            assert len(unique) == n_gt_strokes
            for i in range(n_pred_strokes):
                min_chamfer = 10000000
                
                pred_pc = traj_pred[b, i].view(-1, outdim)[None, :, :]
                for i_gt in range(n_gt_strokes):
                    curr_gt_pc = traj_pc[b, stroke_ids[b, :] == i_gt, :][None, :, :]
                    curr_chamfer = (10**4)*chamfer_distance(pred_pc, curr_gt_pc, asymmetric=asymmetric)[0]
                    # dist1, dist2, _, _ = NND.nnd(pred_pc, curr_gt_pc)  # Chamfer loss
                    # chamfer = (10**4)*(torch.mean(dist1))

                    min_chamfer = min(min_chamfer, curr_chamfer.item())

                chamfer += min_chamfer

            chamfers[b] = chamfer/n_pred_strokes

        chamfers = np.array(chamfers).mean()
        ##############################

        ##### 2° version (would require stroke-padding, so it currently does not work)
        # batch_stroke_chamfer = torch.empty((B, 0))

        # n_pred_strokes = y_pred.shape[1]
        # min_chamfer = torch.ones((B,))*10000000
        # for i in range(n_pred_strokes):

        #     pred_pc = traj_pred[:, i, :].view(B, -1, outdim)
        #     for i_gt in range(n_gt_strokes):
        #         curr_gt_pc = traj_pc[b, stroke_ids[b, :] == i_gt, :][None, :, :]
        #         chamfer = (10**4)*chamfer_distance(pred_pc, curr_gt_pc, asymmetric=True)[0]
        ##############################
        return chamfers

