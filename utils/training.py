from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pdb

import numpy as np
import torch


def sample_autoregressive_inference_sop(model,
                                        sops,
                                        history_length,
                                        output_length,
                                        max_rollout_steps,
                                        config,
                                        object_features=None,
                                        device='cuda'):
    """
        Per-sample autoregressive inference of strokes conditioned
        on (SoPs, object features, fixed history of predictions).

        Strokes for this sample are rolled out in parallel for `max_rollout_steps` steps.
        The returned end-of-path tokens shall be used to truncate each path accordingly.

        sops: Tensor of size [n_strokes, prototype_dim]
              Start-of-Path tokens 
        history_length : number of input passed predictions
        output_length : dimensionality of each prediction
        max_rollout_steps : num of predicted vectors autoregressively
        object_features : Tensor of size [latent_dim]
                          latent features of input point-cloud

    """
    if object_features is not None:
        assert object_features.ndim == 1

    H = history_length
    D = output_length
    n_strokes = len(sops)

    histories = torch.zeros((n_strokes, H, D))  # initial histories, simply zero.

    if config.rollout_model.object_features:
        # Duplicate object features
        stacked_object_features = object_features[None, :][[0]*n_strokes, :]  # (n_strokes, 1024)

    sops, histories = sops.to(device, dtype=torch.float), histories.to(device, dtype=torch.float)

    paths = torch.zeros((n_strokes, max_rollout_steps, D))  # pred paths
    eop_logits = torch.zeros((n_strokes, max_rollout_steps, 1))  # pred end-of-path

    for n in range(max_rollout_steps):
        flat_histories = histories.view(histories.shape[0], -1)
        cat_input = torch.cat((sops, flat_histories), dim=1)

        if config.rollout_model.object_features:
            cat_input = torch.cat((cat_input, stacked_object_features), dim=1)

        curr_pred_nexttoken, curr_eop_logits = model(cat_input)  # .squeeze()

        # Save prediction
        paths[:, n:n+1, :] = curr_pred_nexttoken.detach().cpu()
        eop_logits[:, n:n+1, :] = curr_eop_logits.detach().cpu()

        # Update history vector
        histories[:, :H-1, :] = histories[:, 1:, :]
        histories[:, -1:, :] = curr_pred_nexttoken

    return paths, eop_logits


def get_lr_scheduler(opt, lr_sched: Dict, epochs: int, steplr: int):
    """Return LR scheduler"""
    step_size, step_sizes, step_n_times, step_after_epoch, gamma = lr_sched['step_size'], lr_sched['step_sizes'], lr_sched['step_n_times'], lr_sched['step_after_epoch'], lr_sched['gamma']
    
    # Sanity check
    config_values = [step_size, step_sizes, step_n_times]
    is_none_values = [item is not None for item in config_values]
    assert np.array(is_none_values, dtype=int).sum() <= 1, 'Sanity check: more than 1 value has a not-None value. Which of these did you mean to use?'

    # Sanity check
    assert steplr is None, 'the CLI parameter `steplr` is deprecated. Use `lr_sched.step_size` instead, or other `lr_sched.*` parameters.'

    sched = None

    if step_size is not None:
        # step LR scheduler by gamma every `step_size` epochs
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
        print('StepLR with step_size:', step_size, '| gamma:', gamma)

    elif step_sizes is not None:
        # List of int. Step LR scheduler once the number of epoch reaches one of the milestones in the list.
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=step_sizes, gamma=gamma)
        print('MultistepLR with milestones:', step_sizes, '| gamma:', gamma)

    elif step_n_times is not None:
        # step LR scheduler `step_n_times` during training, i.e. every epochs // (step_n_times+1). Start from `step_after_epoch`, if set.
        milestones = []
        tot_epochs = epochs if step_after_epoch is None else epochs - step_after_epoch
        base = 0 if step_after_epoch is None else step_after_epoch
        for i in range(step_n_times):
            milestone = (i+1)*(tot_epochs // (step_n_times+1)) + base
            milestones.append(milestone)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
        print('MultistepLR (step_n_times) with milestones:', milestones, '| gamma:', gamma)
    
    else:
        print('LR Scheduler is not used.')

    return sched