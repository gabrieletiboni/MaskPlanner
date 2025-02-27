"""Train a MaskPlanner model.
The script can also be used for reproducing MaskPlanner w/out AP2S, and point-wise baselines.


        DEBUGGING:
            python train_maskplanner.py config=[maskplanner,cuboids_v2,longx_v2,debug] seed=42


        OFFICIAL:
            python train_maskplanner.py config=[maskplanner,cuboids_v2,longx_v2] seed=42

            BASELINES:
                python train_maskplanner.py config=[segmentWise,cuboids_v2,longx_v2] seed=42
                python train_maskplanner.py config=[pointWise,cuboids_v2,longx_v2] seed=42

        
        EXAMPLES OF EXPLICIT PARAMETERS & CONFIG FILES (explicit parameters overwrite those in the config files):
            python train_maskplanner.py config=[asymm_chamfer_v9,delayMasksLoss,traj_sampling_v2,sched_v9,cuboids_v2,longx_v2] \
                                         wandb=offline  \
                                         epochs=2 \
                                         batch_size=64 \
                                         seed=42

        OBJECT CATEGORIES
            config=[...,<dataset>]
                <dataset> = {cuboids_v2, windows_v2, shelves_v2, containers_v2}


        JOINT-TRAINING
                python train_maskplanner.py config=[maskplanner,jointTraining_v2,longx_v2] seed=42
                NOTE:
                    joint-training's different dataset downscaling factor affects the final number of maximum points in the trajectory when using
                    equal-spaced points. This could require changing the distance for the equal-spaced points or adjusting the number of predicted points differently.


        DATA-AUGMENTATIONS:
            - point-cloud online subsampling (takes more training time but generally improves generalization):
                python train_maskplanner.py config=[maskplanner,cuboids_v2,longx_v2,augm_v1] seed=42

"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gc
import os
import pdb
import sys
import argparse
from pprint import pprint
import time
import socket
import shutil

import numpy as np
import torch
import wandb
from tqdm import tqdm

from utils import get_random_string, set_seed, create_dirs
from utils.config import save_config
from utils.disk import get_dataset_paths, get_output_dir
from utils.dataset.paintnet_ODv1 import PaintNetODv1Dataloader, Paintnet_ODv1_CollateBatch
from utils.training import get_lr_scheduler
from utils.visualize import *
from utils.args import load_args, pformat_dict, to_dict
from models import get_model
from loss_handler import LossHandler
from metrics_handler import MetricsHandler


config = load_args(root='configs/maskplanner')
config.task_name = 'MaskPlanner'

def main():
    random_str = get_random_string(5)
    set_seed(config.seed)

    run_name = random_str+('_'+config.name if config.name is not None else '')+'-S'+str(config.seed)
    output_dir = get_output_dir(config)
    save_dir = os.path.join((output_dir if not config.debug else 'debug_runs'), run_name)
    create_dirs(save_dir)
    save_config(config, save_dir)

    print('\n ===== RUN NAME:', run_name, f' ({save_dir}) ===== \n')
    print(pformat_dict(config, indent=0))

    wandb_group_name = str(config.group) if config.group is not None else config.auto_wandb_group+str(config.group_suffix)
    wandb.init(config=to_dict(config),
               project="MaskPlanner",
               name=run_name,
               group=config.task_name+'V1_'+wandb_group_name,
               save_code=True,
               notes=config.notes,
               mode=config.wandb)
    
    wandb.config.path = save_dir
    wandb.config.hostname = socket.gethostname()

    # Make sure at least one evaluation is done
    config.eval_freq = min(config.eval_freq, config.epochs)

    """
        Load dataset
    """
    dataset_paths = get_dataset_paths(config.dataset)
    
    tr_dataset = PaintNetODv1Dataloader(roots=dataset_paths,
                                        dataset=config.dataset,
                                        pc_points=config.pc_points,
                                        traj_points=config.traj_points,
                                        lambda_points=config.lambda_points,
                                        overlapping=config.overlapping if not config.asymm_overlapping else config.lambda_points-1,
                                        normalization=config.normalization,
                                        data_scale_factor=config.data_scale_factor,
                                        extra_data=tuple(config.extra_data),
                                        weight_orient=config.weight_orient,
                                        split='train',
                                        config=config,
                                        overfitting=(None if config.overfitting is False else config.seed),
                                        augmentations=config.augmentations,
                                        train_portion=config.train_portion)

    te_dataset = PaintNetODv1Dataloader(roots=dataset_paths,
                                        dataset=config.dataset,
                                        pc_points=config.pc_points,
                                        traj_points=config.traj_points,
                                        lambda_points=config.lambda_points,
                                        overlapping=config.overlapping if not config.asymm_overlapping else config.lambda_points-1,
                                        normalization=config.normalization,
                                        data_scale_factor=config.data_scale_factor,
                                        extra_data=tuple(config.extra_data),
                                        weight_orient=config.weight_orient,
                                        split='test',
                                        config=config)

    collate_fn = Paintnet_ODv1_CollateBatch(config)
    tr_loader = torch.utils.data.DataLoader(tr_dataset,
                                            batch_size=min(config.batch_size, len(tr_dataset)),   # train_portion may lead to training samples < batch_size
                                            shuffle=(True if config.overfitting is False else False),
                                            num_workers=config.workers,
                                            drop_last=True,
                                            collate_fn=collate_fn,
                                            worker_init_fn=lambda x: set_seed(config.seed + x))

    te_loader = torch.utils.data.DataLoader(te_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=config.workers,
                                            collate_fn=collate_fn,
                                            worker_init_fn=lambda x: set_seed(config.seed + x))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load training model
    model = get_model(config=config,
                      which=config.model.backbone,
                      io_type=config.task_name,
                      device=device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = get_lr_scheduler(opt, lr_sched=config.lr_sched, epochs=config.epochs, steplr=config.steplr)
    assert (sched is None) or (sched is not None and not config.legacy), 'Disable scheduler if legacy is True. Legacy version is without lr scheduler.'

    loss_handler = LossHandler(config.loss, config=config)
    metrics_handler = MetricsHandler(config=config, metrics=config.eval_metrics)

    # Handle Point-to-Segment Asymmetric Chamfer Distance (PSA-CD)
    psacd_scheduler = PSACDScheduler(config.psacd_scheduler) if config.psacd_scheduler.active else None

    n_batches_per_epoch = len(tr_dataset) // min(config.batch_size, len(tr_dataset))
    single_sample = None
    best_epoch = None
    best_eval_loss = sys.float_info.max
    tot_train_time = 0
    for epoch in tqdm(range(config.epochs), desc="Epoch"):
        start_ep_time = time.time()
        tot_loss = 0.0
        tot_loss_list = np.zeros(len(loss_handler.loss))  
        data_count = 0
        epoch_count = 0
        last_epoch = (epoch+1) == config.epochs
        model.train()
        for i, data in tqdm(enumerate(tr_loader), desc=f"Batch (out of {n_batches_per_epoch})"):
            model.zero_grad()

            point_cloud = data['point_cloud']  # ndim=3 (B, pc_points, 3)
            traj = data['traj']                # ndim=3 (B, (traj_points-lambda)//(lambda-overlapping)+1, outdim*lambda), padded over dim=1
            traj_as_pc = data['traj_as_pc']    # (B, traj_points )
            stroke_ids = data['stroke_ids']    # ndim=2 (B, (traj_points-lambda)//(lambda-overlapping)+1 ), padded over dim=1
            stroke_ids_as_pc = data['stroke_ids_as_pc']  # ndim=2 (B, traj_points)
            stroke_masks = data['stroke_masks']  # list of size B of Tensors of dim [n_strokes, (traj_points-lambda)//(lambda-overlapping)+1)]
            dirname = data['dirname']          # list of str, size B
            n_strokes = data['n_strokes']      # list of int, size B

            if config.overfitting and single_sample is None:
                single_sample = dirname

            B, N, dim = point_cloud.size()
            data_count += B

            # for b in range(B):
            #     plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
            #     # visualize_pc(point_cloud[b], plotter=plotter, index=(0,0))
            #     visualize_mesh_traj(dirname[b], traj[b], config=config, plotter=plotter, index=(0,0))
            #     plotter.show()
            # pdb.set_trace()

            point_cloud = point_cloud.permute(0, 2, 1) # B, 3, pc_points
            point_cloud, traj = point_cloud.to(device, dtype=torch.float), traj.to(device, dtype=torch.float)

            traj_pred, pred_stroke_masks, mask_scores, seg_logits = model(point_cloud)

            loss, loss_list = loss_handler.compute(y_pred=traj_pred,
                                                   y=traj,
                                                   pred_stroke_masks=pred_stroke_masks,
                                                   mask_scores=mask_scores,
                                                   seg_logits=seg_logits,
                                                   stroke_ids=stroke_ids,
                                                   traj_as_pc=traj_as_pc)

            loss.backward()
            opt.step()

            tot_loss += loss.item() * B
            tot_loss_list += loss_list * B

            del point_cloud, traj, traj_pred, mask_scores, seg_logits, pred_stroke_masks  # free-up gpu memory when evaluating
            model.zero_grad()  # free-up gpu memory when evaluating

        if not config.legacy:
            sched.step()
            # print('Last LR:', sched.get_last_lr())
            
        wandb.log({"TOT_epoch_train_loss": (tot_loss * 1.0 / data_count), "epoch": (epoch+1)})
        tot_loss_list = tot_loss_list * 1.0 / data_count
        loss_handler.log_on_wandb(tot_loss_list, wandb, epoch, suffix='_train_loss')
        print('[%d/%d] Epoch time: %s' % (
            epoch+1, config.epochs, time.strftime("%M:%S", time.gmtime(time.time() - start_ep_time))), '| Epoch train loss: %.5f' % (tot_loss * 1.0 / data_count), '| Epoch train loss list: ', tot_loss_list)

        tot_train_time += time.time() - start_ep_time

        # Evaluate current model every `eval_freq` epochs
        if (epoch+1) % config.eval_freq == 0:
            torch.save(
                {'epoch': epoch + 1,
                 'epoch_train_loss': tot_loss * 1.0 / data_count,
                 'model': model.state_dict(),
                 'optimizer': opt.state_dict(),
                 'scheduler': sched.state_dict() if sched is not None else None,
                },
                os.path.join(save_dir, 'last_checkpoint.pth')
            )

            if not config.overfitting:
                eval_loss, eval_loss_list, eval_metrics = test(model, te_loader, loss_handler=loss_handler, metrics_handler=metrics_handler)
                print('Tot test loss: %.5f | test PCD: %.5f' % (eval_loss, eval_metrics[0]))

                wandb.log({"TOT_test_loss": eval_loss, "epoch": (epoch+1)})
                loss_handler.log_on_wandb(eval_loss_list, wandb, epoch, suffix='_test_loss')

                wandb.log({"test_PCD_metric": eval_metrics[0], "epoch": (epoch+1)})  # Plot PCD with this name for retro-compatibility
                metrics_handler.pprint(eval_metrics, prefix='Test metrics:')
                metrics_handler.log_on_wandb(eval_metrics, wandb, epoch, suffix='_test_metric')

                is_best = eval_loss < best_eval_loss
                best_eval_loss = min(eval_loss, best_eval_loss)
                if is_best:
                    best_epoch = epoch+1
                    shutil.copyfile(
                        src=os.path.join(save_dir, 'last_checkpoint.pth'),
                        dst=os.path.join(save_dir, 'best_model.pth'))


        # Save intermediate models every `save_intermediate_models_freq` epochs
        if config.save_intermediate_models and (epoch+1) % config.save_intermediate_models_freq == 0 and not last_epoch:
                # Save models right before the change of loss weights
                torch.save(
                    {'epoch': epoch + 1,
                     'epoch_train_loss': tot_loss * 1.0 / data_count,
                     'model': model.state_dict(),
                     'optimizer': opt.state_dict(),
                     'scheduler': sched.state_dict() if sched is not None else None,
                    },
                    os.path.join(save_dir, f'intermediate_checkpoint_epoch{epoch+1}.pth')
                )


        # Dynamic re-weighting of the train loss point-to-segment asymmetric chamfer distance (PSACD) for yopo.
        if config.psacd_scheduler.active:
            if psacd_scheduler.is_time_to_step(epoch, config):
                psacd_scheduler.step_loss_weights(config, loss_handler)


        # Activate stroke masks loss after a given number of epochs
        if config.delay_stroke_masks_loss and config.start_stroke_masks_loss_at <= (epoch+1):
            config.explicit_weight_stroke_masks = config.target_explicit_weight_stroke_masks
            config.explicit_weight_stroke_masks_confidence = config.target_explicit_weight_stroke_masks_confidence

            loss_handler.config = config


        # Activate per-segment confidence loss after a given number of epochs
        if config.delay_segment_conf_loss and config.start_segment_conf_loss_at <= (epoch+1):
            config.explicit_weight_segments_confidence = config.target_explicit_weight_segments_confidence

            loss_handler.config = config



    print('\n\n============== TRAINING FINISHED ==============')
    if config.overfitting:
        wandb.run.summary["single_sample"] = single_sample
        print('Overfitting on:', single_sample)
    else:
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_eval_loss"] = best_eval_loss
        print('Best epoch:', best_epoch)
        print('Best test loss:', best_eval_loss)
        print('Last test loss:', eval_loss)

    print('Tot training time:', time.strftime("%H:%M:%S", time.gmtime(tot_train_time)))
    wandb.run.summary["tot_train_seconds"] = round(tot_train_time, 2)

    # Clean GPU memory
    del model, opt, sched
    torch.cuda.empty_cache()

    """
        Test best model and render results
    """
    eval_ckpt = config.eval_ckpt if not config.overfitting else 'last'
    if eval_ckpt == 'best':
        eval_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=torch.device(device))
    elif eval_ckpt == 'last':
        eval_checkpoint = torch.load(os.path.join(save_dir, 'last_checkpoint.pth'), map_location=torch.device(device))
    else:  # default
        print('\n\nWARNING! Falling back to best_model.pth as eval_ckpt has invalid name.\n\n')
        eval_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=torch.device(device))

    model = get_model(config=config,
                      which=config.model.backbone,
                      io_type=config.task_name,
                      device=device)
    model.load_state_dict(eval_checkpoint['model'], strict=True)
    model.to(device)
    model.eval()

    metrics_handler = MetricsHandler(config=config, metrics=config.eval_metrics)
    save_args = {'save_dir': save_dir, 'eval_ckpt': eval_ckpt}

    # Eval on train
    eval_loss, eval_loss_list, train_eval_metrics = test(model, tr_loader, loss_handler=loss_handler, metrics_handler=metrics_handler, save=(not config.no_save), **{'split': 'train', **save_args})
    metrics_handler.pprint(train_eval_metrics, prefix="Train metrics:")
    metrics_handler.log_on_wandb(train_eval_metrics, wandb, suffix='_TRAIN_EVAL_METRIC')

    # Eval on test
    if not config.overfitting:
        eval_loss, eval_loss_list, test_eval_metrics = test(model, te_loader, loss_handler=loss_handler, metrics_handler=metrics_handler, save=(not config.no_save), **{'split': 'test',  **save_args})
        metrics_handler.pprint(test_eval_metrics, prefix="Test metrics:")
        metrics_handler.log_on_wandb(test_eval_metrics, wandb, suffix='_TEST_EVAL_METRIC')

    print('Results saved successfully in:', save_dir)
    wandb.finish()


    del model
    # Free space on disk if it's just for debugging
    if config.no_save or config.debug:
        try:
            os.unlink(os.path.join(save_dir, 'last_checkpoint.pth'))
            os.unlink(os.path.join(save_dir, 'best_model.pth'))
        except OSError as e:
            print("Error while deleting: %s - %s." % (e.filename, e.strerror))

    """
        Rendering
        Note: this won't work when launching the script on ssh protocol.
    """
    if not config.skip_rendering and not config.debug and not config.no_save:
        print('\n\n============== Rendering ==============')
        os.system(f"python render_results.py --run {save_dir} --save_n 16 --with_postprocess")




@torch.no_grad()
def test(model, loader, loss_handler, metrics_handler=None, save=False, **save_args):
    """Test model on dataloader"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tot_loss = 0
    tot_loss_list = np.zeros(len(loss_handler.loss))
    data_count = 0
    tot_metric_list = np.zeros(metrics_handler.tot_num_of_metrics())

    for i, data in enumerate(loader):
        point_cloud = data['point_cloud']  # ndim=3 (B, pc_points, 3)
        traj = data['traj']                # ndim=3 (B, (traj_points-lambda)//(lambda-overlapping)+1, outdim*lambda), padded over dim=1
        traj_as_pc = data['traj_as_pc']    # (B, traj_points )
        stroke_ids = data['stroke_ids']    # ndim=2 (B, (traj_points-lambda)//(lambda-overlapping)+1 ), padded over dim=1
        stroke_ids_as_pc = data['stroke_ids_as_pc']  # ndim=2 (B, traj_points)
        dirname = data['dirname']          # list of str, size B
        n_strokes = data['n_strokes']      # list of int, size B

        B, N, dim = point_cloud.size()
        data_count += B
        point_cloud = point_cloud.permute(0, 2, 1) # B, 3, pc_points
        point_cloud, traj = point_cloud.to(device, dtype=torch.float), traj.to(device, dtype=torch.float)

        traj_pred, pred_stroke_masks, mask_scores, seg_logits = model(point_cloud)

        # Compute loss at test time
        loss, loss_list = loss_handler.compute(y_pred=traj_pred,
                                               y=traj,
                                               pred_stroke_masks=pred_stroke_masks,
                                               mask_scores=mask_scores,
                                               seg_logits=seg_logits,
                                               stroke_ids=stroke_ids,
                                               traj_as_pc=traj_as_pc)

        tot_loss += loss.item() * B
        tot_loss_list += loss_list * B

        # Compute evaluation metrics
        tot_metric_list += B * metrics_handler.compute(y_pred=traj_pred,
                                                       y=traj,
                                                       traj_as_pc=traj_as_pc,

                                                       n_strokes=n_strokes,
                                                       pred_stroke_masks=pred_stroke_masks,
                                                       mask_scores=mask_scores
                                                       )

        if save and (save_args['split'] != 'train' or i == 0):  # Save first training batch for training set, and all batches for test set
            data = {'dirnames': dirname,
                    'traj': traj.detach().cpu().numpy(),
                    'stroke_ids': stroke_ids.detach().cpu().numpy(),
                    'stroke_ids_as_pc': stroke_ids_as_pc.detach().cpu().numpy(),
                    'traj_as_pc': traj_as_pc.detach().cpu().numpy(),
                    'traj_pred': traj_pred.detach().cpu().numpy(),
                    'pred_stroke_masks': pred_stroke_masks.detach().cpu().numpy(),
                    'stroke_masks_scores': mask_scores.detach().cpu().numpy(),
                    'seg_logits': seg_logits.detach().cpu().numpy() if seg_logits is not None else None,
                    'batch': i,
                    'suffix': str(save_args['split'])}
            np.save(os.path.join(save_args['save_dir'], str(save_args['eval_ckpt'])+'_'+str(save_args['split'])+'_batch'+str(i)+'.npy'), data)

        del point_cloud, traj, traj_pred, traj_as_pc, pred_stroke_masks, seg_logits, mask_scores
    
    
    return (tot_loss * 1.0 / data_count,  # total loss
            tot_loss_list * 1.0 / data_count,   # list of each loss component
            tot_metric_list * 1.0 / data_count)   # list of evaluation metrics


class PSACDScheduler:
    """
        Point-to-Segment Asymmetric Chamfer Distance (PSA-CD) Scheduler.
        Change loss weights (point-wise vs. segment-wise) during training.
    """
    def __init__(self, psacd_scheduler):
        self.milestones = psacd_scheduler.milestones
        self.step_freq = psacd_scheduler.freq
        self.factor = psacd_scheduler.factor

        assert not (self.milestones is not None and self.step_freq is not None), 'Define either step_freq or milestones, not both.'
        assert (self.milestones is not None or self.step_freq is not None), 'Define at least one among milestones and step_freq'
        assert self.factor is not None and self.factor > 0

        print('Factor is:', self.factor)

        if self.milestones is not None:
            self.milestones = [int(milestone) for milestone in self.milestones]

    def is_time_to_step(self, epoch, config):
        # Don't step at last epoch (so that final test results compute loss function correctly)
        if (epoch+1) == config.epochs:
            return False

        if self.step_freq is not None:
            # Epochs mode.
            # Step weights every `step_freq` epochs
            return (epoch+1) % self.step_freq == 0

        elif self.milestones:
            # Milestones mode.
            # Step weights when reaching the indicated milestone epochs.
            return epoch+1 in self.milestones


    def step_loss_weights(self, config, loss_handler):
        """
            Change loss weights of the Point-to-Segment asymmetric chamfer distance.
        """
        config.weight_reverse_asymm_point_chamfer *= self.factor  # decrease point-wise importance
        config.weight_reverse_asymm_segment_chamfer /= self.factor  # increase segment-wise importance

        config.weight_symm_point_chamfer *= self.factor  # decrease point-wise importance
        config.weight_symm_segment_chamfer /= self.factor  # decrease point-wise importance
        config.weight_rich_attraction_chamfer /= self.factor  # increase segment-wise importance

        loss_handler.config = config  # these new weights must be propagated to the loss_handler that uses them

        return

if __name__ == '__main__':
    main()