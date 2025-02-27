"""Test a model on PaintNet (trained with train_maskplanner.py)

    Examples:
        python test_maskplanner.py --run $WORKDIR/XXXXXX --model last --batch_size=8 --split test


        SAVE PREDICTIONS:
            [...] --save

        
        JOINT-TRAINING:
            python test_maskplanner.py --run $WORKDIR/XXXXXX --model last --split test --target=cuboids-v2 --batch_size=8 --data_scale_factor=779.2320060197117 --force_fresh_preprocess
            
            OR

            manually remove the `if self.multi_root` statement in the dataset loader `_get_preprocessed_sample_name` method, AND don't use --force_fresh_preprocess :
                python test_maskplanner.py --run $WORKDIR/XXXXXX --model last --split test --target=cuboids-v2 --batch_size=8 --data_scale_factor=779.2320060197117

        
"""
import argparse
from pprint import pprint
import pdb
import time
import os

import torch
import numpy as np
from tqdm import tqdm

from models import get_model
from loss_handler import LossHandler
from metrics_handler import MetricsHandler
from utils import set_seed, create_dirs
from utils.config import load_config, save_config
from utils.disk import get_dataset_paths, get_test_results_save_dir_name, get_dataset_downscale_factor, get_dataset_name
from utils.dataset.paintnet_ODv1 import PaintNetODv1Dataloader, Paintnet_ODv1_CollateBatch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',     default=None, type=str, help='Run directory', required=True)
    parser.add_argument('--model',   default='last', type=str, help='Which model to test: last, best, <filename>')
    parser.add_argument('--save',    default=False, action='store_true', help='Whether to save the predictions .npy')
    parser.add_argument('--split',   default=None, type=str, help='Split to test on. Both test and train by default.')
    parser.add_argument('--target',  default=None, type=str, help='Target dataset to test model on. By default, the category used when first training the model is used.')
    parser.add_argument('--batch_size', default=None, type=int, help='Batch size that overwrites the original run batchsize (also renders these fewer samples for the batch image then!) Used to take less GPU memory while evaluating.')
    parser.add_argument('--data_scale_factor', default=None, type=float, help='Overwrite the original data_scale_factor or the one precomputed')
    parser.add_argument('--force_fresh_preprocess', default=False, action='store_true', help='Avoid using preprocessed data on disk. useful when using a custom --data_scale_factor')

    # Note: the PCD is still not comparable even when using --renormalize_data_to_default, because the number of output and GT points at training time varies according to the training data_scale_factor. I should refactor the code to use a traj_sampling_v2 in the original space, and then normalize. 
    parser.add_argument('--renormalize_data_to_default', default=False, action='store_true', help='If set, compute PCD according to the default normalization per-dataset of this category, instead of using the custom data_scale_factor provided and used at training time.')

    return parser.parse_args()


cli_args = parse_args()
pprint(vars(cli_args))

assert os.path.isdir(cli_args.run), f'dir {cli_args.run} does not exist'
run_dir = cli_args.run
config = load_config(os.path.join(run_dir, 'config.yaml'))  # global config available to all methods

def main():
    set_seed(1)  # inference is stochastic in general, this makes it such that metrics are reproducible with this script. 

    """
        Dataset loading
    """
    if cli_args.target is not None:
        config.dataset = [cli_args.target]
    if cli_args.batch_size is not None:
        config.batch_size = cli_args.batch_size
    if cli_args.data_scale_factor is not None:
        config.data_scale_factor = cli_args.data_scale_factor
    renormalize_output_config = {}
    if cli_args.renormalize_data_to_default:
        assert config.data_scale_factor is not None, 'renormalize_data_to_default only makes sense when a different data_scale_factor than the default one has been used at training time.'
        renormalize_output_config = {
            'active': True,
            'from': config.data_scale_factor,
            'to': get_dataset_downscale_factor(get_dataset_name(config.dataset))
        }


    if 'n_pred_traj_points' not in config:  # Retro-Compatibility
        config['n_pred_traj_points'] = None
    if 'traj_with_equally_spaced_points' not in config:  # Retro-Compatibility
        config['traj_with_equally_spaced_points'] = None
    if 'per_segment_confidence' not in config:  # Retro-Compatibility
        config['per_segment_confidence'] = False
    if 'smooth_target_stroke_masks' not in config:  # Retro-Compatibility
        config['smooth_target_stroke_masks'] = False
    if 'load_extra_data' not in config:  # Retro-Compatibility
        config['load_extra_data'] = ['stroke_masks']
    if 'out_prototypes' not in config:  # Retro-Compatibility
        config['out_prototypes'] = None
    if 'equal_in_3d_space' not in config:  # Retro-Compatibility
        config['equal_in_3d_space'] = False
    if 'stroke_masks_metrics' not in config['eval_metrics']:  # Additionally compute stroke_masks_metrics at test time
        config['eval_metrics'].append('stroke_masks_metrics')

    if cli_args.target is not None:
        assert cli_args.data_scale_factor is not None, '--data_scale_factor is not strictly needed, but its highly recommended when using --target. Bypass this assert if you really know what youre doing.'

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
                                        train_portion=config.train_portion,
                                        force_fresh_preprocess=cli_args.force_fresh_preprocess)

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
                                        config=config,
                                        force_fresh_preprocess=cli_args.force_fresh_preprocess)

    collate_fn = Paintnet_ODv1_CollateBatch(config)
    tr_loader = torch.utils.data.DataLoader(tr_dataset,
                                            batch_size=min(config.batch_size, len(tr_dataset)),   # train_portion may lead to training samples < batch_size
                                            shuffle=False,
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


    """
        Model loading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cli_args.model == 'best':
        assert os.path.isfile(os.path.join(run_dir, 'best_model.pth')), f'best_model.pth not found in {run_dir}'
        ckpt = torch.load(os.path.join(run_dir, 'best_model.pth'), map_location=torch.device(device))
    elif cli_args.model == 'last':
        assert os.path.isfile(os.path.join(run_dir, 'last_checkpoint.pth')), f'last_checkpoint.pth not found in {run_dir}'
        ckpt = torch.load(os.path.join(run_dir, 'last_checkpoint.pth'), map_location=torch.device(device))
    elif 'intermediate' in cli_args.model:
        assert os.path.isfile(os.path.join(run_dir, f'{cli_args.model}.pth')), f'{cli_args.model}.pth not found in {run_dir}'
        ckpt = torch.load(os.path.join(run_dir, f'{cli_args.model}.pth'), map_location=torch.device(device))
    else:
        assert os.path.isfile(os.path.join(run_dir, cli_args.model)), f'given model name {cli_args.model} not found in {run_dir}. You can use placeholders `best` and `last` instead of specifying a filename.'
        ckpt = torch.load(os.path.join(run_dir, cli_args.model), map_location=torch.device(device))

    model = get_model(config=config,
                      which=config.model.backbone,
                      io_type=config.task_name,
                      device=device)
    try:
        model.load_state_dict(ckpt['model'], strict=True)
    except:
        # Try retro-compatible version
        model = get_model(config=config,
                          which=config.model.backbone+"_retrocompatible",
                          io_type=config.task_name,
                          device=device)

        model.load_state_dict(ckpt['model'], strict=True)
        print('\n\nWARNING! Using the _retrocompatible model backbone. The current version of the backbone is no longer compatible with this checkpoint.')

    model.to(device)
    model.eval()

    metrics_handler = MetricsHandler(config=config, metrics=config.eval_metrics, renormalize_output_config=renormalize_output_config)
    loss_handler = LossHandler(config.loss, config=config)


    """
        Test model on training and test sets (or cli_args.split)
    """
    print('====== TESTING MODEL ON DATASET:', config.dataset,'======')

    # save results to a custom directory to avoid replacing results saved after training time
    save_dir = get_test_results_save_dir_name(config, cli_args)
    if cli_args.save:
        create_dirs(save_dir)
        save_config(config, save_dir)  # save config for rendering results in that directory. config file may change at test time if --target or --data_scale_factor are defined.
    save_args = {'save_dir': save_dir, 'eval_ckpt': cli_args.model}

    if cli_args.split is None or cli_args.split == 'train':
        eval_loss, eval_loss_list, eval_metrics = test(model, tr_loader, loss_handler=loss_handler, metrics_handler=metrics_handler, save=cli_args.save, args=config, **{'split': 'train', **save_args})
        print('TRAIN SET:')
        loss_handler.pprint(eval_loss_list, prefix='Train losses:')
        metrics_handler.pprint(eval_metrics, prefix='Train metrics:')
    if (cli_args.split is None or cli_args.split == 'test') and (not config.overfitting):
        eval_loss, eval_loss_list, eval_metrics = test(model, te_loader, loss_handler=loss_handler, metrics_handler=metrics_handler, save=cli_args.save, args=config, **{'split': 'test',  **save_args})
        print('TEST SET:')
        loss_handler.pprint(eval_loss_list, prefix='Test losses:')
        metrics_handler.pprint(eval_metrics, prefix='Test metrics:')

    if cli_args.save:
        print('Results saved successfully in', save_dir)



@torch.no_grad()
def test(model, loader, loss_handler, metrics_handler=None, save=False, **save_args):
    """Test model on dataloader"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tot_loss = 0
    tot_loss_list = np.zeros(len(loss_handler.loss))
    data_count = 0
    tot_metric_list = np.zeros(metrics_handler.tot_num_of_metrics())
    all_ms = []

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
        
        # Compute inference time on one sample
        start = time.time()
        _, _, _, _ = model(point_cloud[:1, ...])
        ms = (time.time() - start)*1000
        all_ms.append(ms)

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

        if save and (save_args['split'] != 'train' or i == 0):  # Save first training batch only for training set
            data = {'dirnames': dirname,
                    'traj': traj.detach().cpu().numpy(),
                    'stroke_ids': stroke_ids.detach().cpu().numpy(),
                    'stroke_ids_as_pc': stroke_ids_as_pc.detach().cpu().numpy(),
                    'traj_as_pc': traj_as_pc.detach().cpu().numpy(),
                    'traj_pred': traj_pred.detach().cpu().numpy(),
                    'pred_stroke_masks': pred_stroke_masks.detach().cpu().numpy(),
                    'stroke_masks_scores': mask_scores.detach().cpu().numpy(),
                    'seg_logits': seg_logits.detach().cpu().numpy() if seg_logits is not None else None,
                    'n_strokes': n_strokes,
                    'batch': i,
                    'suffix': str(save_args['split'])}
            np.save(os.path.join(save_args['save_dir'], str(save_args['eval_ckpt'])+'_'+str(save_args['split'])+'_batch'+str(i)+'.npy'), data)

        del point_cloud, traj, traj_pred, traj_as_pc, pred_stroke_masks, seg_logits
    


    print(f'Elapsed: {round(np.mean(all_ms),1)}ms | FPS: {round((1000/np.mean(all_ms)),1)}')

    return (tot_loss * 1.0 / data_count,  # total loss
            tot_loss_list * 1.0 / data_count,   # list of each loss component
            tot_metric_list * 1.0 / data_count)   # list of evaluation metrics






if __name__ == '__main__':
    main()