import os
import pdb

import torch

try:
    from models.e2e.multipath import MultipathGen
except ImportError:
    print(f'Model import error: transformer (PolyGen) is not available.')
    pass
try:
    from models.concat.gnn_encoder import GNNEncoder
    from models.concat.nar_model import NARModel
except ImportError:
    print('Model import error: gnn is not available.')
    pass
try:
    from models.pointnet import PointNetRegressor, PointNetSegmenter, PointNetSegmenterConv1d
    from models.pointnet_deeper import PointNetRegressor as PointNetDeeperRegressor
    from models.pointnet2_cls_ssg import PointNet2Regressor_StrokeWise, PointNet2Regressor, PointNet2Regressor_3Dbbox, PointNet2Regressor_SoPs, PointNet2Regressor_StrokeMasks, PointNet2Regressor_StrokeMasks_RetroCompatible
    from models.pointnet2_seg import PointNet2Segmenter_v1
except ImportError:
    print('Model import error: pointnet, pointnet2, pointnet_segmenter, pointnet_segmenter_conv1d, pointnet2_segmenter_v1 and pointnet_deeper are not available.')
    pass
try:
    from models.mlp import MLPGenerator, MLPRegressor
except ImportError:
    print('Model import error: mlp_generator is not available.')
    pass

try:
    from models.point_transformer import PointTransformer
except ImportError:
    print('Model import error: PointTransformer is not available.')
    pass

from utils import orient_in
from utils.config import load_config
from utils.pointcloud import get_dim_orient_traj_points, get_dim_traj_points


def get_model(config, which, io_type=None, custom_model_config=None, device="cpu"):
    """
        ---
        io_type : str
                  input/output network type [paintnet, multipathregression, None]
                  - paintnet: the input and output are set to be
                  the object point-cloud and the segments respectively.
                  - multipathregression: the `stroke_pred` task (train_stroke.py) is instead considered. 
    """
    assert io_type in {None, 'paintnet', 'multipathregression', 'ContrastiveClustering', 'ODv1_strokeProposal', 'ODv1_strokeRollout', 'MaskPlanner', 'StrokeWise'}
    model_config = config.model if custom_model_config is None else custom_model_config  # use the "config.model" params by default

    model = get_raw_model(config=config, which=which, io_type=io_type)

    if model_config.pretrained:
        if model_config.pretrained_custom is None:
            model = init_from_pretrained(model, which=which, device=device)
        else:
            model = init_from_pretrained_custom(model, pretrained_custom=model_config.pretrained_custom, model_config=model_config, device=device)
    
    model.to(device)
    return model


def get_raw_model(config, which, io_type=None):
    if io_type is not None:
        io_info = get_io_info(io_type, config=config)  # input/output info for the network
        # inputdim, vector_outdim_transl, vector_outdim_orient, out_vectors = get_io_info(io_type, config=config)

    if which == 'pointnet': # PointNet
        assert io_info['vector_outdim_orient'] == 0, 'pointnet does not yet support output normals'
        return PointNetRegressor(out_vectors=io_info['out_vectors'],
                                 outdim=io_info['vector_outdim_transl'],
                                 hidden_size=config.model.hidden_size,
                                 affinetrans=config.model.affinetrans,
                                 in_channel=io_info['inputdim'])

    elif which == 'pointnet_segmenter':  # PointNet for segmentation: output per-point scores
        return PointNetSegmenter(outdim=config.latent_dim,
                                 affinetrans=config.model.affinetrans,
                                 inputdim=io_info['inputdim'],
                                 augment_point_features_by=(0 if not config.one_hot_encoding_sample else config.overfitting_n_samples))

    elif which == 'pointnet_segmenter_conv1d':  # PointNet with only conv1ds
        return PointNetSegmenterConv1d(outdim=config.latent_dim,
                                       lambda_points=config.lambda_points,
                                       input_normals_only=config.input_normals_only)

    elif which == 'pointnet2_segmenter_v1':  # PointNet++
        return PointNet2Segmenter_v1(outdim=config.latent_dim,
                                     input_orient_dim=get_dim_orient_traj_points(config.extra_data),
                                     lambda_points=config.lambda_points,
                                     ball_in_xyz_space=config.model.ball_in_xyz_space)

    elif which == 'pointnet_deeper':  # PointNet with 5 layers as feature extractor
        assert io_info['vector_outdim_orient'] == 0, 'pointnet does not yet support output normals'
        return PointNetDeeperRegressor(out_vectors=io_info['out_vectors'],
                                       outdim=io_info['vector_outdim_transl'],
                                       hidden_size=config.model.hidden_size,
                                       affinetrans=config.model.affinetrans)

    elif which == 'pointnet2':  # PointNet++
        assert config['pc_points'] > 512, 'farthest point sampling set to 512'
        return PointNet2Regressor(out_vectors=io_info['out_vectors'],
                                  outdim=io_info['vector_outdim_transl'],
                                  outdim_orient=io_info['vector_outdim_orient'],
                                  weight_orient=config.weight_orient,
                                  hidden_size=config.model.hidden_size)

    elif which == 'pointnet2_strokemasks':  # PointNet++
        assert config['pc_points'] > 512, 'farthest point sampling set to 512'
        return PointNet2Regressor_StrokeMasks(out_vectors=io_info['out_vectors'],
                                              outdim=io_info['vector_outdim_transl'],
                                              outdim_orient=io_info['vector_outdim_orient'],
                                              weight_orient=config.weight_orient,
                                              hidden_size=config.model.hidden_size,
                                              pred_stroke_masks=True,
                                              n_stroke_masks=io_info['n_stroke_masks'],
                                              mask_confidence_scores=True,
                                              segment_confidence_scores=config.per_segment_confidence
                                              )

    elif which == 'pointnet2_strokemasks_retrocompatible':  # PointNet++
        assert config['pc_points'] > 512, 'farthest point sampling set to 512'
        return PointNet2Regressor_StrokeMasks_RetroCompatible(out_vectors=io_info['out_vectors'],
                                              outdim=io_info['vector_outdim_transl'],
                                              outdim_orient=io_info['vector_outdim_orient'],
                                              weight_orient=config.weight_orient,
                                              hidden_size=config.model.hidden_size,
                                              pred_stroke_masks=True,
                                              n_stroke_masks=io_info['n_stroke_masks'],
                                              mask_confidence_scores=True,
                                              segment_confidence_scores=config.per_segment_confidence
                                              )

    elif which == 'pointnet2_strokewise':  # PointNet++
        assert config['pc_points'] > 512, 'farthest point sampling set to 512'
        return PointNet2Regressor_StrokeWise(out_vectors=io_info['out_vectors'],
                                             outdim=io_info['vector_outdim_transl'],
                                             outdim_orient=io_info['vector_outdim_orient'],
                                             weight_orient=config.weight_orient,
                                             hidden_size=config.model.hidden_size,
                                             stroke_confidence_scores=True,
                                             point_confidence_scores=True,
                                             n_points_per_out_vector=config.max_n_stroke_points
                                              )

    elif which == 'pointnet2_3dbbox':  # PointNet++ regression of 3d bounding boxes
        assert config['pc_points'] > 512, 'farthest point sampling set to 512'
        return PointNet2Regressor_3Dbbox(out_bboxes=config.out_prototypes,
                                         hidden_size=config.proposal_model.hidden_size)

    elif which == 'pointnet2_sops':  # PointNet++ regression of start-of-path tokens
        assert config['pc_points'] > 512, 'farthest point sampling set to 512'
        return PointNet2Regressor_SoPs(out_vectors=config.out_prototypes,
                                       outdim=io_info['vector_outdim_transl'],
                                       outdim_orient=io_info['vector_outdim_orient'],
                                       weight_orient=config.weight_orient,
                                       hidden_size=config.proposal_model.hidden_size,
                                       sop_confidence_scores=True)

    elif which == 'mlp_rollout':
        return MLPRegressor(input_size=io_info['input_size'],
                            out_vectors=io_info['out_vectors'],
                            outdim_trasl=io_info['outdim_trasl'],
                            outdim_orient=io_info['outdim_orient'],
                            weight_orient=config.weight_orient,
                            hidden_sizes=config.rollout_model.hidden_size,
                            confidence_scores=io_info['end_of_path_confidence'])

    elif which == 'point_transformer':
        return PointTransformer(d_model=64,
                                nhead=4,
                                num_layers=2,
                                dim_feedforward=256,
                                max_seq_len=config.out_points_per_stroke,
                                input_dim=io_info['outdim']*config.lambda_points,
                                outdim=io_info['outdim'],
                                weight_orient=config.weight_orient)

    elif which == 'mlp_generator':  # MLP network for use with random input noise (GAN)
        assert io_info['vector_outdim_orient'] == 0, 'mlp generator network does not yet support output normals'
        return MLPGenerator(input_size=config.random_input_dim,
                            hidden_sizes=[512, 1024],
                            out_vectors=io_info['out_vectors'],
                            outdim=io_info['vector_outdim_transl'])

    elif which == 'samplenet':  # SampleNet (https://github.com/itailang/SampleNet)
        raise NotImplementedError('SampleNet is not yet implemented. Requires installing PytorchPointnet2 repo and others. Try on Hades first')
        sampler = SampleNet(
            num_out_points=io_info['out_vectors'],
            bottleneck_size=1024,
            group_size=config['sampler']['projection_group_size'],
            initial_temperature=1.0,
            input_shape="bcn",
            output_shape="bnc",
        )
        return sampler

    elif which == 'gnn':
        return NARModel(
            embedding_dim=config.gnn.embedding_dim,
            encoder_class=GNNEncoder,
            n_encode_layers=config.gnn.n_encode_layers,
            lambda_points=config.lambda_points,
            outdim=get_dim_traj_points(config.extra_data),
            aggregation=config.gnn.aggregation,
            normalization=config.gnn.normalization,
            learn_norm=config.gnn.learn_norm,
            track_norm=config.gnn.track_norm,
            mask_graph=False)

    elif which == 'transformer':
        return MultipathGen(config)

    raise ValueError(f'Network backbone not found: {which}')


def get_io_info(io_type, config):
    """Returns input/output information for the network (layer sizes, etc.).
        It uses the string io_type to encode different configurations.
    """
    if io_type == 'ODv1_strokeProposal':
        info = {}
        outdim = get_dim_traj_points(config.extra_data)
        orient_outdim = get_dim_orient_traj_points(config.extra_data)

        if config.stroke_prototype_kind == 'start_of_path_token':
            assert config.stroke_prototype_dim % outdim == 0 and config.stroke_prototype_dim // outdim == config.start_of_path_token_length

            info = {
                'vector_outdim_transl': (outdim - orient_outdim) * config.start_of_path_token_length,
                'vector_outdim_orient': orient_outdim * config.start_of_path_token_length
            }

        return info

    elif io_type == 'ODv1_strokeRollout':
        input_size = config.stroke_prototype_dim  # e.g. 6 for 3D bboxes
        if config.rollout_model.object_features:
            input_size += 1024  # concat object point-cloud global features

        outdim = get_dim_traj_points(config.extra_data)
        outdim_orient = get_dim_orient_traj_points(config.extra_data)

        end_of_path_confidence = False

        if 'mse_strokes' in config.rollout_loss:
            out_vectors = config.stroke_points
        elif 'chamfer_strokes' in config.rollout_loss:
            out_vectors = config.out_segments_per_stroke
        elif 'masked_mse_strokes' in config.rollout_loss:
            out_vectors = config.out_points_per_stroke
            end_of_path_confidence = True
        elif 'masked_mse_strokes_from_segments' in config.rollout_loss:
            out_vectors = config.out_points_per_stroke
        elif 'mse_nexttoken' in config.rollout_loss:
            out_vectors = 1
            input_size += (config.substroke_points - 1) * outdim * config.lambda_points  # autoregressive_v1: additionally conditioned on the previously predicted segments
            assert not config['end_of_path_confidence'], 'Not yet implemented for autoregressive_v1'
        elif 'mse_nexttoken_v2' in config.rollout_loss:
            out_vectors = 1
            input_size += (config.substroke_points) * outdim * config.lambda_points  # autoregressive_v2: additionally conditioned on the previously predicted segments. NOTE: `substroke_points` is now the length history.
            end_of_path_confidence = config['end_of_path_confidence']


        info = {
                'input_size': input_size,
                'outdim_trasl': (outdim - outdim_orient) * config.lambda_points,
                'outdim_orient': outdim_orient * config.lambda_points,
                'out_vectors': out_vectors,
                'outdim': outdim,
                'end_of_path_confidence': end_of_path_confidence
        }
        return info

    elif io_type == 'paintnet':
        inputdim = 3  # 3-d points of the object point-cloud
        outdim = get_dim_traj_points(config.extra_data)
        orient_outdim = get_dim_orient_traj_points(config.extra_data)

        vector_outdim_transl =  (outdim - orient_outdim) * config.lambda_points  # Translation dimensionality of each output vector
        vector_outdim_orient = orient_outdim * config.lambda_points  # Orientation dimensionality of each output vector
        out_vectors = (config.traj_points-config.lambda_points)//(config.lambda_points-config.overlapping) + 1   # Rounded number of overlapping sequences
        print('Number of output vectors (mini-sequences or single poses):', out_vectors)

        info = {
            'inputdim': inputdim,
            'out_vectors': out_vectors,
            'vector_outdim_transl': vector_outdim_transl,
            'vector_outdim_orient': vector_outdim_orient
        }

        return info

    elif io_type == 'MaskPlanner':
        # predict stroke masks on top of segments

        inputdim = 3  # 3-d points of the object point-cloud
        outdim = get_dim_traj_points(config.extra_data)
        orient_outdim = get_dim_orient_traj_points(config.extra_data)

        vector_outdim_transl =  (outdim - orient_outdim) * config.lambda_points  # Translation dimensionality of each output vector
        vector_outdim_orient = orient_outdim * config.lambda_points  # Orientation dimensionality of each output vector

        if config.traj_with_equally_spaced_points:
            assert config.n_pred_traj_points is not None
            out_vectors = (config.n_pred_traj_points-config.lambda_points)//(config.lambda_points-config.overlapping) + 1   # Rounded number of overlapping sequences
        else:
            out_vectors = (config.traj_points-config.lambda_points)//(config.lambda_points-config.overlapping) + 1   # Rounded number of overlapping sequences

        print('Number of output vectors (mini-sequences or single poses):', out_vectors)

        n_stroke_masks = config.max_n_strokes

        info = {
            'inputdim': inputdim,
            'out_vectors': out_vectors,
            'vector_outdim_transl': vector_outdim_transl,
            'vector_outdim_orient': vector_outdim_orient,
            'n_stroke_masks': n_stroke_masks
        }

        return info

    elif io_type == 'StrokeWise':
        # predict long-horizon strokes directly

        inputdim = 3  # 3-d points of the object point-cloud
        outdim = get_dim_traj_points(config.extra_data)
        orient_outdim = get_dim_orient_traj_points(config.extra_data)

        vector_outdim_transl =  (outdim - orient_outdim) * config.max_n_stroke_points  # Translation dimensionality of each output vector
        vector_outdim_orient = orient_outdim * config.max_n_stroke_points  # Orientation dimensionality of each output vector

        out_vectors = config.max_n_strokes

        print('Number of output strokes:', out_vectors, 'number of points per stroke:', config.max_n_stroke_points)

        info = {
            'inputdim': inputdim,
            'out_vectors': out_vectors,
            'vector_outdim_transl': vector_outdim_transl,
            'vector_outdim_orient': vector_outdim_orient,
        }

        return info

    elif io_type == 'multipathregression':
        inputdim = 3  # 3-d points of the object point-cloud
        outdim = get_dim_traj_points(config.extra_data)
        orient_outdim = get_dim_orient_traj_points(config.extra_data)

        vector_outdim_transl = (outdim - orient_outdim) * config.stroke_points
        vector_outdim_orient = orient_outdim * config.stroke_points
        out_vectors = config.n_strokes

        print('Number of output vectors (mini-sequences or single poses):', out_vectors)

        info = {
            'inputdim': inputdim,
            'out_vectors': out_vectors,
            'vector_outdim_transl': vector_outdim_transl,
            'vector_outdim_orient': vector_outdim_orient
        }

        return info

    elif io_type == 'ContrastiveClustering':
        outdim = get_dim_traj_points(config.extra_data)
        orient_outdim = get_dim_orient_traj_points(config.extra_data)

        inputdim = outdim * config.lambda_points

        info = {
            'inputdim': inputdim
        }

        return info

    else:
        raise ValueError(f'io_type value is not valid: {io_type}')


def init_from_pretrained(model, which, device='cpu'):
    """Initialize feature encoder with a pretrained model on
    ShapeNet or similar datasets for common tasks
    (PartSeg, Classification, etc.)
    """
    if which == 'pointnet2' or which == 'pointnet2_sops' or which == 'pointnet2_3dbbox' or which == 'pointnet2_strokemasks' or which == 'pointnet2_strokemasks_retrocompatible' or which == 'pointnet2_strokewise':
        state_dict = torch.load(os.path.join('pretrained_models', 'pointnet2_cls_ssg.pth'), map_location=device)['model_state_dict']
        feature_encoder_state_dict = _filter_out_dict(state_dict, ['fc1.weight', 'fc1.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'fc2.weight', 'fc2.bias', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked', 'fc3.weight', 'fc3.bias'])
        model.load_state_dict(feature_encoder_state_dict, strict=False)
        return model

    elif which == 'pointnet2_segmenter_v1':
        state_dict = torch.load(os.path.join('pretrained_models', 'pointnet2_segmenter_v1_J0FH0.pth'), map_location=device)['model']
        model.load_state_dict(state_dict, strict=True)
        return model
    else:
        raise ValueError(f"No pretrained model exists for this backbone: {which}")


def init_from_pretrained_custom(model, pretrained_custom, model_config, device='cpu'):
    """Load custom model pre-trained with name `pretrained_custom`
        
        Used for few-shot experiments, where a model pre-trained on PaintNet is used.
    """
    if 'allow_different_configs' not in model_config or model_config.allow_different_configs is False:
        pre_args = load_config(os.path.join(pretrained_custom, 'config.yaml'))
        assert model_config.backbone == pre_args.model.backbone, 'Pretraining run has a different backbone.'

    state_dict = torch.load(os.path.join(pretrained_custom, 'last_checkpoint.pth'), map_location=device)['model']
    if model_config.load_strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        feature_encoder_state_dict = _filter_out_dict(state_dict, ['fc3.weight', 'fc3.bias', 'fc_normals.weight', 'fc_normals.bias'])
        model.load_state_dict(feature_encoder_state_dict, strict=False)
    return model


def _filter_out_dict(state_dict, remove_layers):
    """Filter out layers that you do not want to initialize with transfer learning"""
    pretrained_dict = {k: v for k, v in state_dict.items() if k not in remove_layers}
    return pretrained_dict