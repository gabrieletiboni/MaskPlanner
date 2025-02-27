"""Handler class for loss function terms

To add a loss term:
    - insert its name and its method name in the constructor
    - add the method implementation itself
    - add a --weight_<lossname> arg parameter
"""
from threadpoolctl import threadpool_limits
import pdb

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

try:
    from pytorch3d_chamfer import chamfer_distance 
except ImportError:
    print(f'Warning! Unable to import pytorch3d package.'\
          f'Chamfer distance with velocities won\'t be available.'\
          f'(Check troubleshooting.txt for info on how to install pytorch3d)')
    pass


from models.dgcnn import DGCNNDiscriminator
from models.pointnet2_cls_ssg import PointNet2Regressor
from models.pointnet import PointNetRegressor
from models.mlp import MLP
from models.gradient_penalty import GradientPenalty

from utils import orient_in
from utils.pointcloud import get_dim_traj_points, mean_knn_distance


class LossHandler():
    def __init__(self, loss, config=None):
        """
        loss : list of str
                   list of loss terms, each weighted by the
                   corresponding specified weight as command argument
        config : dict with loss term weights
        """
        self.loss_names =   ['chamfer',
                             'repulsion',
                             'mse',
                             'align',
                             'velcosine',
                             'intra_align',
                             'discriminator',
                             'wdiscriminator',
                             'attraction_chamfer',
                             'rich_attraction_chamfer',
                             'contrastive_v1',
                             'asymm_segment_chamfer',
                             'reverse_asymm_point_chamfer',
                             'stoch_reverse_asymm_segment_chamfer',
                             'reverse_asymm_segment_chamfer',
                             'chamfer_bbox',
                             'mse_strokes',
                             'chamfer_strokes',
                             'asymm_v6_chamfer_strokes',
                             'masked_mse_strokes',
                             'masked_mse_strokes_v2',
                             'symm_segment_chamfer',
                             'symm_point_chamfer',
                             'mse_nexttoken',  # used for autoregressive_v1 rollout task
                             'mse_nexttoken_v2',  # used for autoregressive_v2 rollout task
                             'emd',  # Earth mover's distance (hungarian matching + MSE)
                             'chamfer_with_stroke_masks',  # chamfer distance + loss on matched stroke masks
                             'asymm_v6_chamfer_with_stroke_masks',   # asymm_chamfer_v6.yaml + loss on matched stroke masks
                             'asymm_v11_chamfer_with_stroke_masks',   # asymm_chamfer_v11.yaml + loss on matched stroke masks
                             'symm_v1_chamfer_with_stroke_masks',
                             'masked_mse_strokes_from_segments',
                             'hungarian_SoPs'
                             ]  
        self.loss_methods = [self.get_chamfer,
                             self.get_repulsion,
                             self.get_mse,
                             self.get_align_loss,
                             self.get_vel_cosine,
                             self.get_intra_align,
                             self.get_discr_loss,
                             self.get_wdiscr_loss,
                             self.get_attraction_chamfer,
                             self.get_rich_attraction_chamfer,
                             self.get_contrastive_v1,
                             self.get_asymm_segment_chamfer,
                             self.get_reverse_asymm_point_chamfer,
                             self.get_stoch_reverse_asymm_segment_chamfer,
                             self.get_reverse_asymm_segment_chamfer,
                             self.get_chamfer_bbox,
                             self.get_mse_strokes,
                             self.get_chamfer_strokes,
                             self.get_asymm_v6_chamfer_strokes,
                             self.get_masked_mse_strokes,
                             self.get_masked_mse_strokes_v2,
                             self.get_symm_segment_chamfer,
                             self.get_symm_point_chamfer,
                             self.get_mse_nexttoken,
                             self.get_mse_nexttoken_v2,
                             self.get_emd,
                             self.get_chamfer_with_stroke_masks,
                             self.get_asymm_v6_chamfer_with_stroke_masks,
                             self.get_asymm_v11_chamfer_with_stroke_masks,
                             self.get_symm_v1_chamfer_with_stroke_masks,
                             self.masked_mse_strokes_from_segments,
                             self.get_hungarian_SoPs]

        self.loss_index = {loss_name: i for i, loss_name in enumerate(self.loss_names)}
        assert (set(loss) <= set(self.loss_names)), f'Specified loss list {loss} contains non-valid names ({self.loss_names})'

        self.loss = list(loss)
        self.config = config


        """
            Loss initializations
        """
        if 'discriminator' in self.loss:  # Initialize discriminator
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.D = DGCNNDiscriminator(inputdim=3, k=self.config['knn_gcn']).to(self.device)
            
            self.minimax_loss = nn.BCEWithLogitsLoss().cuda()
            self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.9, 0.999))

        if 'wdiscriminator' in self.loss:  # Initialize wasserstein discriminator
            assert not (config.discr_input_type == 'singlestrokes' and config.discr_backbone != 'mlp'), f'Discr input type "singlestrokes" only supports discr_backbone "mlp".'
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            outdim = get_dim_traj_points(config.extra_data)  # Single pose dimensionality

            if config.discr_input_type == 'pointcloud':
                discr_input_dim = outdim
            elif config.discr_input_type == 'strokecloud':
                discr_input_dim = outdim*config.stroke_points
            elif config.discr_input_type == 'singlestrokes':
                discr_input_dim = outdim*config.stroke_points
            else:
                ValueError(f'Discriminator input type is not valid: {config.discr_input_type}')
            
            if config.discr_backbone == 'dgcnn':
                self.D = DGCNNDiscriminator(inputdim=discr_input_dim, k=self.config['knn_gcn']).to(self.device)
            elif config.discr_backbone == 'pointnet2':
                self.D = PointNet2Regressor(inputdim=discr_input_dim,
                                            out_vectors=1,
                                            outdim=1,
                                            outdim_orient=0,
                                            hidden_size=[512, 128]).to(self.device)
            elif config.discr_backbone == 'pointnet':
                self.D = PointNetRegressor(inputdim=discr_input_dim,
                                           out_vectors=1,
                                           outdim=1,
                                           hidden_size=[512, 128]).to(self.device)
            elif config.discr_backbone == 'mlp':
                self.D = MLP(input_size=discr_input_dim, hidden_sizes=[512, 256, 128], output_size=1).to(self.device)
            else:
                raise ValueError(f'Discriminator backbone is not valid: {config.discr_backbone}')
            
            self.GradPenalty = GradientPenalty(self.config['discr_lambdaGP'], gamma=1, device=self.device)
            self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.9, 0.999))

        if 'contrastive_v1' in self.loss:
            self.margin = config.contrastive_loss_margin
            self.contrastive_balance_negatives = config.contrastive_balance_negatives
            self.max_workers = config.max_workers

        if 'masked_mse_strokes' in self.loss or \
           'masked_mse_strokes_v2' in self.loss:
            self.bcewithlogits = nn.BCEWithLogitsLoss(reduction='none').cuda()

        if 'emd' in self.loss or 'hungarian_SoPs' in self.loss:
            from models.hungarianMatcher import HungarianMatcher
            self.matcher = HungarianMatcher()

        """
            Asserts for loss compatibility
        """
        for l in self.loss:
            assert 'weight_'+str(l) in self.config.keys(), f'weight parameter does not exist in the current config' \
            f' for loss {l}. Make sure to include a --weight_<loss_name> arg par for each loss you use.'

        assert not ('chamfer' in self.loss and 'mse' in self.loss), f'Incompatible losses: chamfer with mse'

        if self.config['lambda_points'] > 1:
            assert set(loss) <=  {'hungarian_SoPs', 'masked_mse_strokes_from_segments', 'asymm_v6_chamfer_with_stroke_masks', 'symm_v1_chamfer_with_stroke_masks', 'asymm_v11_chamfer_with_stroke_masks', 'chamfer_with_stroke_masks', 'emd', 'chamfer', 'symm_segment_chamfer', 'symm_point_chamfer', 'intra_align', 'attraction_chamfer', 'rich_attraction_chamfer', 'repulsion', 'contrastive_v1', 'asymm_segment_chamfer', 'reverse_asymm_point_chamfer', 'stoch_reverse_asymm_segment_chamfer', 'reverse_asymm_segment_chamfer', 'chamfer_strokes', 'mse_nexttoken', 'mse_nexttoken_v2'}, 'Losses must be one of the following when lambda > 1.'

        assert not ('discriminator' in self.loss and 'wdiscriminator' in self.loss), 'Choose between Minimax discriminator and Wasserstein Discriminator'

        if 'intra_align' in self.loss:
            assert self.config['lambda_points'] > 3, 'Fitting a plane to 3 points in 3D would always have degenerate covariance matrix.'
        if 'align' in self.loss:
            assert 'mse' not in self.loss, 'Align loss is not meant to be used with MSE'
            assert self.config['knn_repulsion'] > 1, 'Using Align loss with 1 NN -> unexplained variance would always be zero.'
        if 'attraction_chamfer' in self.loss:
            assert self.config['lambda_points'] > 1
        if 'rich_attraction_chamfer' in self.loss:
            assert self.config['lambda_points'] > 1
            assert orient_in(self.config['extra_data'])[0]
            assert 'vel' not in self.config['extra_data']
        if 'asymm_segment_chamfer' in self.loss or 'stoch_reverse_asymm_segment_chamfer' in self.loss or 'reverse_asymm_point_chamfer' in self.loss or 'reverse_asymm_segment_chamfer' in self.loss:
            assert self.config['lambda_points'] > 1
        if 'masked_mse_strokes' in self.loss:
            assert self.config['lambda_points'] == 1, 'the number of GT points per stroke is computed automatically from the traj, and lambda must be one to create it correctly.'
        if 'masked_mse_strokes_v2' in self.loss:
            assert self.config['lambda_points'] == 1, 'the number of GT points per stroke is computed automatically from the traj, and lambda must be one to create it correctly.'
        if 'symm_point_chamfer' in self.loss:
            assert self.config['lambda_points'] > 1, 'symm_point_chamfer is designed for weight scheduling which progressively give more importance to segments predictions. Why are you using it with lambda=1?'
        return


    def compute(self, return_list=True, **loss_args):
        """Return loss function
    
        return_list: bool
                     if True, additionally return seperate loss terms as list
        """
        loss_val = 0
        loss_val_list = []

        for l in self.loss:  # Compute each loss term
            l_ind = self.loss_index[l]
            l_value = self.loss_methods[l_ind](  **loss_args  )  # (y_pred, y, **loss_args) as input parameters

            loss_val += self.config['weight_'+str(l)]*l_value  # Weight * loss_term
            loss_val_list.append(l_value.detach().cpu().numpy())

        if return_list:
            return loss_val, np.array(loss_val_list)
        else:
            return loss_val


    def log_on_wandb(self, loss_list, wandb, epoch, suffix='_train_loss'):
        """Log loss list on wandb"""
        loss_list_names = self.loss.copy()

        if 'discriminator' in self.loss or 'wdiscriminator' in self.loss:
            if self.last_discr_internal_loss is not None:
                loss_list_names.append('discr_internal')
                loss_list = np.append(loss_list, self.last_discr_internal_loss.detach().cpu().numpy())

        for loss_term, train_loss_term in zip(loss_list_names, loss_list):
                wandb.log({str(loss_term)+str(suffix): train_loss_term, "epoch": (epoch+1)})

    def pprint(self, loss_values, prefix=''):
        """Pretty print loss values"""
        print(prefix)
        for name, value in zip(self.loss, loss_values):
            print(f"{name}:\t{round(value, 3)}")
        print('------------')


    """
        
        Loss list

    """
    def get_discr_loss(self, y_pred, y, **args):
        """A discriminator is used to learn a loss
        function adversarially (mesh-agnostic).

        """
        y, y_pred = y.permute(0, 2, 1), y_pred.permute(0, 2, 1) # B, outdim*stroke_points, n_stroke

        ###### DISCRIMINATOR TRAINING ######
        if 'train' not in args or args['train'] == True:
            self.D.train()
            self.D.zero_grad()

            real_out = self.D(y)
            real_loss = self.minimax_loss(real_out, Variable(torch.ones(real_out.size()).to(self.device))) # -log(D(traj_real))

            fake_out = self.D(y_pred.detach())
            fake_loss = self.minimax_loss(fake_out, Variable(torch.zeros(fake_out.size()).to(self.device)))  # -log(1-D(traj_predicted))

            d_loss = self.config['weight_discr_training']*(real_loss + fake_loss)

            
            d_loss.backward()
            self.D_optimizer.step()

            self.last_discr_internal_loss = d_loss
        else:
            self.D.train(False)
            self.last_discr_internal_loss = torch.zeros(1)
        ####################################


        ###### Learned loss term #########
        D_out = self.D(y_pred)

        learned_loss = self.minimax_loss(D_out, Variable(torch.ones(D_out.size()).to(self.device)))  # -log(D(traj_predicted))
        ####################################

        return learned_loss


    def get_wdiscr_loss(self, y_pred, y, **args):
        """Wasserstein-loss discriminator
        https://github.com/jtpils/TreeGAN
        """
        if self.config.discr_input_type == 'pointcloud':  # Reshape strokes as 3D point-clouds
            outdim = get_dim_traj_points(self.config.extra_data)
            B = y.shape[0]
            y = y.reshape(B, -1, outdim)  # (B, traj_points, 3)
            y_pred = y_pred.reshape(B, -1, outdim)  # (B, traj_points, 3)
            
            y, y_pred = y.permute(0, 2, 1), y_pred.permute(0, 2, 1)  # B, outdim*stroke_points, n_strokes
        elif self.config.discr_input_type == 'strokecloud':
            y, y_pred = y.permute(0, 2, 1), y_pred.permute(0, 2, 1)  # B, outdim*stroke_points, n_strokes
        elif self.config.discr_input_type == 'singlestrokes':  # Stack all individual strokes in batch_size dimension
            outdim = get_dim_traj_points(self.config.extra_data)
            B = y.shape[0]
            y = y.view(B*self.config.n_strokes, -1)  # (B*n_strokes, stroke_points*outdim)
            y_pred = y_pred.view(B*self.config.n_strokes, -1) 

            if self.config.singlestrokes_norm:  # Standardize single strokes to zero mean
                y = y.reshape(B*self.config.n_strokes, self.config.stroke_points, outdim)  # shape is (B*n_strokes, N, 3)
                y_mean = y.mean(dim=1, keepdim=True)  # shape is (B*n_strokes, 1, 3)
                y = y - y_mean  # shape is (B*n_strokes, N, 3)

                y_pred = y_pred.reshape(B*self.config.n_strokes, self.config.stroke_points, outdim)  # shape is (B*n_strokes, N, 3)
                y_pred_mean = y_pred.mean(dim=1, keepdim=True)  # shape is (B*n_strokes, 1, 3)
                y_pred = y_pred - y_pred_mean  # shape is (B*n_strokes, N, 3)

                # --- Standardization to unit scale is ill-posed as cuboids have zero-variance on one dimension (flat strokes)
                # y_std = y.std(dim=1, keepdim=True)  # shape is (B*n_strokes, 1, 3)
                # y = y / y_std  # shape is (B*n_strokes, N, 3)

                # --- visualize set of points y
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(y[:6, :, 0].detach().cpu().numpy(), y[:6, :, 1].detach().cpu().numpy(), y[:6, :, 2].detach().cpu().numpy())
                # plt.show()

                y = y.reshape(B*self.config.n_strokes, self.config.stroke_points*outdim) # shape is (N*n_strokes, N*3)
                y_pred = y_pred.reshape(B*self.config.n_strokes, self.config.stroke_points*outdim) # shape is (N*n_strokes, N*3)
        

        # -------------------- Discriminator -------------------- #
        discr_train_frequency_mask = ('epoch' not in args) or (args['epoch'] % self.config.discr_train_freq == 0)  # Train discriminator once every self.config.discr_train_freq epochs
        discr_train_flag = ('train' not in args or args['train'] == True)

        if discr_train_flag and discr_train_frequency_mask:  # Train discriminator
            self.D.train()
            for d_iter in range(self.config['discr_train_iter']):
                self.D.zero_grad()        
                    
                D_real = self.D(y)
                D_realm = D_real.mean()

                D_fake = self.D(y_pred.detach())
                D_fakem = D_fake.mean()

                gp_loss = self.GradPenalty(self.D, y.data, y_pred.detach().data)
                
                d_loss = self.config.weight_discr_training*(-D_realm + D_fakem)
                d_loss_gp = d_loss + gp_loss

                d_loss_gp.backward()
                self.D_optimizer.step()

                self.last_discr_internal_loss = d_loss_gp
        else:
            self.last_discr_internal_loss = None

        
        # ---------------------- Generator ---------------------- #
        self.D.train(False)
        G_fake = self.D(y_pred)
        G_fakem = G_fake.mean()
        g_loss = -G_fakem

        return g_loss

    def get_wdiscr_loss_chatgpt(self, y_pred, y, **args):
        """Code generated by chatGPT when asked to create
            a WGAN"""

        # config.discr_on_point_clouds is deprecated (use discr_input_type)
        # if self.config.discr_on_point_clouds:  # Reshape strokes as 3D point-clouds
        #     outdim = get_dim_traj_points(self.config.extra_data)
        #     B = y.shape[0]
        #     y = y.reshape(B, -1, outdim)  # (B, traj_points, 3)
        #     y_pred = y_pred.reshape(B, -1, outdim)  # (B, traj_points, 3)

        y, y_pred = y.permute(0, 2, 1), y_pred.permute(0, 2, 1)  # B, outdim*stroke_points, n_strokes

        # -------------------- Discriminator -------------------- #
        discr_train_frequency_mask = ('epoch' not in args) or (args['epoch'] % self.config.discr_train_freq == 0)  # Train discriminator once every self.config.discr_train_freq epochs
        discr_train_flag = ('train' not in args or args['train'] == True)

        if discr_train_flag and discr_train_frequency_mask:  # Train discriminator
            self.D.train()
            for d_iter in range(self.config['discr_train_iter']):
                # Compute the output of the discriminator for the real and generated data
                real_output = self.D(y)
                generated_output = self.D(y_pred)

                # Compute the Wasserstein loss and gradients for the discriminator
                D_loss = wasserstein_loss_chatgpt(real_output, generated_output)
                D_loss.backward()

                # Update the weights of the discriminator using the optimizer
                self.D_optimizer.step()

                self.last_discr_internal_loss = D_loss

        else:
            self.last_discr_internal_loss = None

        # ---------------------- Generator ---------------------- #
        self.D.eval()

        generated_output = self.D(y_pred)

        # Compute the Wasserstein loss and gradients for the generator
        G_loss = wasserstein_loss(torch.ones_like(generated_output), generated_output, generator=True)
        # G_loss.backward()

        # Update the weights of the generator using the optimizer
        # optimizer_G.step()
        return G_loss


    def wasserstein_loss_chatgpt(real_output, generated_output, generator=False):
        # Compute the Wasserstein distance between the real and generated distributions
        if generator:
            w_loss = -torch.mean(generated_output)
        else:
            w_loss = -torch.mean(real_output) + torch.mean(generated_output)

        # Implement the gradient penalty term
        alpha = torch.rand(real_output.size(0), 1)
        interpolated = alpha * real_output + (1 - alpha) * generated_output
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
        interpolated_output = discriminator(interpolated)
        grad = torch.autograd.grad(outputs=interpolated_output, inputs=interpolated,
                                   grad_outputs=torch.ones_like(interpolated_output),
                                   create_graph=True, retain_graph=True)[0]
        grad_penalty = 10 * torch.mean((grad.norm(2, dim=1) - 1) ** 2)

        # Return the total loss
        return w_loss + grad_penalty


    def get_rich_attraction_chamfer(self, y_pred, **args):
        """first and last points are enriched with orientation and inferred velocity.

        See attraction_loss for the standard version.
        """
        outdim = get_dim_traj_points(self.config['extra_data'])

        starting_points = y_pred[:, :, :outdim]
        ending_points = y_pred[:, :, -outdim:]      

        inferred_vel_starting = y_pred[:, :, outdim:outdim+3] - y_pred[:, :, :3]
        inferred_vel_ending = y_pred[:, :, -outdim:-(outdim-3)] - y_pred[:, :, -(outdim*2):-(outdim*2-3)]

        starting_points = torch.cat((starting_points, inferred_vel_starting), dim=-1)
        ending_points = torch.cat((ending_points, inferred_vel_starting), dim=-1)

        if not self.config['soft_attraction']:
            # Full version (all points get attracted, a different 2nd-nn sequence is taken into account in case same sequence is 1st-nn)
            chamfer = 100*chamfer_distance(starting_points, ending_points, padded=False, avoid_in_sequence_collapsing=True)[0]
        else:
            # Soft version (only a few points are attracted, those whose 1-nn is not in-sequence)
            chamfer = 100*chamfer_distance(starting_points,
                                           ending_points,
                                           padded=False,
                                           avoid_in_sequence_collapsing=True,
                                           soft_attraction=True,
                                           point_reduction=None,
                                           batch_reduction=None)[0]

        return chamfer


    def get_contrastive_v1(self, latent_segments, stroke_ids, **args):
        """Pairwise contrastive loss as in https://arxiv.org/abs/2003.13834,
            with public code at: https://github.com/matheusgadelha/PointCloudLearningACD
            
            In practice, we have a loss term for all pair of latent_segments (i,j),
            where each pair is encouraged to be further if i and j don't belong to
            the same stroke, and viceversa.
        """
        with threadpool_limits(user_api="openmp", limits=self.max_workers):
            n_pts = latent_segments.size(1)

            feat = latent_segments.permute(0, 2, 1) 
            target = stroke_ids

            feat = F.normalize(feat, p=2, dim=1)
            pair_sim = torch.bmm(feat.transpose(1,2), feat)  # (n_pts, n_pts): for each point, dot product with all other points

            one_hot_target = F.one_hot(target).float()  # (n_pts, num_strokes): for each point, one-hot encoding of stroke it belongs to
            pair_target = torch.bmm(one_hot_target, one_hot_target.transpose(1,2))  # (n_pts, n_pts): for each point, 1 for points that belong to the same stroke, 0 otherwise

            cosine_loss = pair_target * (1. - pair_sim) + (1. - pair_target) * F.relu(pair_sim - self.margin)  # pair-wise contrastive loss (Eq. 4 in https://arxiv.org/abs/2003.13834): (n_pts, n_pts)

            with torch.no_grad():
                """
                    Balance positive and negative pairs.

                    In practice: positive are often much fewer than negatives.
                    Therefore, we only consider a subset of negatives that is roughly
                    equal to the number of positive.
                """
                if self.contrastive_balance_negatives:
                    pos_fraction = (pair_target.data == 1).float().mean()  # fraction of positives
                    sample_neg = torch.zeros(pair_target.shape, dtype=torch.float32, device='cuda').uniform_() > 1 - pos_fraction  # roughly as many points as positives
                else:
                    sample_neg = torch.zeros(pair_target.shape, dtype=torch.float32, device='cuda').uniform_() > 0
                sample_mask = (pair_target.data == 1) | sample_neg  # sample all positives, and a subset of the negatives

            diag_mask = 1 - torch.eye(n_pts)  # Discard diag elems, i.e. do not compute loss on pairs (i,j) for i=j. It's always zero but it affects the .mean()
            cosine_loss = diag_mask.unsqueeze(0).cuda() * sample_mask.type(torch.cuda.FloatTensor) * cosine_loss 
            total_loss = cosine_loss.mean()

        return total_loss


    def get_attraction_chamfer(self, y_pred, **args):
        """Chamfer loss between ending points (1st point-cloud) and starting points (2nd point-cloud).

        It encourages predicted mini-sequences to be contiguous.

        y_pred: (B, n_strokes, outdim*stroke_points) torch tensor
        """
        starting_points = y_pred[:, :, :3]
        ending_points = y_pred[:, :, -3:]

        chamfer = 100*chamfer_distance(starting_points, ending_points, padded=False)[0]
        return chamfer


    def get_chamfer(self, y_pred, y, **args):
        """Compute chamfer distance

            y: (B, n_strokes, outdim*stroke_points) torch tensor
            y_pred: (B, n_strokes, outdim*stroke_points) torch tensor
        """
        if 'vel' in self.config['extra_data']:  # Fallback to custom chamfer distance for velocities
            chamfer = 100*chamfer_distance(y_pred, y, velocities=True)[0]
        
        # elif self.config['lambda_points'] > 1 or self.config['stroke_pred'] == True:
        is_gt_padded = True if self.config['stroke_pred'] == False else False  # No padding if stroke_pred (TEMPORARY)
        chamfer = 100*chamfer_distance(y_pred, y, padded=is_gt_padded, min_centroids=self.config['min_centroids'])[0]  # Handle padded GT trajs for dataloader

        return chamfer


    def _transform_segment_distance_to_confidence(self, distance):
        """Transformation: a given distance among segments is mapped to
            a value in [0, 1] as described in https://www.desmos.com/calculator/esc9rs7jl2
            such that higher distance leads to lower confidence values.
        """
        # Coeffs
        c = 2.17
        d = -4.63

        return -1 * (  1 / (  1 + torch.exp(-c*torch.log10(distance) + d)  )  ) + 1


    def _get_per_segment_confidence_loss(self, nn_distance, logits):
        """Learn a confidence score for each predicted segment,
           proportionally to how close it is to the nearest GT segment.
           In practice, the target is set using such transformation
           (https://www.desmos.com/calculator/esc9rs7jl2) and it's
           followed with an L2 loss.
           This is inspired by the confidence learned in YOLO, which aims
           at learning the IoU with the GT with an L2 loss.

        Params:
            nn_distance: [B, out_segments]
                          distance with nearest GT segment, for each predicted segment

            logits: [B, out_segments]
                    network pred confidence logits in [0, 1] for each predicted segment

        Returns:
            L2 loss scalar
        """

        # Get targets for the given distances
        targets = self._transform_segment_distance_to_confidence(nn_distance)

        # L2 loss
        per_segment_confidence_loss = (logits - targets).square().sum(-1).mean()

        loss = self.config['explicit_weight_segments_confidence']*per_segment_confidence_loss
        return loss    


    def get_asymm_v6_chamfer_with_stroke_masks(self, y_pred, y, pred_stroke_masks, mask_scores, seg_logits, stroke_ids, traj_as_pc, **kwargs):
        """Computes:
            - asymm_chamfer_v6.yaml distance among segments
            - (optional) loss on confidence for each segment (proportional to closest GT segment)
            - loss on stroke masks
        """

        # 1. asymm_segment_chamfer
        preds_to_gt_segments_chamfer_noReduction, _, pred_to_gt_match, _ = chamfer_distance(y_pred,
                                                                                            y,
                                                                                            padded=True,
                                                                                            asymmetric=True,
                                                                                            return_matching=True,
                                                                                            point_reduction=None,
                                                                                            batch_reduction=None)
        preds_to_gt_segments_chamfer = 100*(preds_to_gt_segments_chamfer_noReduction.mean())



        # 1.1 per-segment confidence loss, proportional to distance
        if self.config.per_segment_confidence:
            per_segment_confidence_loss = self._get_per_segment_confidence_loss(nn_distance=preds_to_gt_segments_chamfer_noReduction,
                                                                                logits=seg_logits)
        else:
            per_segment_confidence_loss = 0



        # 2. reverse_asymm_point_chamfer
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if not traj_as_pc.is_cuda:
            traj_as_pc = traj_as_pc.to('cuda', dtype=torch.float)

        point_wise_y_pred = y_pred.reshape(B, -1, outdim)  # From pred segments to point-cloud

        gt_to_preds_points_chamfer = 100*chamfer_distance(point_wise_y_pred,
                                                          traj_as_pc,
                                                          padded=True,
                                                          reverse_asymmetric=True)[0]  # reverse asymmetric instead of reverting the first two arguments, because padding only exists in the second argument
        traj_as_pc = traj_as_pc.cpu()



        # 3. reverse_asymm_segment_chamfer
        gt_to_preds_segment_chamfer = 100*chamfer_distance(y_pred,
                                                           y,
                                                           padded=True,
                                                           reverse_asymmetric=True)[0]  # reverse asymmetric instead of reverting the first two arguments, because padding only exists in the second argument

        

        # 4. stroke masks loss
        stroke_masks_loss = self.get_stroke_masks_loss(pred_to_gt_match,
                                                       pred_stroke_masks,
                                                       mask_scores,
                                                       stroke_ids,
                                                       nn_distance=preds_to_gt_segments_chamfer_noReduction,
                                                       smooth_targets=self.config.smooth_target_stroke_masks,
                                                       **kwargs)



        loss = self.config['weight_asymm_segment_chamfer']*preds_to_gt_segments_chamfer + \
               per_segment_confidence_loss + \
               self.config['weight_reverse_asymm_point_chamfer']*gt_to_preds_points_chamfer + \
               self.config['weight_reverse_asymm_segment_chamfer']*gt_to_preds_segment_chamfer + \
               stroke_masks_loss

        return loss


    def get_asymm_v11_chamfer_with_stroke_masks(self, y_pred, y, pred_stroke_masks, mask_scores, seg_logits, stroke_ids, traj_as_pc, **kwargs):
        """Computes:
            - asymm_chamfer_v11.yaml chamfer distance + loss on stroke masks
            - forward pred-to-gt: segment-wise 
            - reverse gt-to-pred: point-wise
        """

        # 1. asymm_segment_chamfer
        preds_to_gt_segments_chamfer_noReduction, _, pred_to_gt_match, _ = chamfer_distance(y_pred,
                                                                                            y,
                                                                                            padded=True,
                                                                                            asymmetric=True,
                                                                                            return_matching=True,
                                                                                            point_reduction=None,
                                                                                            batch_reduction=None)
        preds_to_gt_segments_chamfer = 100*(preds_to_gt_segments_chamfer_noReduction.mean())



        # 1.1 per-segment confidence loss, proportional to distance
        if self.config.per_segment_confidence:
            per_segment_confidence_loss = self._get_per_segment_confidence_loss(nn_distance=preds_to_gt_segments_chamfer_noReduction,
                                                                                logits=seg_logits)
        else:
            per_segment_confidence_loss = 0



        # 2. reverse_asymm_point_chamfer
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if not traj_as_pc.is_cuda:
            traj_as_pc = traj_as_pc.to('cuda', dtype=torch.float)

        point_wise_y_pred = y_pred.reshape(B, -1, outdim)  # From pred segments to point-cloud

        gt_to_preds_points_chamfer = 100*chamfer_distance(point_wise_y_pred,
                                                  traj_as_pc,
                                                  padded=True,
                                                  reverse_asymmetric=True)[0]  # reverse asymmetric instead of reverting the first two arguments, because padding only exists in the second argument
        traj_as_pc = traj_as_pc.cpu()



        # 3. stroke masks loss
        stroke_masks_loss = self.get_stroke_masks_loss(pred_to_gt_match,
                                                       pred_stroke_masks,
                                                       mask_scores,
                                                       stroke_ids,
                                                       nn_distance=preds_to_gt_segments_chamfer_noReduction,
                                                       smooth_targets=self.config.smooth_target_stroke_masks,
                                                       **kwargs)



        loss = self.config['weight_asymm_segment_chamfer']*preds_to_gt_segments_chamfer + \
               per_segment_confidence_loss + \
               self.config['weight_reverse_asymm_point_chamfer']*gt_to_preds_points_chamfer + \
               stroke_masks_loss

        return loss


    def get_symm_v1_chamfer_with_stroke_masks(self, y_pred, y, pred_stroke_masks, mask_scores, seg_logits, stroke_ids, traj_as_pc, **kwargs):
        """Computes:
            - symm_v1.yaml chamfer distance + loss on stroke masks
            - you start point-wise, and progressively give more importance to segment-wise
        """
        if self.config.smooth_target_stroke_masks:
            raise NotImplementedError()
        if self.config.per_segment_confidence:
            raise NotImplementedError()


        # 1. symm_segment_chamfer
        symm_segment_wise, _, pred_to_gt_match, _ = chamfer_distance(y_pred, y, padded=True, return_matching=True)  # pred_to_gt_match: [B, num_pred_segments]
        symm_segment_wise *= 100


        # 2. symm_point_chamfer
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if not traj_as_pc.is_cuda:
            traj_as_pc = traj_as_pc.to('cuda', dtype=torch.float)

        point_wise_y_pred = y_pred.reshape(B, -1, outdim)  # From pred segments to point-cloud

        symm_point_wise = 100*chamfer_distance(point_wise_y_pred,
                                               traj_as_pc,
                                               padded=True)[0]
        traj_as_pc = traj_as_pc.cpu()


        # 3. stroke masks loss
        stroke_masks_loss = self.get_stroke_masks_loss(pred_to_gt_match,
                                                       pred_stroke_masks,
                                                       mask_scores,
                                                       stroke_ids,
                                                       **kwargs)



        loss = self.config['weight_symm_segment_chamfer']*symm_segment_wise + \
               self.config['weight_symm_point_chamfer']*symm_point_wise + \
               stroke_masks_loss

        return loss


    def get_chamfer_with_stroke_masks(self, y_pred, y, pred_stroke_masks, mask_scores, stroke_ids, **kwargs):
        """Computes (1) chamfer distance among segments + (2) loss on stroke masks (target stroke id matched
           using closest GT segment to pred i-th segment)
        """
        if self.config.smooth_target_stroke_masks:
            raise NotImplementedError()
        if self.config.per_segment_confidence:
            raise NotImplementedError()
            
        # Chamfer loss for segments prediction
        chamfer, _, pred_to_gt_match, _ = chamfer_distance(y_pred, y, padded=True, return_matching=True)  # pred_to_gt_match: [B, num_pred_segments]
        chamfer *= 100

        stroke_masks_loss = self.get_stroke_masks_loss(pred_to_gt_match,
                                                       pred_stroke_masks,
                                                       mask_scores,
                                                       stroke_ids,
                                                       **kwargs)
        
        loss = chamfer + stroke_masks_loss

        return loss


    def _compute_stroke_mask_loss(self, input, target, kind='bce'):
        """Compute loss on given stroke masks.
            No batch reduction is performed.
        """
        if kind == 'bce':
            return F.binary_cross_entropy_with_logits(input, target, reduction="none").sum(-1)
        elif kind == 'mse':
            return (input - target).square().sum(-1)
        else:
            raise NotImplementedError()


    def get_stroke_masks_loss(self, pred_to_gt_match, pred_stroke_masks, scores, stroke_ids, nn_distance=None, smooth_targets=False, **kwargs):
        """Loss on predicted stroke masks (target stroke id matched
           using closest GT segment to pred i-th segment)
        
        Params:
            nn_distance: [B, out_segments]
                          distance with nearest GT segment, for each predicted segment
            smooth_targets: bool
                            if set, the target_stroke_ids are transformed into continuous-valued
                            stroke masks instead of binary stroke masks, where the positive target values
                            of 1 are replaced with f(distance), i.e. a value in [0, 1] inv. proportional to the distance
                            with the closest GT segment. This way, we implicitly learn a confidence on the segment,
                            and do not learn very confident stroke masks for segments that are badly placed.
        """
        stroke_mask_loss_kind = 'bce' if smooth_targets is False else 'mse'
        
        """
            Construct target_stroke_masks for predicted segments

            stroke id of pred i-th segment = stroke id of GT segment that's closest to pred i-th segment
        """
        # Assign stroke_ids to pred segments according to closest GT segment
        target_stroke_ids = stroke_ids.cuda().gather(dim=1, index=pred_to_gt_match)  # target_stroke_ids [B, out_segments]

        # Split for readability
        if smooth_targets:
            # Create real-valued stroke masks from stroke_ids
            target_stroke_masks = [self._from_stroke_ids_to_masks(batch_target_stroke_ids, nn_distance=batch_nn_distance, smooth_targets=smooth_targets)
                                   for batch_target_stroke_ids, batch_nn_distance in zip(target_stroke_ids, nn_distance)]  # list of size B [n_strokes[b], out_segments]
        else:
            # Create binary stroke masks from stroke ids,
            target_stroke_masks = [self._from_stroke_ids_to_masks(batch_target_stroke_ids)
                                   for batch_target_stroke_ids in target_stroke_ids]  # list of size B [n_strokes[b], out_segments]


        # Temp sanity checks
        assert not torch.any(target_stroke_ids == -1), 'temp sanity check: no pred segment should be associated with the fake stroke id -1'
        if not smooth_targets:
            assert torch.all(torch.stack([torch.all(b_target_stroke_mask.sum(0) == 1) for b_target_stroke_mask in target_stroke_masks])), 'temp sanity check: masks should be mutually exclusive across strokes, hence all equal to ones when summed.'


        """
            Find hungarian matching between pred_stroke_masks and target_stroke_masks
        """
        B, n_pred_masks, out_segments = pred_stroke_masks.shape
        indices = []
        with torch.no_grad():
            for b, (b_pred_stroke_masks, b_target_stroke_masks) in enumerate(zip(pred_stroke_masks, target_stroke_masks)):  # iterate over batch elements
                # Compute cost matrix for this batch
                n_target_masks = b_target_stroke_masks.shape[0]

                # all pairs in single-column format for loss computation ([n_pred_masks, n_target_masks] as [n_pred_masks*n_target_masks, 1])
                exp_b_pred_stroke_masks = b_pred_stroke_masks.repeat_interleave(n_target_masks, dim=0)
                exp_b_target_stroke_masks = b_target_stroke_masks.repeat(n_pred_masks, 1)
                
                # bce = F.binary_cross_entropy_with_logits(exp_b_pred_stroke_masks, exp_b_target_stroke_masks.float(), reduction="none").sum(-1)
                bce = self._compute_stroke_mask_loss(exp_b_pred_stroke_masks, exp_b_target_stroke_masks.float(), kind=stroke_mask_loss_kind)
                bce = bce.view(n_pred_masks, n_target_masks).cpu()  # cost matrix [n_pred_masks, n_target_masks]
                
                indices.append(linear_sum_assignment(bce))  # solve LAP

        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  # list of size B, with elements (index_i, index_j). index_* is Tensor


        """
            Compute stroke masks loss given optimal matching
        """
        pred_idx = self._get_pred_permutation_idx(indices)  # tuple for indexing pred masks
        gt_idx = self._get_gt_permutation_idx(indices)  # tuple for indexing target masks

        matched_pred_masks = pred_stroke_masks[pred_idx]  # (stacked) sub-selected pred masks given optimal matching ([tot num of gt masks in batch, out_segments])

        # Pad target masks with fake masks for easy indexing (fake stroke masks aren't used in the LAP, so they won't be matched to any pred mask anyways)
        if smooth_targets:
            padded_target_stroke_masks = torch.stack([self._concat_fake_vectors(
                                                          self._from_stroke_ids_to_masks(batch_target_stroke_ids, nn_distance=batch_nn_distance, smooth_targets=smooth_targets),
                                                          tot_desired=n_pred_masks
                                                      )
                                                      for batch_target_stroke_ids, batch_nn_distance in zip(target_stroke_ids, nn_distance)])  # Tensor [B, n_pred_masks, out_segments]
        else:
            padded_target_stroke_masks = torch.stack([self._concat_fake_vectors(
                                                        self._from_stroke_ids_to_masks(batch_target_stroke_ids),
                                                        tot_desired=n_pred_masks
                                                      )
                                                      for batch_target_stroke_ids in target_stroke_ids])  # Tensor [B, n_pred_masks, out_segments]
        
        matched_target_masks = padded_target_stroke_masks[gt_idx]  # (stacked) selected gt masks given optimal matching ([tot num of gt masks in batch, out_segments])
        assert not torch.any(matched_target_masks == -100), 'temp sanity check: no fake stroke masks should be selected given the matching'

        # stroke_mask_loss = F.binary_cross_entropy_with_logits(matched_pred_masks, matched_target_masks.float(), reduction="none").sum(-1).mean()
        stroke_mask_loss = self._compute_stroke_mask_loss(matched_pred_masks, matched_target_masks.float(), kind=stroke_mask_loss_kind).mean()
                           # F.loss -> [tot num of gt masks in batch, out_segments]
                           # .sum(-1) -> [tot num of gt masks in batch,]
                           # .mean() -> []



        """
            Compute confidence loss (`strokeness`)

            scores: Tensor of dim [B, max_n_strokes]
                    Confidence scores for each pred mask
        """
        # targets for scores (1.0 for masks matched with Hungarian to GT masks, 0 otherwise)
        target_scores = torch.zeros(scores.shape)
        target_scores[pred_idx] = 1.

        # weights for BCE
        weights = self.config['explicit_no_stroke_weight']*torch.ones(scores.shape)  # less weight to predictions of "no stroke", as often times there are many more predicted masks than needed
        weights[pred_idx] = 1.

        target_scores = target_scores.to(scores.get_device())
        weights = weights.to(scores.get_device())

        confidence_loss = F.binary_cross_entropy_with_logits(scores, target_scores, reduction="none", weight=weights).mean()  # mean over all predicted masks



        loss = self.config['explicit_weight_stroke_masks']*stroke_mask_loss + self.config['explicit_weight_stroke_masks_confidence']*confidence_loss
        return loss


    def _from_stroke_ids_to_masks(self, stroke_ids, smooth_targets=False, nn_distance=None):
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


    def _concat_fake_vectors(self, tens, tot_desired):
        """Concat a number of fake tensors to first dim of given tensor `tens`,
            so that total number of vectors is tot_desired
            
            fake tensors have a symbolic value of -100
        """        
        pad_value = -100

        shape = tens.shape
        n_fakes = tot_desired - tens.shape[0]
        
        if n_fakes > 0:
            fake_shape = list(tens.shape)
            fake_shape[0] = n_fakes

            return torch.cat((tens, pad_value*torch.ones(fake_shape).to(tens.device)), dim=0)
        else:
            return tens


    def get_emd(self, y_pred, y, **kwargs):
        """Computes Earth Mover's distance (hungarian match + MSE loss between matched segments)
        
        Params:
            y_pred: predicted segments
            y: padded GT segments
        """
        # Remove fake GT segments
        y_unpadded_list = [self.remove_padding_from_tensors(gt_segments) for gt_segments in y]

        indices = self.matcher(outputs=y_pred, targets=y_unpadded_list)  # list of size B, with elements (index_i, index_j). index_* is tensor

        pred_idx = self._get_pred_permutation_idx(indices)  # tuple for indexing predictions
        gt_idx = self._get_gt_permutation_idx(indices)  # tuple for indexing GT

        matched_y_pred = y_pred[pred_idx]  # stacked pred segments in order of matching
        matched_y = y[gt_idx]  # stacked gt segments in order of matching
        
        # MSE
        return (matched_y_pred - matched_y).square().sum(-1).mean()

    def _get_pred_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        pred_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, pred_idx

    def _get_gt_permutation_idx(self, indices):
        # Permute gt following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        gt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, gt_idx


    def get_chamfer_bbox(self, bbox_pred, bbox_gt, **kwargs):
        """
            Compute chamfer distance among predicted
            3D bboxes and GT 3D bboxes
        """
        
        # GT 3D bboxes are padded with -100
        chamfer = 100*chamfer_distance(bbox_pred, bbox_gt, padded=True)[0]
        return chamfer


    def get_symm_segment_chamfer(self, y_pred, y, **args):
        """Both forward and reverse terms are computed on segments.
            
            Same as `get_chamfer`. Just a duplicate for clarity in the
            naming convention.
        """
        return self.get_chamfer(y_pred, y, **args)


    def get_symm_point_chamfer(self, y_pred, y, traj_as_pc, **args):
        """Both forward and reverse terms are computed on single poses.

            Same as `get_chamfer`, but single poses are considered even
            when lambda_points > 1. Used for the dynamic re-weighting experiments
            where more importance to per-point predictions are given first,
            and then more importance to segments later on.
        """
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if not traj_as_pc.is_cuda:
            traj_as_pc = traj_as_pc.to('cuda', dtype=torch.float)

        # From pred segments to point-cloud
        point_wise_y_pred = y_pred.reshape(B, -1, outdim)

        # Symmetric CD
        symm_point_chamfer = 100*chamfer_distance(point_wise_y_pred,
                                                  traj_as_pc,
                                                  padded=True)[0]

        traj_as_pc = traj_as_pc.cpu()

        return symm_point_chamfer


    def get_asymm_segment_chamfer(self, y_pred, y, **args):
        """Asymmetric (forward) segment-wise chamfer
            
            Computes asymmetric chamfer from predicted segments
            to GT segments, but not the opposite.
            Hence, predicted segments may all collapse to a single GT segments
            if only this loss were to be used.
        """
        preds_to_gt_chamfer = 100*chamfer_distance(y_pred,
                                                   y,
                                                   padded=True,
                                                   asymmetric=True)[0]

        
        return preds_to_gt_chamfer


    def get_reverse_asymm_point_chamfer(self, y_pred, y, traj_as_pc, **args):
        """Asymmetric point-wise chamfer, from GT to pred.

            Computes asymmetric chamfer from GT points to
            predicted points, but not the oppose.
            Hence, predicted segments will try to stay around all GT
            points to minimize this loss.
        """
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if not traj_as_pc.is_cuda:
            traj_as_pc = traj_as_pc.to('cuda', dtype=torch.float)

        # From pred segments to point-cloud
        point_wise_y_pred = y_pred.reshape(B, -1, outdim)
        
        # [DEPRECATED]
        # Select first point of each segment in y, as max-overlapping is assumed in gt traj y
        # going from segments to point-cloud is ill-posed. end-of-stroke points would be cut out.
        # point_wise_y = y[:, :, :outdim]

        gt_to_pred_chamfer = 100*chamfer_distance(point_wise_y_pred,
                                                  traj_as_pc,
                                                  padded=True,
                                                  reverse_asymmetric=True)[0]  # reverse asymmetric instead of reverting the first two arguments, because padding only exists in the second argument

        traj_as_pc = traj_as_pc.cpu()

        return gt_to_pred_chamfer


    def get_reverse_asymm_segment_chamfer(self, y_pred, y, **args):
        """Asymmetric segment-wise chamfer, from GT to pred.

            Computes asymmetric chamfer from GT segments to
            predicted segments, but not the oppose.
            Hence, predicted segments will try to stay around all GT
            segments to minimize this loss.

            Note: only a random sub-set of the max-overlapping GT segments
            is considered."""
        B, N_pred, D = y_pred.shape
        B, N_gt, D = y.shape
        # outdim = get_dim_traj_points(self.config['extra_data'])

        gt_to_pred_chamfer = 100*chamfer_distance(y_pred,
                                                  y,
                                                  padded=True,
                                                  reverse_asymmetric=True)[0]  # reverse asymmetric instead of reverting the first two arguments, because padding only exists in the second argument

        return gt_to_pred_chamfer
        

    def get_stoch_reverse_asymm_segment_chamfer(self, y_pred, y, **args):
        """Asymmetric segment-wise chamfer, from GT to pred.

            Computes asymmetric chamfer from GT segments to
            predicted segments, but not the oppose.
            Hence, predicted segments will try to stay around all GT
            segments to minimize this loss.

            Note: only a random sub-set of the max-overlapping GT segments
            is considered."""
        B, N_pred, D = y_pred.shape
        B, N_gt, D = y.shape
        # outdim = get_dim_traj_points(self.config['extra_data'])

        # From segments to point-cloud
        # point_wise_y_pred = y_pred.reshape(B, -1, outdim)
        
        # Select first point of each segment in y, as max-overlapping is assumed in gt traj y
        # point_wise_y = y[:, :, :outdim]

        # print(N_pred, N_gt)

        # idx = np.stack([np.random.choice(N_gt, size=N_pred, replace=False) for _ in range(B)])
        indices = torch.stack([torch.randperm(N_gt)[:N_pred] for _ in range(B)], dim=0).to('cuda')
        # idx = y.multinomial(num_samples=n, replacement=replace)
        # gt_segments = y[idx]
        selected_gt_segments = torch.gather(y, 1, indices.unsqueeze(-1).expand(B, min(N_pred, N_gt), D))

        gt_to_pred_chamfer = 100*chamfer_distance(y_pred,
                                                  selected_gt_segments,
                                                  padded=True,
                                                  reverse_asymmetric=True)[0]  # reverse asymmetric instead of reverting the first two arguments, because padding only exists in the second argument

        return gt_to_pred_chamfer


    def get_repulsion(self, y_pred, y, **args):
        if 'mse' in self.loss:  # Ordered repulsion if MSE is used
            return self.get_ordered_repulsion(y_pred, y, **args)
        elif 'chamfer' in self.loss:
            return self.get_unordered_repulsion(y_pred, y, **args)
        else:  # Fallback to unordered repulsion
            return self.get_unordered_repulsion(y_pred, y, **args)


    def get_unordered_repulsion(self, y_pred, y, **args):
        outdim = get_dim_traj_points(self.config['extra_data'])

        B = y_pred.shape[0]  # Batch size

        if self.config['lambda_points'] > 1:
            # traj_pred = from_seq_to_pc(y_pred.clone(), extra_data=self.config['extra_data'])
            traj_pc = y_pred.view(B, -1, outdim)
        else:
            traj_pc = y_pred

        traj_pc = traj_pc[:, :, :3]

        if self.config['rep_target'] is not None:
            target_dist = self.config['rep_target']
        else:
            y_lengths = None
            if self.config['lambda_points'] > 1:
                ridx, cidx = torch.where(y[:,:,0] == -100)
                y_lengths = []
                for b in range(B):
                    y_lengths.append(cidx[torch.argmax((ridx == b).type(torch.IntTensor))].item())
                y_lengths = torch.tensor(y_lengths, device=y.device)

            target_dist = mean_knn_distance(y[:, :, :3], k=self.config['knn_repulsion'], y_lengths=y_lengths)

        k = self.config['knn_repulsion']
        h = target_dist*np.sqrt(2)
        distances = torch.cdist(traj_pc, traj_pc, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        top_dists, ind = distances.topk(k+1, largest=False, sorted=True, dim=-1)

        top_dists = top_dists[:, :, 1:]  # Remove self-distance
        top_dists = torch.maximum(top_dists, torch.tensor([1e-12]).to(top_dists.device))  # Regularization

        if torch.is_tensor(h) and h.ndim == 1:
            h = h.view(B,1,1)  # For broadcasting

        weight = torch.exp(-(top_dists.square())/(h**2))

        rep = 100*torch.mean(-top_dists*weight)  # Repulsion loss is weighted by 100

        return rep


    def get_ordered_repulsion(self, y_pred, y, **args):
        raise NotImplementedError('If you want to use MSE with repulsion, change the get_repulsion method temporarily.')
        return



    def get_align_loss(self, y_pred, **args):
        # Generate some data that lies along a line

        # x = np.mgrid[-2:5:120j]
        # y = np.mgrid[1:9:120j]
        # z = np.mgrid[-5:3:120j]

        # data = np.concatenate((x[:, np.newaxis], 
        #                        y[:, np.newaxis], 
        #                        z[:, np.newaxis]), 
        #                       axis=1)

        # # Perturb with some Gaussian noise
        # data += np.random.normal(size=data.shape) * 0.4

        # # Calculate the mean of the points, i.e. the 'center' of the cloud
        # datamean = data.mean(axis=0)

        # y_pred_mean = y_pred.mean(axis=1)
        # y_pred_mean = y_pred_mean[:, np.newaxis, :]

        # pdb.set_trace()

        # Do an SVD on the mean-centered data.
        # S = torch.linalg.svdvals(y_pred - y_pred_mean)  # Returns singular values of input matrix
        # Now vv[0] contains the first principal component, i.e. the direction
        # vector of the 'best fit' line in the least squares sense.

        # y = y[:, :, :3]
        y_pred = y_pred[:, :, :3] 

        B = y_pred.shape[0]  # Batch size
        traj_points = y_pred.shape[1]  # Traj_points

        k = self.config['knn_repulsion']
        distances = torch.cdist(y_pred, y_pred, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        top_dists, ind = distances.topk(k+1, largest=False, sorted=True, dim=-1)
        
        # top_dists = top_dists[:, :, 1:]  # Remove self-distance
        # top_dists = torch.maximum(top_dists, torch.tensor([1e-12]).to(top_dists.device))
        # ind = ind[:, :, 1:]

        # tot_unexp_var = 0
        # for b, batch in enumerate(ind): # per batch in indices of top distances
        #     unexplained_variance = 0
        #     for indices in batch:  # per point, consider its k-NNs
        #         # current_point = indices[0]
        #         # nns = indices[1:] # k-NNs indices
        #         data = y_pred[b, indices, :] # considering itself and its k-NNs
        #         datamean = data.mean(axis=0)
        #         S = torch.linalg.svdvals(data - datamean) # singuar values of itself and its k-NNs
        #         unexplained_variance += S[1:].sum()
        #     tot_unexp_var += (unexplained_variance / traj_points)
        # tot_unexp_var /= B

        tot_unexp_var2 = 0
        for b, batch in enumerate(ind):
            unexplained_variance2 = 0

            data = y_pred[b, ind[b, :, :], :]
            datamean = data.mean(axis=-2)
            datamean = datamean[:, None, :]

            S = torch.linalg.svdvals(data - datamean)

            unexplained_variance2 = S[:, 1:].sum(axis=-1)

            tot_unexp_var2 += unexplained_variance2.mean()

        tot_unexp_var2 /= B

        # assert tot_unexp_var == tot_unexp_var2, f'NON ERA UGUALE QUI 1: {tot_unexp_var} ||| 2: {tot_unexp_var2}'
        return tot_unexp_var2



    def get_intra_align(self, y_pred, **args):
        """Encourage sub-sequences to lay on planes

        Fit a plane to points in each sequence,
        and penalizes least-squares to plane.
        """
        B, N_seq, outdim = y_pred.size()
        lmbda = outdim//3

        # tot_unexp_variance = 0
        # for b in range(B):
        #     flatten_data = y_pred[b, :, :].view(-1, 3)  # (traj_points, 3)
        #     slices = torch.arange(0, flatten_data.shape[0]).view((flatten_data.shape[0]//lmbda), lmbda)
        #     data = flatten_data[slices, :]  # (N_seq, lmbda, 3)
        #     datamean = data.mean(axis=-2)
        #     zeromean = (data-datamean[:, None, :])
        #     S = torch.linalg.svdvals(zeromean)
        #     unexplained_variance = S[:, 2]  # Last singular value per sequence
        #     tot_unexp_variance += unexplained_variance.mean()

        flatten_data = y_pred.view(B, -1, 3)  # (B, traj_points, 3)
        slices = torch.arange(0, flatten_data.shape[1]).view((flatten_data.shape[1]//lmbda), lmbda)

        data = flatten_data[:, slices, :]  # (B, N_seq, lmbda, 3)

        datamean = data.mean(axis=-2)
        zeromean = (data-datamean[:, :, None, :])

        S = torch.linalg.svdvals(zeromean)

        unexplained_variance = S[:, :, 2]  # Last singular value

        return unexplained_variance.mean()



    def get_vel_cosine(self, y_pred, **args):
        """Encourage each point's velocity to be close to
        the mean of velocities of k-NNs, in terms of
        cosine similarity"""


        # todo: direct to unordered and ordered, depending on whether chamfer and lambda_points are considered.
        assert 'vel' in self.config['extra_data'], 'Velocity cosine loss cannot be used if velocities are not learned.'

        # input1 = torch.randn(100, 128)
        # input2 = torch.randn(100, 128)
        # cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        # output = cos(input1, input2)

        y_pred_vel = y_pred[:, :, 3:]
        y_pred_pos = y_pred[:, :, :3]

        B = y_pred.shape[0]  # Batch size
        traj_points = y_pred.shape[1]  # Traj_points

        k = self.config['knn_repulsion']
        distances = torch.cdist(y_pred_pos, y_pred_pos, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        top_dists, ind = distances.topk(k+1, largest=False, sorted=True, dim=-1)

        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        tot_cos = 0
        for b, batch in enumerate(ind):
            # current_point = indices[0]
            # nns = indices[1:] # k-NNs indices

            curr_points = ind[b, :, 0] # Indices of curr points
            nns = ind[b, :, 1:]  # Indices of k-NNs

            curr_vels = y_pred_vel[b, curr_points, :] # Vel of curr points
            vel_nns = y_pred_vel[b, nns, :] # Velocities of k-NNs
            mean_vel_nns = vel_nns.mean(axis=-2)  # Mean vel of k-NNs

            tot_cos += cos(curr_vels, mean_vel_nns).mean()
            
        tot_cos /= B

        return -tot_cos


    def get_mse(self, y_pred, y, **args):
        return F.mse_loss(y_pred, y)


    def get_mse_strokes(self, stacked_strokes_pred, stacked_strokes_gt, **kwargs):
        """
            stacked_strokes_pred: (K, stroke_points*outdim)
            stacked_strokes_gt: (K, stroke_points*outdim)
                                K = tot num of strokes, merged across
                                    the batch dimension
        """
        # They should be equal
        # return F.mse_loss(stacked_strokes_pred, stacked_strokes_gt, reduction='sum') / stacked_strokes_pred.shape[0]
        return (stacked_strokes_pred - stacked_strokes_gt).square().sum(-1).mean()


    def get_mse_nexttoken(self, stacked_pred_nexttoken, stacked_gt_nexttoken, **kwargs):
        """Computes MSE between predicted next-token and GT next-token, when stroke rollout task
            is done with autoregressive technique.
            
            stacked_pred_nexttoken: (K, outdim*lambda_points)
            stacked_gt_nexttoken: (K, outdim*lambda_points)
                                  K = tot num of strokes, merged across
                                      the batch dimension
        """
        return (stacked_pred_nexttoken - stacked_gt_nexttoken).square().sum(-1).mean()


    def get_mse_nexttoken_v2(self,
                             stacked_pred_nexttoken,
                             stacked_gt_nexttoken,
                             end_of_path_scores,
                             end_of_path_gt,
                             **kwargs):
        """Computes MSE between predicted next-token and GT next-token, when stroke rollout task
            is done with autoregressive technique.
            
            stacked_pred_nexttoken: (K, outdim*lambda_points)
            stacked_gt_nexttoken: (K, outdim*lambda_points)
                                  K = tot num of history samples, merged across
                                      the batch dimension
            end_of_path_scores: (K,), logits of EoP
            end_of_path_gt: (K,), binary EoP
        """

        """
            MSE among predicted next segments and GT next segments
        """
        mse = 100*((stacked_pred_nexttoken - stacked_gt_nexttoken).square().sum(-1).mean())


        """
            BCE among predicted end-of-path scores and GT end-of-path bools
        """
        # weight non-end-of-path tokens much lower as they are many more.
        weights = torch.ones(end_of_path_gt.shape).float()
        false_counts, true_counts = torch.unique(end_of_path_gt, return_counts=True)[1]
        relative_imbalance = true_counts / false_counts
        weights[end_of_path_gt == 0] = relative_imbalance

        end_of_path_gt = end_of_path_gt.to(end_of_path_scores.get_device()).float()
        weights = weights.to(end_of_path_scores.get_device())
        
        bce = F.binary_cross_entropy_with_logits(end_of_path_scores,
                                                 end_of_path_gt,
                                                 reduction="none",
                                                 weight=weights)
        bce = bce.mean()


        # print(f'Plain losses: [{round(mse.detach().cpu().item(), 2)}] | [{round(bce.detach().cpu().item(), 2)}]')
        # print(f'Adjusted losses: [{round(mse.detach().cpu().item(), 2)}] | [{round(self.config["explicit_weight_endofpath_confidence_loss"]*bce.detach().cpu().item(), 2)}]')
        loss = mse + \
               self.config['explicit_weight_endofpath_confidence_loss']*bce
        
        return loss


    def get_chamfer_strokes(self, stacked_segments_per_stroke_pred, stacked_segments_per_stroke_gt, **kwargs):
        """Compute chamfer between pred segments and GT segments of a stroke specifically.
            Used in the strokeRollout task, where only the segments of a particular
            stroke are predicted.

            Stroke segments are stacked across the batch dimension, therefore treated independently
            of each sample as far as loss computation goes.

            stacked_segments_per_stroke_pred : (tot num of real strokes across mini-batch, out_segments_per_stroke, outdim*lambda)
            stacked_segments_per_stroke_gt : (tot num of real strokes across mini-batch, max_num_of_segments across mini-batch, outdim*lambda), padded over dim=1

            Ideas:
                - give more importance to errors in strokes with fewer segments.
        """
        if not stacked_segments_per_stroke_gt.is_cuda:
            stacked_segments_per_stroke_gt = stacked_segments_per_stroke_gt.to('cuda', dtype=torch.float)

        # GT stroke segments are padded with -100 values
        chamfer = 100*chamfer_distance(stacked_segments_per_stroke_pred,
                                       stacked_segments_per_stroke_gt,
                                       padded=True)[0]

        stacked_segments_per_stroke_gt = stacked_segments_per_stroke_gt.cpu()

        return chamfer


    def get_asymm_v6_chamfer_strokes(self, stacked_segments_per_stroke_pred, stacked_segments_per_stroke_gt, **kwargs):

        return


    def get_masked_mse_strokes(self, stacked_points_per_stroke_pred, stacked_points_per_stroke_gt, confidence_scores, **kwargs):
        """Computes the MSE among predicted (ordered) poses and GT poses, for each stroke separately.
            
            As strokes have different lengths, a confidence probability is also learned for each pred pose,
            telling whether it's exceeding the needed length or not.
        """ 
        if not stacked_points_per_stroke_gt.is_cuda:
            stacked_points_per_stroke_gt = stacked_points_per_stroke_gt.to('cuda', dtype=torch.float)

        B, N_gt, _ = stacked_points_per_stroke_gt.shape
        _, N_pred, _ = stacked_points_per_stroke_pred.shape

        enough_points_needed_pred = stacked_points_per_stroke_pred[:, :N_gt, :]  # we may predict more points than the max number of points in GT strokes (make sure you always predict at least as many though.)

        fake_mask = torch.all((stacked_points_per_stroke_gt == -100), axis=-1)  # find fake points within strokes
        # more_to_be_masked_for_pred = torch.ones((B, N_pred - N_gt), dtype=torch.bool).to('cuda')  # we may predict more points than the max number of points in GT strokes (make sure you always predict at least as many though.)
        # pred_fake_mask = torch.cat((fake_mask, more_to_be_masked_for_pred),dim=-1)


        stacked_points_per_stroke_gt[fake_mask] *= 0  # do not take into account last fake points
        enough_points_needed_pred[fake_mask] *= 0  # do not take into account last fake points predicted

        # stacked_points_per_stroke_gt : (B, N_gt, 6)
        # stacked_points_per_stroke_pred : (B, N_pred, 6)
        mse = (enough_points_needed_pred - stacked_points_per_stroke_gt).square().sum(-1).sum(-1).mean()  # two sums, each stroke is treated as a single vector of all points


        # Confidence scores : 1 for points before the end of the stroke, 0 for points after GT length
        enough_confidence_scores_needed = confidence_scores[:, :N_gt, :]  # (B, N_gt, 1)
        confidence_gt = (~fake_mask).unsqueeze(-1).float()  # (B, N_gt, 1)
        bce = self.bcewithlogits(enough_confidence_scores_needed, confidence_gt).squeeze().sum(-1).mean()  # mean over batch, sum within the same stroke

        tot_loss = bce + mse

        stacked_points_per_stroke_gt = stacked_points_per_stroke_gt.cpu()

        return tot_loss


    def masked_mse_strokes_from_segments(self,
                                         stacked_points_per_stroke_pred,
                                         stacked_points_per_stroke_gt,
                                         confidence_scores,
                                         output_mask):
        # For the sake of compatibility of names
        output_points = stacked_points_per_stroke_pred
        tgt_points = stacked_points_per_stroke_gt
        eos_probs = confidence_scores
        mask = output_mask.unsqueeze(-1).to(torch.device("cuda"))


        point_criterion = nn.MSELoss(reduction='none')  # We'll handle reduction manually
        eos_criterion = WeightedBCELoss(pos_weight=10.0, neg_weight=1.0)  # Increase pos_weight to emphasize EOS

        # Calculate point prediction loss
        point_loss = point_criterion(output_points, tgt_points[:, :, :])
        point_loss = (point_loss * mask[:, :, :]).mean()

        # Generate EOS targets (1 at the last point, 0 otherwise)
        eos_targets = torch.zeros_like(eos_probs)
        eos_targets[torch.arange(eos_targets.size(0)), (mask[:, :, 0].sum(dim=1) - 1).long()] = 1.0

        # Calculate EOS prediction loss
        eos_loss = eos_criterion(eos_probs, eos_targets)
        eos_loss = (eos_loss * mask[:, :, :]).mean()

        # Total loss is a combination of both
        loss = point_loss + eos_loss

        return loss


    def _compute_masked_mse_strokes(self, pred_strokes, target_strokes, outdim):
        """Compute MSE among predicted and GT strokes, by truncating the predicted
            sequence to the correct number of points. Predicted points beyond the length
            in GT do not affect the loss.
        """
        K, N_gt = target_strokes.shape  # padded_max_n_stroke_points_in_gt = N_gt
        K, N_pred = pred_strokes.shape

        enough_points_needed_pred = pred_strokes[:, :N_gt]  # we may predict more points than the max number of points in GT strokes (make sure you always predict at least as many though.)

        # Find padding points in GT. Used for truncating predicted strokes
        fake_points_mask = torch.isclose(target_strokes, torch.tensor(-100).float())

        assert torch.all(torch.isclose(torch.remainder(fake_points_mask.float().sum(-1), outdim), torch.tensor(0).float())), 'sanity check. Make sure that the number of padded points is divisible by the outdim.'

        target_strokes_copy = target_strokes.clone()
        enough_points_needed_pred_copy = enough_points_needed_pred.clone()

        target_strokes_copy[fake_points_mask] = 0.
        enough_points_needed_pred_copy[fake_points_mask] = 0.

        mse = (enough_points_needed_pred_copy - target_strokes_copy).square().sum(-1)  # Tensor of [K,] values

        return mse


    def _compute_point_confidence_loss(self, pred_strokes, target_strokes, pred_point_scores, outdim):
        """Given predicted, GT strokes and predicted point confidence scores, compute a loss that
            pushes to 1 sigmoid(point scores) that match the length of GT strokes, and pushes to 0 point logits beyond
            the length of the corresponding GT stroke.
            
            target_strokes : Tensor of size [K, N_gt * outdim]
            pred_point_scores: Tensor of size [K, N_pred * outdim]
        """
        K, N_gt = target_strokes.shape  # padded_max_n_stroke_points_in_gt = N_gt
        K, N_pred = pred_strokes.shape

        assert N_pred >= N_gt, 'Error: the model is expected to predict more points per stroke than needed, and rely on the learned per-point confidence to figure out the stroke length at inference time.'

        # Add fake points to GT to match the number of predicted points.
        aligned_target_strokes = target_strokes
        if N_pred > N_gt:
            additional_fake_points = torch.ones((K, (N_pred-N_gt)))*(-100)
            aligned_target_strokes = torch.cat((target_strokes, additional_fake_points.to(target_strokes.device)), dim=-1)  # add fake points to match the predicted number of points

        exp_target_strokes = aligned_target_strokes.reshape(K, (N_pred // outdim), outdim)

        # Find fake points in GT. This essentially gives you a mask of effective GT stroke lengths
        fake_points_mask = torch.all(torch.isclose(exp_target_strokes, torch.tensor(-100).float()), dim=2)

        targets = (~fake_points_mask).float()  # (K, (N_pred // outdim))
        bce = self.bcewithlogits(pred_point_scores, targets).sum(-1)  # sum within the same stroke, Tensor of [K,] values

        return bce


    def get_hungarian_SoPs(self, sop_pred, sop_gt, pred_sop_conf_scores, **kwargs):
        """
            Compute loss among predicted Start-of-Path tokens (SoP's)
            and GT ones. More SoPs are predicted than needed, so 
            match predictions to GT with bipartite matching through
            hungarian algorithm.

            sop_pred: Tensor of size [B, config.out_prototypes, config.stroke_prototype_dim]
            sop_gt: Tensor of size [B, config.out_prototypes, config.max_n_strokes]
            pred_sop_conf_scores : Tensor of size [B, config.out_prototypes]
        """
        if not sop_gt.is_cuda:
            sop_gt = sop_gt.to('cuda', dtype=torch.float)

        # Compute bipartite matching among predicted and GT SoPs
        unpadded_sop_gt = [self.remove_padding_from_tensors(sop_gt_b) for sop_gt_b in sop_gt]
        
        indices = self.matcher(outputs=sop_pred, targets=unpadded_sop_gt)  # list of size B, with elements (index_i, index_j). index_* is tensor

        pred_idx = self._get_pred_permutation_idx(indices)  # tuple for indexing predictions
        gt_idx = self._get_gt_permutation_idx(indices)  # tuple for indexing GT

        matched_sop_pred = sop_pred[pred_idx]  # stacked pred segments in order of matching
        matched_sop = sop_gt[gt_idx]  # stacked gt segments in order of matching


        # 1. Compute MSE among matched prototypes
        mse = (matched_sop_pred - matched_sop).square().sum(-1).mean()


        # 2. Compute per-SoP confidence loss
        target_sop_conf_scores = torch.zeros(pred_sop_conf_scores.shape)
        target_sop_conf_scores[pred_idx] = 1.  # for matched SoPs is 1, else is 0
        
        weights = self.config['explicit_no_sop_weight']*torch.ones(pred_sop_conf_scores.shape)  # less weight to predictions of "no SoP", as often times there are many more predicted SoPs than needed
        weights[pred_idx] = 1.

        target_sop_conf_scores = target_sop_conf_scores.to(pred_sop_conf_scores.get_device())
        weights = weights.to(pred_sop_conf_scores.get_device())

        sop_confidence_loss = F.binary_cross_entropy_with_logits(pred_sop_conf_scores, target_sop_conf_scores, reduction="none", weight=weights).mean()  # mean over all predicted SoPs in batch
        
        # print(f'\n\nPlain losses: [{round(mse.detach().cpu().item(), 2)}] | [{round(sop_confidence_loss.detach().cpu().item(), 2)}]')
        # print(f'Adjusted losses: [{round(mse.detach().cpu().item(), 2)}] | [{round(self.config["explicit_weight_sop_confidence_loss"]*sop_confidence_loss.detach().cpu().item(), 2)}]')
        
        loss = mse + \
               self.config['explicit_weight_sop_confidence_loss']*sop_confidence_loss
        

        return loss


    def get_masked_mse_strokes_v2(self, pred_points_per_stroke, points_per_stroke, pred_point_scores, pred_stroke_scores, **kwargs):
        """Used in StrokeWise baseline for computing the MSE among predicted (ordered) poses and GT poses,
            for each stroke separately.
            
            As strokes have different lengths, a confidence probability is also learned for each pred pose,
            telling whether it's exceeding the needed length or not.

            More strokes than needed are also predicted, so the Hungarian matching is used to find the optimal
            match among predicted and GT strokes.

            It's similar to masked_mse_strokes, but that is only used for the strokeRollout task, has no hungarian
            matching, and expects different parameters.
        

            Parameters
            ----------
            pred_points_per_stroke : Tensor of [B, max_n_strokes, max_n_stroke_points*outdim]
                                     predicted points per stroke
            points_per_stroke : list of size B of Tensors [actual_n_strokes, padded_max_n_stroke_points_in_gt, outdim]
                                GT points per stroke
            pred_point_scores : Tensor of [B, max_n_strokes, max_n_stroke_points]
                                confidence scores for each predicted point
            pred_stroke_scores : Tensor of [B, max_n_strokes]
        """ 
        _, _, outdim = points_per_stroke[0].shape

        # Stack stroke points into single vectors
        points_per_stroke = [stroke.reshape(stroke.shape[0], -1) for stroke in points_per_stroke]  # list of size B of Tensors [actual_n_strokes, padded_max_n_stroke_points_in_gt*outdim]
        

        """
            Find hungarian matching between predicted strokes and GT strokes, truncating pred strokes to the correct GT stroke length
        """
        indices = []
        with torch.no_grad():
            for b, (b_pred_points_per_stroke, b_points_per_stroke) in enumerate(zip(pred_points_per_stroke, points_per_stroke)):  # iterate over batch elements
                # Compute cost matrix for this batch
                b_points_per_stroke = b_points_per_stroke.cuda()

                n_gt_strokes = b_points_per_stroke.shape[0]
                n_pred_strokes = b_pred_points_per_stroke.shape[0]

                # all pairs in single-column format for loss computation ([n_pred_strokes, n_gt_strokes] as [n_pred_strokes*n_gt_strokes, 1])
                exp_b_pred_points_per_stroke = b_pred_points_per_stroke.repeat_interleave(n_gt_strokes, dim=0)
                exp_b_target_points_per_stroke = b_points_per_stroke.repeat(n_pred_strokes, 1)

                # bce = F.binary_cross_entropy_with_logits(exp_b_pred_points_per_stroke, exp_b_target_points_per_stroke.float(), reduction="none").sum(-1)
                bce = self._compute_masked_mse_strokes(exp_b_pred_points_per_stroke, exp_b_target_points_per_stroke.float(), outdim=outdim)
                bce = bce.view(n_pred_strokes, n_gt_strokes).cpu()  # cost matrix [n_pred_strokes, n_gt_strokes]
                
                indices.append(linear_sum_assignment(bce))  # solve LAP

        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  # list of size B, with elements (index_i, index_j). index_* is Tensor

        pred_idx = self._get_pred_permutation_idx(indices)  # tuple for indexing pred strokes
        gt_idx = self._get_gt_permutation_idx(indices)  # tuple for indexing target strokes


        # Collect matched predicted strokes 
        matched_pred_strokes = pred_points_per_stroke[pred_idx]  # (stacked) sub-selected pred strokes given optimal matching ([tot num of gt strokes in batch, max_n_stroke_points*outdim])


        # Collect matched GT strokes
        matched_target_strokes = []  # list of size B of selected gt strokes given optimal matching with items [num of gt strokes, padded_max_n_stroke_points_in_gt*outdim])
        curr_stroke_index = 0
        for b, b_points_per_stroke in enumerate(points_per_stroke):
            curr_n_strokes = b_points_per_stroke.shape[0]
            assert torch.all(torch.isclose(gt_idx[0][curr_stroke_index:curr_stroke_index+curr_n_strokes].float(), torch.tensor(b).float())), 'sanity check. gt_idx[0] should contain the belonging id in the batch of each stroke'
            curr_matched_gt_idx = gt_idx[1][curr_stroke_index:curr_stroke_index+curr_n_strokes]
            matched_target_strokes += b_points_per_stroke[curr_matched_gt_idx, :]
            curr_stroke_index += curr_n_strokes
        matched_target_strokes = torch.stack(matched_target_strokes).cuda()  # (stacked) selected gt strokes given optimal matching ([tot num of gt strokes in batch, padded_max_n_stroke_points_in_gt*outdim])


        # 1. Compute masked mse loss among matched strokes
        masked_mse_loss = self._compute_masked_mse_strokes(matched_pred_strokes, matched_target_strokes, outdim=outdim)  # Tensor [tot num of gt strokes in batch]
        masked_mse_loss = masked_mse_loss.mean()  # mean over the number of stacked strokes in batch


        # 2. Compute point confidence loss
        matched_pred_point_scores = pred_point_scores[pred_idx]  # (stacked) loss computed only for matched strokes
        point_confidence_loss = self._compute_point_confidence_loss(matched_pred_strokes, matched_target_strokes, matched_pred_point_scores, outdim=outdim)  # Tensor [tot num of gt strokes in batch]
        point_confidence_loss = point_confidence_loss.mean()  # mean over the number of stacked strokes in batch


        # 3. Compute stroke confidence loss
        target_stroke_scores = torch.zeros(pred_stroke_scores.shape)
        target_stroke_scores[pred_idx] = 1.

        weights = self.config['explicit_no_stroke_weight']*torch.ones(pred_stroke_scores.shape)  # less weight to predictions of "no stroke", as often times there are many more predicted masks than needed
        weights[pred_idx] = 1.

        target_stroke_scores = target_stroke_scores.to(pred_stroke_scores.get_device())
        weights = weights.to(pred_stroke_scores.get_device())

        stroke_confidence_loss = F.binary_cross_entropy_with_logits(pred_stroke_scores, target_stroke_scores, reduction="none", weight=weights).mean()  # mean over all predicted strokes in batch


        loss = self.config['explicit_weight_masked_mse_loss']*masked_mse_loss + \
               self.config['explicit_weight_point_confidence_loss']*point_confidence_loss + \
               self.config['explicit_weight_stroke_confidence_loss']*stroke_confidence_loss
        
        return loss


    def remove_padding_from_tensors(self, tensors):
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


# Custom weighted binary cross-entropy loss for EOS prediction
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, input, target):
        # Compute binary cross-entropy loss with weights
        loss = -self.pos_weight * target * torch.log(input) - self.neg_weight * (1 - target) * torch.log(1 - input)
        return loss.mean()