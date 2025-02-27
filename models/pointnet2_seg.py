"""PointNet++ segmentation networks

    Inspired by:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class PointNet2Segmenter_v1(nn.Module):
    """v1: inspired by PointNet++ classifier, simply concatenates
        the latest set abstraction features with the input points,
        then predicts per-point scores.

        inspired by PointNet++ for classification:
        https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/log/classification/pointnet2_ssg_wo_normals
    """
    def __init__(self,
                 outdim=2,
                 input_orient_dim=0,
                 lambda_points=1,
                 ball_in_xyz_space=False
                 ):
        super(PointNet2Segmenter_v1, self).__init__()
        """
            outdim: e.g. latent_dim for clustering with contrastive loss
            ball_in_xyz_space: bool
                               compute fps and neighbours within ball in R^3 space instead of R^(outdim*lambda_points),
                               hence use the segments centroids (R^3)
            input_orient_dim: int
                              number of dims dedicated to orientations encoding for input poses
        """
        self.ball_in_xyz_space = ball_in_xyz_space
        self.lambda_points = lambda_points
        self.input_orient_dim = input_orient_dim
        self.outdim = outdim
        self.in_channel = (3 + self.input_orient_dim)*self.lambda_points
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=self.in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.conv1 = torch.nn.Conv1d(1024 + self.in_channel, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.outdim, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, input_set, **kwargs):
        B, D, N = input_set.shape

        if self.ball_in_xyz_space:
            centroids = input_set.unsqueeze(-1)
            centroids = centroids.reshape(B, N, self.lambda_points, self.in_channel//self.lambda_points)[:, :, :, :3]
            xyz = centroids.mean(axis=-2).permute(0, 2, 1)
            full_points = input_set
        else:
            xyz = input_set
            full_points = None

        l1_xyz, l1_points = self.sa1(xyz, points=None, full_points=full_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        global_feat = l3_points.view(B, 1024)
        x = global_feat.view(B, 1024, 1).repeat(1, 1, N)
        x = torch.cat([x, input_set], 1)  # cat input segments with global features
        # x = self.dropout(F.relu(self.bn1(self.fc1(global_feat))))
        # final = self.dropout(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(final)

        # if self.outdim_orient > 0:  # Output normal for each point
        #     normals = self.tanh(self.fc_normals(final))
        #     normals = normals.view(B, -1, 3)
        #     normals = F.normalize(normals, dim=-1)
        #     normals *= self.weight_orient
        #     x = x.view(B, -1, 3)
        #     out = torch.cat((x, normals), dim=-1)
        #     out = out.view(B, self.out_vectors, -1)
        # else:
        #     out = x.view(B, self.out_vectors, self.outdim)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.permute(0, 2, 1)  # (batchsize, n_pts, latent_dim)
        
        return x  # out


class PointNet2Segmenter_v2(nn.Module):
    """v2: towards the original paper implementation,
        but bypasses the featurePropagation layers by
        always sampling all points as centroids in all set abstraction
        layers. This way, we always have sets of the same number of input points. 
    """
    def __init__(self,
                 outdim=3,
                 outdim_orient=3,
                 weight_orient=1.,
                 normal_channel=False,
                 out_vectors=1500,
                 hidden_size=(1024, 1024),
                 inputdim=None
                 ):
        super(PointNet2Segmenter_v2, self).__init__()
        """
        outdim: translational dims of each output vector
        outdim_orient: orientation dims of each output vector
        out_vectors: number of output vectors
        """
        raise NotImplementedError('TODO: SetAbstraction with sample_all_as_centroids=True flag')

        self.outdim = outdim
        self.outdim_orient = outdim_orient
        self.out_vectors = out_vectors
        self.weight_orient = weight_orient

        in_channel = 6 if normal_channel else 3
        if inputdim is not None:
            in_channel = inputdim
        self.normal_channel = normal_channel



        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], out_vectors*outdim)

        if outdim_orient > 0:
            self.fc_normals = nn.Linear(hidden_size[1], out_vectors*outdim_orient)
            self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        global_feat = l3_points.view(B, 1024)
        x = self.dropout(F.relu(self.bn1(self.fc1(global_feat))))
        final = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(final)

        if self.outdim_orient > 0:  # Output normal for each point
            normals = self.tanh(self.fc_normals(final))
            normals = normals.view(B, -1, 3)
            normals = F.normalize(normals, dim=-1)
            normals *= self.weight_orient
            x = x.view(B, -1, 3)
            out = torch.cat((x, normals), dim=-1)
            out = out.view(B, self.out_vectors, -1)
        else:
            out = x.view(B, self.out_vectors, self.outdim)
        
        return out


class PointNet2Segmenter_v3(nn.Module):
    """v3: PointNet++ with single-scale grouping (SSG) layers
        and featurePropagation to do segmentation as proposed in the original paper.

        Inspired by: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/log/sem_seg/pointnet2_sem_seg/pointnet2_sem_seg.py
    """
    def __init__(self,
                 outdim=2,
                 inputdim=None
                ):
        super(PointNet2Segmenter_v3, self).__init__()

        raise NotImplementedError('TODO')

        self.outdim = outdim
        in_channel = 3
        if inputdim is not None:
            in_channel = inputdim

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, in_channel, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(128, self.outdim, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points)), inplace=True))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)

        x = x.view(batchsize, n_pts, self.outdim)

        return x  # , l4_points


class PointNet2Segmenter_v4(nn.Module):
    """v4: PointNet++ with multi-scale grouping (MSG) layers
        and featurePropagation to do segmentation as proposed in the original paper.

        See https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/log/part_seg/pointnet2_part_seg_msg/pointnet2_part_seg_msg.py
        for implementing this.
    """
    def __init__(self,
                 outdim=2,
                 inputdim=None
                ):
        super(PointNet2Segmenter_v4, self).__init__()

        raise NotImplementedError('TODO')

    def forward(self, xyz):
        return


class PointNet2Segmenter_PaintNet_v1(nn.Module):
    """v1: inspired by PointNet++ classifier, simply concatenates
        the latest set abstraction features with the input points,
        then predicts per-point scores.

        inspired by PointNet++ for classification:
        https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/log/classification/pointnet2_ssg_wo_normals
    """
    def __init__(self,
                 inputdim=3,
                 outdim_trasl=3,
                 outdim_orient=3,
                 weight_orient=1.,
                 lambda_points=1
                 ):
        super(PointNet2Segmenter_PaintNet_v1, self).__init__()
        """

        """
        self.lambda_points = lambda_points
        self.outdim_trasl = outdim_trasl
        self.outdim_orient = outdim_orient
        self.weight_orient = weight_orient

        self.in_channel = inputdim

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=self.in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.conv1 = torch.nn.Conv1d(1024 + self.in_channel, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)

        self.conv4_trasl = torch.nn.Conv1d(128, self.outdim_trasl*self.lambda_points, 1)

        if self.outdim_orient > 0:
            self.conv4_orient = torch.nn.Conv1d(128, self.outdim_orient*self.lambda_points, 1)
            self.tanh = nn.Tanh()

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, input_set, **kwargs):
        B, D, N = input_set.shape

        xyz = input_set
        full_points = None

        l1_xyz, l1_points = self.sa1(xyz, points=None, full_points=full_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        global_feat = l3_points.view(B, 1024)
        x = global_feat.view(B, 1024, 1).repeat(1, 1, N)
        x = torch.cat([x, input_set], 1)  # cat input segments with global features

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        last = F.relu(self.bn3(self.conv3(x)))

        x = self.conv4_trasl(last)
        x = x.permute(0, 2, 1)  # (batchsize, n_pts, outdim_trasl*lambda_points)

        if self.outdim_orient > 0:
            normals = self.tanh(self.conv4_orient(last))
            normals = normals.permute(0, 2, 1)  # (batchsize, n_pts, outdim_orient*lambda_points)
            normals = normals.view(B, N, self.lambda_points, -1)
            normals = F.normalize(normals, dim=-1)

            normals *= self.weight_orient

            x = x.view(B, N, self.lambda_points, -1)

            out = torch.cat((x, normals), dim=-1)
            out = out.view(B, N, -1)

        else:
            raise NotImplementedError()
        
        return out