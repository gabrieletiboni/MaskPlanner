"""Fxia22 implementation of PointNet
https://github.com/fxia22/pointnet.pytorch
"""
from __future__ import print_function
import pdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, affinetrans=True, in_channel=3):
        super(PointNetfeat, self).__init__()
        self.affinetrans = affinetrans
        if self.affinetrans:
            self.stn = STN3d()

        self.conv1 = torch.nn.Conv1d(in_channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]

        if self.affinetrans:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            # Return global features only, i.e. after torch.max (e.g. for classification)
            return x, trans, trans_feat
        else:
            # Return per-point features (e.g. for segmentation):
            # i.e. concatenation of per-point embeddings + global_features + features at first level (pointfeat)
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetRegressor(nn.Module):
    """By Gabriele Tiboni, outputs regressed points for
    spray painting trajectory
    """
    def __init__(self,
                 out_vectors=1500,
                 outdim=3,
                 feature_transform=False,
                 affinetrans=False,
                 hidden_size=(1024, 1024),
                 inputdim=None
                ):
        super(PointNetRegressor, self).__init__()
        self.outdim = outdim
        self.out_vectors = out_vectors

        in_channel = 3
        if inputdim is not None:
            in_channel = inputdim

        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, affinetrans=affinetrans, in_channel=in_channel)

        self.fc1 = nn.Linear(1024, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], out_vectors*outdim)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        self.relu = nn.ReLU()
        pass

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans, trans_feat = self.feat(x)
        if batchsize == 1:  # No batchnorm
            x = F.relu(self.fc1(x))
            x = F.relu(self.dropout(self.fc2(x)))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = x.view(batchsize, self.out_vectors, self.outdim)
        return x  #, trans, trans_feat


class PointNetSegmenter(nn.Module):
    """Custom segmentation network by Gabriele Tiboni.

    Inspired by PointNetDenseCls, i.e. the original
    used in https://github.com/fxia22/pointnet.pytorch/
    for point-cloud segmentation
    """

    def __init__(self,
                 outdim=2,
                 feature_transform=False,
                 affinetrans=False,
                 inputdim=None,
                 augment_point_features_by=0
                ):
        super(PointNetSegmenter, self).__init__()
        self.outdim = outdim
        self.feature_transform = feature_transform

        in_channel = 3
        if inputdim is not None:
            in_channel = inputdim

        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform, affinetrans=affinetrans, in_channel=in_channel)

        self.conv1 = torch.nn.Conv1d(1088+augment_point_features_by, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.outdim, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x, **kwargs):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)

        if 'one_hot_encoding_sample' in kwargs and kwargs['one_hot_encoding_sample'] is not None:
            feature_dim = x.size()[1]
            num_of_one_hot_classes = kwargs['one_hot_encoding_sample'].size()[1]

            one_hot = kwargs['one_hot_encoding_sample'].view(batchsize, num_of_one_hot_classes, 1)  # unsqueeze the n_pts dimension
            one_hot = one_hot.repeat(1, 1, n_pts)  # repeat same one-hot for all points

            x = torch.cat((x, one_hot), dim=1)  # concatenate on the point features dimension

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.permute(0, 2, 1)  # (batchsize, n_pts, latent_dim)
        return x  #, trans, trans_feat


class PointNetSegmenterConv1d(nn.Module):
    """Custom segmentation network by Gabriele Tiboni.
        
        Point-wise segmentation that has no information propagation
        across points of the input point-cloud (no max operation.
        Also, only the normals may be optionally considered.
    """

    def __init__(self,
                 outdim=2,
                 lambda_points=1,
                 input_normals_only=False
                ):
        super(PointNetSegmenterConv1d, self).__init__()
        self.outdim = outdim
        self.lambda_points = lambda_points
        self.input_normals_only = input_normals_only

        in_channel = 6
        if self.input_normals_only:
            in_channel = 3

        self.conv1 = torch.nn.Conv1d(in_channel*self.lambda_points, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, self.outdim, 1)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x, **kwargs):        
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        if self.input_normals_only:
            # Select point normals only
            subindexes = []
            for l in range(self.lambda_points):
                subindexes += [l*6 + 3 + i for i in range(3)]  # orientation
            x = x[:, subindexes, :]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        x = x.permute(0, 2, 1)  # (batchsize, n_pts, latent_dim)
        return x  #, trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())