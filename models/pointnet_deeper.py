"""PointNet with more layers (5) in global feature encoder.

Inspired from the pointnet backbones I used in colorization-partseg
"""
from __future__ import print_function
import pdb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


# class STN3d(nn.Module):
#     def __init__(self):
#         super(STN3d, self).__init__()
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 9)
#         self.relu = nn.ReLU()

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)


#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)

#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, 3, 3)
#         return x


# class STNkd(nn.Module):
#     def __init__(self, k=64):
#         super(STNkd, self).__init__()
#         self.conv1 = torch.nn.Conv1d(k, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k*k)
#         self.relu = nn.ReLU()

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)

#         self.k = k

#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)

#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, affinetrans=True):
        super(PointNetfeat, self).__init__()
        self.affinetrans = affinetrans
        if self.affinetrans:
            self.stn = STN3d()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
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
        x = F.relu(self.bn3(self.conv3(x))) 
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x))) # Or without ReLu

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetRegressor(nn.Module):
    """By Gabriele Tiboni, outputs regressed points for
    spray painting trajectory
    """
    def __init__(self,
                 outdim=3,
                 out_vectors=1500,
                 feature_transform=False,
                 affinetrans=False,
                 hidden_size=(1024, 1024)):
        super(PointNetRegressor, self).__init__()
        self.outdim = outdim
        self.out_vectors = out_vectors
        
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, affinetrans=affinetrans)

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