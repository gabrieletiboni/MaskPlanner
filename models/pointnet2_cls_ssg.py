"""PointNet++ classification network (ssg: single-scale grouping)
https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/log/classification/pointnet2_ssg_wo_normals
"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction


class PointNet2Regressor(nn.Module):
    def __init__(self,
                 outdim=3,
                 outdim_orient=3,
                 weight_orient=1.,
                 normal_channel=False,
                 out_vectors=1500,
                 hidden_size=(1024, 1024),
                 inputdim=None
                 ):
        super(PointNet2Regressor, self).__init__()
        """
        outdim: translational dims of each output vector
        outdim_orient: orientation dims of each output vector
        out_vectors: number of output vectors
        """
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



class PointNet2Regressor_SoPs(nn.Module):
    """PointNet++ as regressor for start-of-path tokens
       (concatenation of poses)
    """
    def __init__(self,
                 out_vectors=10,
                 outdim=3,
                 outdim_orient=3,
                 weight_orient=1.,
                 normal_channel=False,
                 hidden_size=(1024, 1024),
                 inputdim=None,
                 sop_confidence_scores=False
                 ):
        super(PointNet2Regressor_SoPs, self).__init__()
        """
            out_vectors: number of output vectors / prototypes
            outdim: translational dims of each output vector
            outdim_orient: orientation dims of each output vector
            sop_confidence_scores: bool,
                                   if True, learn a per-sop confidence score
        """
        in_channel = 6 if normal_channel else 3
        if inputdim is not None:
            in_channel = inputdim
        self.normal_channel = normal_channel

        self.outdim = outdim
        self.outdim_orient = outdim_orient
        self.out_vectors = out_vectors
        self.weight_orient = weight_orient
        self.sop_confidence_scores = sop_confidence_scores

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], out_vectors*outdim)

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        if outdim_orient > 0:
            self.fc_normals = nn.Linear(hidden_size[1], out_vectors*outdim_orient)
            self.tanh = nn.Tanh()

        if self.sop_confidence_scores:
            self.sop_conf_out = torch.nn.Linear(hidden_size[1], out_vectors)


    def forward(self, xyz, return_object_features=False):
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
        last = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(last)

        sop_conf_scores = None
        if self.sop_confidence_scores:
            sop_conf_scores = self.sop_conf_out(last)  # [B, out_prototypes]

        if self.outdim_orient > 0:  # Output normal for each point
            normals = self.tanh(self.fc_normals(last))
            normals = normals.view(B, -1, 3)
            normals = F.normalize(normals, dim=-1)
            normals *= self.weight_orient
            x = x.view(B, -1, 3)
            out = torch.cat((x, normals), dim=-1)
            out = out.view(B, self.out_vectors, -1)
        else:
            out = x.view(B, self.out_vectors, self.outdim)
        
        if return_object_features:
            return out, sop_conf_scores, global_feat
        else:
            return out, sop_conf_scores


class PointNet2Regressor_3Dbbox(nn.Module):
    """PointNet++ as regressor for 3D bounding boxes"""
    def __init__(self,
                 out_bboxes=10,
                 normal_channel=False,
                 hidden_size=(1024, 1024),
                 inputdim=None
                 ):
        super(PointNet2Regressor_3Dbbox, self).__init__()
        """
            out_bboxes: num of predicted 3D bounding boxes
        """
        in_channel = 6 if normal_channel else 3
        if inputdim is not None:
            in_channel = inputdim
        self.normal_channel = normal_channel

        self.out_bboxes = out_bboxes
        self.outdim = 6  # 3D bounding boxes following (non-rotated) as (x,y,z) center + (w,h,d) size

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], self.out_bboxes*self.outdim)

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
        last = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(last)

        out = x.view(B, self.out_bboxes, self.outdim)
        
        return out



class PointNet2Regressor_StrokeMasks(nn.Module):
    def __init__(self,
                 outdim=3,
                 outdim_orient=3,
                 weight_orient=1.,
                 normal_channel=False,
                 out_vectors=1500,
                 hidden_size=(1024, 1024),
                 inputdim=None,
                 pred_stroke_masks=False,
                 n_stroke_masks=None,
                 mask_confidence_scores=False,
                 segment_confidence_scores=False
                 ):
        super(PointNet2Regressor_StrokeMasks, self).__init__()
        """
        outdim: translational dims of each output vector
        outdim_orient: orientation dims of each output vector
        out_vectors: number of output vectors
        """
        self.outdim = outdim
        self.outdim_orient = outdim_orient
        self.out_vectors = out_vectors
        self.weight_orient = weight_orient
        self.pred_stroke_masks = pred_stroke_masks
        self.n_stroke_masks = n_stroke_masks
        self.mask_confidence_scores = mask_confidence_scores
        self.segment_confidence_scores = segment_confidence_scores

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
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        if outdim_orient > 0:
            self.fc_normals = nn.Linear(hidden_size[1], out_vectors*outdim_orient)
            self.tanh = nn.Tanh()

        if self.segment_confidence_scores:
            self.seg_conf_fc1 = nn.Linear(1024, hidden_size[0])
            self.seg_conf_fc2 = nn.Linear(hidden_size[0], hidden_size[1])
            self.seg_conf_out = nn.Linear(hidden_size[1], out_vectors*1)

        if self.pred_stroke_masks:
            self.sm_fc1 = nn.Linear(1024, hidden_size[0])
            self.sm_fc2 = nn.Linear(hidden_size[0], hidden_size[1])
            self.sm_fc3 = nn.Linear(hidden_size[1], out_vectors*1*self.n_stroke_masks)

            self.sm_bn1 = nn.BatchNorm1d(hidden_size[0])
            self.sm_bn2 = nn.BatchNorm1d(hidden_size[1])

            if self.mask_confidence_scores:
                self.mask_conf_out = torch.nn.Linear(hidden_size[1], self.n_stroke_masks)

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

        seg_conf_logits = None
        if self.segment_confidence_scores:
            segconf1 = self.dropout(F.relu(self.seg_conf_fc1(global_feat)))
            segconf2 = self.dropout(F.relu(self.seg_conf_fc2(segconf1)))
            seg_conf_scores = self.seg_conf_out(segconf2)
            seg_conf_logits = torch.sigmoid(seg_conf_scores)

        sm_out, mask_conf_scores = None, None
        if self.pred_stroke_masks:
            sm_1 = self.dropout(F.relu(self.sm_bn1(self.sm_fc1(global_feat))))
            sm_2 = self.dropout(F.relu(self.sm_bn2(self.sm_fc2(sm_1))))
            sm_out = self.sm_fc3(sm_2)
            sm_out = sm_out.view(B, self.n_stroke_masks, -1)

            if self.mask_confidence_scores:
                mask_conf_scores = self.mask_conf_out(sm_2)  # [B, self.n_stroke_masks]


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
        

        return out, sm_out, mask_conf_scores, seg_conf_logits



class PointNet2Regressor_StrokeMasks_RetroCompatible(nn.Module):
    def __init__(self,
                 outdim=3,
                 outdim_orient=3,
                 weight_orient=1.,
                 normal_channel=False,
                 out_vectors=1500,
                 hidden_size=(1024, 1024),
                 inputdim=None,
                 pred_stroke_masks=False,
                 n_stroke_masks=None,
                 mask_confidence_scores=False,
                 segment_confidence_scores=False
                 ):
        super(PointNet2Regressor_StrokeMasks_RetroCompatible, self).__init__()
        """
        outdim: translational dims of each output vector
        outdim_orient: orientation dims of each output vector
        out_vectors: number of output vectors
        """
        self.outdim = outdim
        self.outdim_orient = outdim_orient
        self.out_vectors = out_vectors
        self.weight_orient = weight_orient
        self.pred_stroke_masks = pred_stroke_masks
        self.n_stroke_masks = n_stroke_masks
        self.mask_confidence_scores = mask_confidence_scores
        self.segment_confidence_scores = segment_confidence_scores

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
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        if outdim_orient > 0:
            self.fc_normals = nn.Linear(hidden_size[1], out_vectors*outdim_orient)
            self.tanh = nn.Tanh()

        if self.segment_confidence_scores:
            self.seg_conf_fc1 = nn.Linear(1024, hidden_size[0])
            self.seg_conf_fc2 = nn.Linear(hidden_size[0], hidden_size[1])
            self.seg_conf_out = nn.Linear(hidden_size[1], out_vectors*1)

        if self.pred_stroke_masks:
            self.sm_fc1 = nn.Linear(1024, hidden_size[0])
            self.sm_fc2 = nn.Linear(hidden_size[0], hidden_size[1])
            self.sm_fc3 = nn.Linear(hidden_size[1], out_vectors*1*self.n_stroke_masks)

            self.sm_bn1 = nn.BatchNorm1d(hidden_size[0])
            self.sm_bn2 = nn.BatchNorm1d(hidden_size[1])

            if self.mask_confidence_scores:
                self.out_confidence = torch.nn.Linear(hidden_size[1], self.n_stroke_masks)

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

        seg_conf_logits = None
        if self.segment_confidence_scores:
            segconf1 = self.dropout(F.relu(self.seg_conf_fc1(global_feat)))
            segconf2 = self.dropout(F.relu(self.seg_conf_fc2(segconf1)))
            seg_conf_scores = self.seg_conf_out(segconf2)
            seg_conf_logits = torch.sigmoid(seg_conf_scores)

        sm_out, mask_conf_scores = None, None
        if self.pred_stroke_masks:
            sm_1 = self.dropout(F.relu(self.sm_bn1(self.sm_fc1(global_feat))))
            sm_2 = self.dropout(F.relu(self.sm_bn2(self.sm_fc2(sm_1))))
            sm_out = self.sm_fc3(sm_2)
            sm_out = sm_out.view(B, self.n_stroke_masks, -1)

            if self.mask_confidence_scores:
                mask_conf_scores = self.out_confidence(sm_2)  # [B, self.n_stroke_masks]


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
        

        return out, sm_out, mask_conf_scores, seg_conf_logits



class PointNet2Regressor_StrokeWise(nn.Module):
    def __init__(self,
                 outdim=3,
                 outdim_orient=3,
                 weight_orient=1.,
                 normal_channel=False,
                 out_vectors=1500,
                 hidden_size=(1024, 1024),
                 inputdim=None,
                 stroke_confidence_scores=False,
                 point_confidence_scores=False,
                 n_points_per_out_vector=None
                 ):
        super(PointNet2Regressor_StrokeWise, self).__init__()
        """
        outdim: translational dims of each output vector (i.e. stroke). I.e., number of points per stroke * 3
        outdim_orient: orientation dims of each output vector (i.e. stroke). I.e., number of points per stroke * 3
        out_vectors: number of output vectors (i.e. strokes)

        stroke_confidence_scores:
        point_confidence_scores:
        n_points_per_out_vector: 
        """
        self.outdim = outdim
        self.outdim_orient = outdim_orient
        self.out_vectors = out_vectors
        self.weight_orient = weight_orient
        self.stroke_confidence_scores = stroke_confidence_scores
        self.point_confidence_scores = point_confidence_scores
        self.n_points_per_out_vector = n_points_per_out_vector

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
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        if outdim_orient > 0:
            self.fc_normals = nn.Linear(hidden_size[1], out_vectors*outdim_orient)
            self.tanh = nn.Tanh()

        if self.stroke_confidence_scores:
            self.stroke_conf_out = torch.nn.Linear(hidden_size[1], out_vectors)

        if self.point_confidence_scores:
            self.point_conf_out = torch.nn.Linear(hidden_size[1], out_vectors*n_points_per_out_vector)

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

        stroke_conf_scores = None
        if self.stroke_confidence_scores:
            stroke_conf_scores = self.stroke_conf_out(final)  # [B, self.n_stroke_masks]

        if self.point_confidence_scores:
            point_conf_scores = self.point_conf_out(final)
            # point_conf_logits = torch.sigmoid(point_conf_scores)
            point_conf_scores = point_conf_scores.view(B, self.out_vectors, self.n_points_per_out_vector)

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
        

        return out, point_conf_scores, stroke_conf_scores