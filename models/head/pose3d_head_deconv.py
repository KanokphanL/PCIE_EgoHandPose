# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn
from .rtmcc_block import RTMCCBlock, ScaleNorm

OptIntSeq = Optional[Sequence[int]]

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Pose3DHead_deconv(nn.Module):

    def __init__(
        self,
        height_dim: int,
        width_dim: int,
        embed_dims:int,
        num_joints: int,
        final_layer_kernel_size: int = 7,

    ):
        super(Pose3DHead_deconv,self).__init__()
        self.height_dim =height_dim
        self.width_dim = width_dim
        self.embed_dims = embed_dims
        self.num_joints =num_joints
        self.final_layer_kernel_size = final_layer_kernel_size
        self.depth_dim = self.width_dim 
        ######### 2D pose head #########
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dims, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.final_layer = nn.Conv2d(
            256, self.num_joints * self.depth_dim * 8, kernel_size=1, stride=1, padding=0
        )
        self.pose_layer = nn.Sequential(
            nn.Conv3d(self.num_joints, self.num_joints, 1),
            nn.GELU(),
            nn.GroupNorm(self.num_joints, self.num_joints),
            nn.Conv3d(self.num_joints, self.num_joints, 1),
        )

        ######### 3D pose head #########
        self.pose_3d_head = nn.Sequential(
            # nn.Linear(self.depth_dim * 3, 512),
            nn.Linear(self.depth_dim * 8 + self.height_dim * 8 + self.width_dim * 8 , 512),
            nn.ReLU(),
            nn.GroupNorm(self.num_joints, self.num_joints),
            nn.Linear(512, 3),
        )

    def forward(self, feats: Tensor): #  convNeXt [N, 1024, 7, 7]
        # Intermediate layer
            out = self.up_sample(feats)  # [N, 256, H_feat *8, W_feat*8]
            out = self.final_layer(out)  # [N, num_joints*D_feat*8, H_feat*8, W_feat*8]
            out = self.pose_layer(
                out.reshape(
                    out.shape[0],
                    self.num_joints,
                    self.depth_dim*8,
                    out.shape[2],
                    out.shape[3],
                )
            )  # (N, num_joints, D_feat*8, H_feat*8, W_feat*8)

            # 3D pose head
            hm_x0 = out.sum((2, 3))
            hm_y0 = out.sum((2, 4))
            hm_z0 = out.sum((3, 4))
            pose_3d_pred = torch.cat((hm_x0, hm_y0, hm_z0), dim=2)
            pose_3d_pred = self.pose_3d_head(pose_3d_pred)
            return pose_3d_pred