# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from .aggregator import Aggregator
from ..heads.camera_head import CameraHead
from ..heads.dpt_head import DPTHead
from ..heads.track_head import TrackHead

from ..layers.block import Block
from ..layers.rope import RotaryPositionEmbedding2D, PositionGetter


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, gaussian_dim=12, num_frames=2, enable_gs=True,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, enable_extra=False):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        
        self.gs_head = DPTHead(dim_in=2 * embed_dim, output_dim=gaussian_dim, activation="sigmoid", enable_gs=True) if enable_gs else None
        self.motion_head = DPTHead(dim_in=2 * embed_dim, output_dim=3, activation="sigmoid", enable_motion=True) if enable_gs else None
        
        extra_depth = 2
        self.enable_extra = enable_extra
        self.extra_token_offset = nn.Parameter(torch.randn(1, num_frames-2, 2*embed_dim))
        self.extra_blocks = nn.Sequential(
            *[
                Block(dim=2*embed_dim, num_heads=16, mlp_ratio=4.0, init_values=0.01)
                for _ in range(extra_depth)
            ]
        )
        
        self.time_embedding = nn.Sequential(
            nn.Linear(256, 2*embed_dim), nn.SiLU(), nn.Linear(2*embed_dim, 2*embed_dim))

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, time_list, patch_start_idx, images = self.aggregator(images)
        
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                
                if self.enable_extra:
                    # Extract known camera tokens
                    tokens = aggregated_tokens_list[-1]
                    pose_tokens = tokens[:, :, 0]
                    v = pose_tokens.shape[1]
                    
                    # Apply offset on pose token from the last timestamp 
                    extra_tokens = pose_tokens[:, -1:] + self.extra_token_offset
                    
                    all_tokens = torch.cat([pose_tokens, extra_tokens], dim=1)
                    # Add time embedding to all tokens
                    timestep = torch.arange(all_tokens.shape[1], device=all_tokens.device, dtype=all_tokens.dtype)
                    time_embedding = repeat(sinusoidal_embedding_1d(256, timestep), "v c -> b v c", b=all_tokens.shape[0])
                    time_embedding = self.time_embedding(time_embedding)
                    all_tokens = torch.stack([all_tokens, time_embedding], dim=2).view(all_tokens.shape[0], -1, all_tokens.shape[-1])
                    for _ in range(4):
                        all_tokens = self.extra_blocks(all_tokens)
                    
                    all_tokens = rearrange(all_tokens, "b (v n) c -> b v n c", n=2)
                    time_embedding = all_tokens[:, :, 1]
                    extra_enc_list = self.camera_head(aggregated_tokens_list, all_tokens[:, :, 0])
                    pose_enc = extra_enc_list[-1]
                else:
                    pose_enc_list = self.camera_head(aggregated_tokens_list)
                    pose_enc = pose_enc_list[-1]  # pose encoding of the last iteration
                    
                predictions["pose_enc"] = pose_enc
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
                
            if self.gs_head is not None:
                gaussian = self.gs_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["gaussians"] = gaussian
                
            if self.motion_head is not None:
                motion_tokens_list = []
                for time_tokens in time_list:
                    x = time_tokens[:, :, patch_start_idx:]
                    indices = torch.stack([torch.cat([torch.arange(i), torch.arange(i+1, time_embedding.shape[1])]) for i in range(x.shape[1])])  # [v1, v-1]
                    timeline = repeat(time_embedding[:, indices, :], "b v1 v2 c -> b v1 v2 g c", g=x.shape[2]) 
                    timelift = (x.unsqueeze(2) * timeline).view(x.shape[0], -1, x.shape[2], x.shape[3])
                    motion_tokens_list.append(timelift)
                
                motion_enc = self.motion_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, motion_tokens_list=motion_tokens_list
                )
                predictions["motion_enc"] = motion_enc

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

