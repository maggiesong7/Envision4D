from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, rearrange
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .decoder import Decoder, DecoderOutput

from gsplat.rendering import rasterization


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        mode: str,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.mode = mode
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        image_shape: tuple[int, int],
        global_step: int,
    ) -> DecoderOutput:
        h, w = image_shape
        b, v, _, _ = extrinsics.shape
        
        batch_colors = []
        batch_depth_maps = []
        for b_idx in range(b):
            view_colors = []
            view_depth_maps = []
            for tv_idx in range(v):
                means_list = []
                quats_list = []
                opacities_list = []
                scales_list = []
                colors_list = []
                for sc_idx in range(gaussians.means.shape[1]):
                    if (tv_idx - sc_idx) == 0:
                        continue
                    t_idx = tv_idx if tv_idx < sc_idx else tv_idx - 1
                    means_list.append(gaussians.means[b_idx, sc_idx] + gaussians.motions[b_idx, sc_idx, t_idx])

                    quats_list.append(gaussians.rotations[b_idx, sc_idx])
                    scales_list.append(gaussians.scales[b_idx, sc_idx])
                    opacities_list.append(gaussians.opacities[b_idx, sc_idx])
                    colors_list.append(gaussians.colors[b_idx, sc_idx])
                means = torch.stack(means_list, dim=0)
                quats = torch.stack(quats_list, dim=0)
                scales = torch.stack(scales_list, dim=0)
                opacities = torch.stack(opacities_list, dim=0)
                gcolors = torch.stack(colors_list, dim=0)
                del means_list, quats_list, opacities_list, scales_list, colors_list
                
                render_colors, render_alphas, _ = rasterization(
                    means=rearrange(means, "v g ... -> (v g) ..."),
                    quats=rearrange(quats, "v g ... -> (v g) ..."),
                    scales=rearrange(scales, "v g ... -> (v g) ..."),
                    opacities=rearrange(opacities, "v g ... -> (v g) ..."),
                    colors=rearrange(gcolors, "v g ... -> (v g) ..."),
                    viewmats=extrinsics[b_idx:b_idx+1, tv_idx],
                    Ks=intrinsics[b_idx:b_idx+1, tv_idx],
                    width=w,
                    height=h,
                    render_mode='RGB+ED',  
                )
                    
                view_colors.append(render_colors[..., :-1])
                view_depth_maps.append(render_colors[..., -1])
            
            batch_colors.append(rearrange(torch.cat(view_colors, dim=0), "v h w c -> v c h w"))
            batch_depth_maps.append(torch.cat(view_depth_maps, dim=0))

        colors = torch.stack(batch_colors, dim=0)    
        depth_maps = torch.stack(batch_depth_maps, dim=0)

        return DecoderOutput(
            colors,
            depth_maps,
        )