from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from einops import rearrange, repeat

from ...dataset import DatasetCfg
from ..types import Gaussians

from .envision4d.models.vggt import VGGT
from .envision4d.utils.pose_enc import pose_encoding_to_extri_intri
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder, EncoderOutput
from .visualization.encoder_visualizer_envision4d_cfg import EncoderVisualizerEnvi4DCfg

from ...global_cfg import get_cfg


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderEnvi4DCfg:
    name: Literal["envision4d"]
    num_surfaces: int
    visualizer: EncoderVisualizerEnvi4DCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    downscale_factor: int
    shim_patch_size: int
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


class EncoderEnvi4D(Encoder[EncoderEnvi4DCfg]):
    backbone: VGGT

    def __init__(self, cfg: EncoderEnvi4DCfg, dataset_cfg: DatasetCfg) -> None:
        super().__init__(cfg, dataset_cfg)
        
        enable_extra = True
        self.enable_extra = enable_extra

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # Gaussian Num Channels = 
            # Opacity (1) + 
            # Scales (3) +
            # Rotations (4) +
            # Colors (3)
        gaussian_dim = 1 + 3 + 4 + 3 
        
        # multi-view Transformer backbone
        self.backbone = VGGT(
            gaussian_dim=gaussian_dim,
            enable_gs=True, 
            enable_camera=True, 
            enable_point=False, 
            enable_depth=True, 
            enable_track=False,
            enable_extra=self.enable_extra,
            num_frames=dataset_cfg.num_frames
        )
    
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
    ) -> EncoderOutput:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        
        predictions = self.backbone(context["image"])
        
        # get predicted camera metrics
        extrin3x4, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (h, w))
        extrinsics = repeat(torch.eye(4, device=extrin3x4.device), 
                            "i j -> b v i j", 
                            b=b, v=extrin3x4.shape[1]).clone()
        extrinsics[:, :, :3, :] = extrin3x4
        intr_normed = intrinsics.clone()
        intr_normed[..., 0, :] /= w
        intr_normed[..., 1, :] /= h
        
        gaussians = predictions["gaussians"]  # (b, v, h, w, c)
        motions = predictions['motion_enc']
        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics[:, :v], "b v i j -> b v () () i j"),
            rearrange(intr_normed[:, :v], "b v i j -> b v () () i j"),
            rearrange(predictions["depth"], "b v h w () -> b v h w"),
            self.map_pdf_to_opacity(gaussians[..., 0].sigmoid(), global_step),
            gaussians[..., 1:],
            motions,
            (h, w)
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                predictions["depth"], "b v h w () -> b v h w ()", h=h, w=w
            )
            visualization_dump["scales"] = gaussians.scales
            visualization_dump["rotations"] = gaussians.rotations

        return EncoderOutput(
            pred_gaussian=gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            extra_ext4sup=predictions['pose_enc']
        )

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
