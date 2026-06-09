from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from .transform import cam_quat_xyzw_to_world_quat_wxyz
from ...types import Gaussians
from ....geometry.projection import get_world_rays, sample_image_grid


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        motions: Float[Tensor, "b v_t h w _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        device = extrinsics.device
        b, v = extrinsics.shape[:2]
        
        cam2worlds = affine_inverse(extrinsics)
        
        h, w = image_shape
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # b v h w xy
        origins, directions = get_world_rays(
            xy_ray,
            cam2worlds,
            intrinsics,
        )
        means = origins + directions * depths[..., None]
        means = rearrange(means, "b v h w ... -> b v (h w) ...")
        
        scales, rotations, colors = raw_gaussians.split((3, 4, 3), dim=-1)

        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None]
        scales = rearrange(scales, "b v h w ... -> b v (h w) ...")

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        rotations = rearrange(rotations, "b v h w ... -> b (v h w) ...")
        c2w_mat = repeat(
            cam2worlds,
            "b v () () i j -> b (v h w) i j",
            h=h,
            w=w,
        )
        world_quat_wxyz = cam_quat_xyzw_to_world_quat_wxyz(rotations, c2w_mat)
        world_quat_wxyz = rearrange(world_quat_wxyz, "b (v g) ... -> b v g ...", v=v)

        # Apply sigmoid to get valid colors.
        colors = rearrange(colors.sigmoid(), "b v h w ... -> b v (h w) ...")
        opacities = rearrange(opacities, "b v h w -> b v (h w)")
        
        t = motions.shape[1] // v
        motions = rearrange(motions, "b (v t) h w c -> b v t (h w) c", v=v, t=t)

        return Gaussians(
                means=means,
                scales=scales,
                rotations=world_quat_wxyz,
                colors=colors,
                opacities=opacities,
                motions=motions
            )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh


def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)