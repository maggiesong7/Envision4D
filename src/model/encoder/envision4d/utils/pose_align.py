from typing import List
import numpy as np
import torch
from evo.core.trajectory import PosePath3D


def apply_batch_alignment_to_ext(
    rots: torch.Tensor, trans: torch.Tensor, scales: torch.Tensor, pose_est: torch.Tensor
):
    pose_new_align_rot = rots[:, None] @ pose_est[..., :3, :3]
    pose_new_align_trans = (
        scales[:, None, None] * (rots[:, None] @ pose_est[..., :3, 3:])[..., 0] + trans[:, None]
    )
    pose_new_align = torch.zeros_like(pose_est)
    pose_new_align[..., :3, :3] = pose_new_align_rot
    pose_new_align[..., :3, 3] = pose_new_align_trans
    pose_new_align[..., 3, 3] = 1.0
    return pose_new_align


def batch_align_poses_umeyama(ext_ref: torch.Tensor, ext_est: torch.Tensor):
    device, dtype = ext_ref.device, ext_ref.dtype
    assert ext_ref.dtype in [torch.float32, torch.float64]
    assert ext_est.dtype in [torch.float32, torch.float64]
    assert ext_ref.requires_grad is False
    assert ext_est.requires_grad is False
    rots, trans, scales = [], [], []
    for b in range(ext_ref.shape[0]):
        r, t, s, _ = _umeyama_sim3_from_paths(ext_ref[b].cpu().numpy(), ext_est[b].cpu().numpy())
        rots.append(torch.from_numpy(r).to(device=device, dtype=dtype))
        trans.append(torch.from_numpy(t).to(device=device, dtype=dtype))
        scales.append(torch.tensor(s, device=device, dtype=dtype))
    return torch.stack(rots), torch.stack(trans), torch.stack(scales)


def _umeyama_sim3_from_paths(pose_ref, pose_est):
    path_ref = PosePath3D(poses_se3=pose_ref.copy())
    path_est = PosePath3D(poses_se3=pose_est.copy())
    r, t, s = path_est.align(path_ref, correct_scale=True)
    pose_est_aligned = np.stack(path_est.poses_se3)
    return r, t, s, pose_est_aligned


def estimate_sim3_from_poses(pose_est, pose_ref):
    """
    Estimate Sim(3) transformation for each pair in batch
    
    Parameters:
        pose_est: [b,4,4] Estimated camera poses
        pose_ref: [b,4,4] Reference camera poses
    
    Returns:
        scale: [b] Scale factor for each pair
        rotation: [b,3,3] Rotation matrix for each pair
        translation: [b,3] Translation vector for each pair
    """
    assert pose_est.shape == pose_ref.shape
    assert pose_est.dim() == 3 and pose_est.shape[1:] == (4, 4)
    
    batch_size = pose_est.shape[0]
    device = pose_est.device
    dtype = pose_est.dtype
    
    # Initialize outputs
    scale = torch.zeros(batch_size, device=device, dtype=dtype)
    rotation = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    translation = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    
    # Extract camera positions
    cam_pos_est = pose_est[:, :3, 3]  # [b, 3]
    cam_pos_ref = pose_ref[:, :3, 3]  # [b, 3]
    
    # Process each batch separately
    for i in range(batch_size):
        # Get single pose pair
        pos_est_i = cam_pos_est[i:i+1]  # [1, 3]
        pos_ref_i = cam_pos_ref[i:i+1]  # [1, 3]
        
        # 1. Estimate scale (single pair case)
        dist_est = torch.norm(pos_est_i)
        dist_ref = torch.norm(pos_ref_i)
        
        if dist_est > 1e-6 and dist_ref > 1e-6:
            scale[i] = dist_ref / dist_est
        else:
            scale[i] = torch.tensor(1.0, device=device, dtype=dtype)
        
        # 2. Estimate rotation (for single pair, rotation is identity if only one point)
        # With only one correspondence, we can't determine rotation uniquely
        # Use the rotation from the pose matrices directly
        R_est_i = pose_est[i, :3, :3]
        R_ref_i = pose_ref[i, :3, :3]
        
        # Rotation from est to ref: R = R_ref * R_est^T
        rotation[i] = R_ref_i @ R_est_i.T
        
        # Ensure right-handed coordinate system
        if torch.det(rotation[i]) < 0:
            # Flip the sign
            rotation[i] = -rotation[i]
        
        # 3. Estimate translation
        # t = p_ref - s * (R @ p_est)
        translation[i] = pos_ref_i - scale[i] * (rotation[i] @ pos_est_i.T)
    
    return rotation, translation, scale