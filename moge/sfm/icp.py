"""Pose engine 1: ICP odometry on MoGe metric clouds.

Chain point-to-plane ICP of consecutive MoGe metric point clouds (frame i registered to
frame i-1) into cam_from_world poses accumulated from frame 0. Robust on a continuous
walk and needs no cross-image matching for the init; global BA removes the residual drift
afterwards. A gate rejects low-overlap / implausibly-large registrations and coasts on the
constant-velocity prior, so one bad ICP can't jump the whole downstream chain."""

from __future__ import annotations

import numpy as np
from PIL import Image

from moge.sfm.config import MoGe3SfMConfig
from moge.sfm.moge_infer import _FrameGeom, _lift_frame, _resolve_moge_pretrained


def _icp_odometry(image_paths, kpts, cfg: MoGe3SfMConfig):
    """MoGe-3 inference + ICP odometry: chain point-to-plane ICP of consecutive metric clouds
    (frame i registered to frame i-1) into cam_from_world poses, accumulated from frame 0.
    Returns (geoms, poses_w2c (N,4,4)). Streaming — only the previous cloud is kept."""
    import torch
    import open3d as o3d
    from moge.model import import_model_class_by_version
    import utils3d_moge as u3d

    device = torch.device(cfg.device)
    model = (import_model_class_by_version(cfg.moge_version)
             .from_pretrained(_resolve_moge_pretrained(cfg)).to(device).eval())
    rng = np.random.default_rng(0)

    def to_pcd(points, mask):
        pts = points[mask & np.isfinite(points).all(-1)].astype(np.float64)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd = pcd.voxel_down_sample(cfg.icp_voxel)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.icp_voxel * 3, max_nn=30))
        return pcd

    geoms: list[_FrameGeom] = []
    poses: list[np.ndarray] = []
    prev_pcd = None
    cfw = np.eye(4)                      # cam_from_world of the previous frame (frame 0 = world)
    prev_rel = np.eye(4)                 # last ACCEPTED frame_i->frame_{i-1} transform (constant-velocity prior)
    n_fallback = 0
    for i, q in enumerate(image_paths):
        rgb = np.array(Image.open(q).convert("RGB"))
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            pred = model.infer(t)
        points = pred["points"].float().cpu().numpy()
        mask = (pred["mask"].cpu().numpy().astype(bool) if "mask" in pred
                else np.isfinite(points).all(-1))
        K = u3d.np.denormalize_intrinsics(pred["intrinsics"].cpu().numpy(), (rgb.shape[1], rgb.shape[0]))
        kp3d, kp_valid, kp_rgb, fuse_xyz, fuse_rgb = _lift_frame(points, mask, rgb, kpts[i], cfg, rng)
        geoms.append(_FrameGeom(q.name, (rgb.shape[1], rgb.shape[0]), K,
                                kp3d, kp_valid, kp_rgb, fuse_xyz, fuse_rgb))

        pcd = to_pcd(points, mask)
        if prev_pcd is not None:
            # ICP source=frame i, target=frame i-1: T maps frame-i coords -> frame-(i-1) coords.
            # Init from the previous accepted motion (constant velocity), coarse then fine radius.
            T = prev_rel.copy()
            reg = None
            for dist in (cfg.icp_max_corr_dist, cfg.icp_max_corr_dist / 6.0):
                reg = o3d.pipelines.registration.registration_icp(
                    pcd, prev_pcd, dist, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iter))
                T = np.asarray(reg.transformation)
            # Gate: a bad registration (low overlap fitness or implausibly large motion) would jump
            # this camera and, since odometry is a chain, everything downstream. Reject it and coast
            # on the constant-velocity prior instead.
            if reg.fitness < cfg.icp_min_fitness or np.linalg.norm(T[:3, 3]) > cfg.icp_max_motion:
                T = prev_rel
                n_fallback += 1
            else:
                prev_rel = T
            cfw = np.linalg.inv(T) @ cfw
        poses.append(cfw.copy())
        prev_pcd = pcd
        if (i + 1) % 50 == 0:
            print(f"[moge3-sfm] MoGe inference + ICP {i + 1}/{len(image_paths)}")
    print(f"[moge3-sfm] ICP odometry: {n_fallback} constant-velocity fallbacks (bad/implausible ICP)")
    print(f"[moge3-sfm] ICP odometry done: {len(geoms)} frames")
    return geoms, np.stack(poses)
