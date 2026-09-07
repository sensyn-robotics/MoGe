"""Pose engine 1: globally-optimized ICP pose graph on MoGe metric clouds.

Chained ICP odometry accumulates unbounded rotation + scale drift — indoor planar
scenes give point-to-plane ICP a sliding ambiguity, so pure odometry collapsed into a
non-planar blob (cameras spanning tens of metres, some flung km away). This engine builds
a POSE GRAPH instead:

  * odometry edges: consecutive-frame ICP (trusted),
  * loop-closure edges: retrieval pairs, initialised from learned-match 3D-3D poses and
    ICP-refined, kept only when the registration is confident,
  * Open3D global optimization: Levenberg-Marquardt + a robust line process that closes
    loops, distributes drift, and rejects wrong loop edges,
  * gravity align: rotate the solved scene so the averaged MoGe floor-normal points up,
    enforcing the room's planar-floor prior.

Returns per-frame `_FrameGeom` + globally-consistent cam_from_world poses; downstream BA
only polishes (with a per-camera divergence guard in bundle_adjust)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from moge.sfm.config import MoGe3SfMConfig
from moge.sfm.moge_infer import _FrameGeom, _lift_frame, _resolve_moge_pretrained


def _to_pcd(points, mask, voxel):
    import open3d as o3d
    pts = points[mask & np.isfinite(points).all(-1)].astype(np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return pcd


def _up_from_normals(normal, mask):
    """Per-frame 'up' (opposite gravity) in the CAMERA frame from MoGe normals.

    Horizontal surfaces (floor/ceiling/tables) dominate an indoor scene and share one axis
    (vertical), so the top eigenvector of sum(n nᵀ) is that axis. Sign is ambiguous; orient
    it upward (OpenCV camera y points down, so world-up has a negative y component)."""
    n = normal[mask & np.isfinite(normal).all(-1)].astype(np.float64)
    if n.shape[0] < 50:
        return None
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    w, V = np.linalg.eigh(n.T @ n)
    v = V[:, -1]
    return v if v[1] < 0 else -v          # y<0 == upward in the OpenCV camera frame


def _pairwise_icp(src, tgt, init, max_corr, max_iter):
    import open3d as o3d
    r = o3d.pipelines.registration
    T = init
    for dist in (max_corr, max_corr / 6.0):
        reg = r.registration_icp(src, tgt, dist, T,
                                 r.TransformationEstimationPointToPlane(),
                                 r.ICPConvergenceCriteria(max_iteration=max_iter))
        T = np.asarray(reg.transformation)
    info = r.get_information_matrix_from_point_clouds(src, tgt, max_corr / 6.0, T)
    return T, info, reg.fitness, reg.inlier_rmse


def _loop_init_3d3d(geom_a, geom_b, m, cfg, rng):
    """Rough source(a)->target(b) transform from learned matches lifted to MoGe 3D
    (robust rigid Umeyama/RANSAC). Returns 4x4 T or None if too few inliers."""
    import utils3d_moge as u3d
    va, vb = geom_a.kp_valid[m[:, 0]], geom_b.kp_valid[m[:, 1]]
    v = va & vb
    if v.sum() < cfg.min_loop_inliers:
        return None
    pa, pb = geom_a.kp3d[m[v, 0]], geom_b.kp3d[m[v, 1]]
    pose, inl = u3d.np.solve_pose_ransac(pa, pb, mode="rigid",
                                         threshold=cfg.ransac_threshold, rng=rng)
    if inl.sum() < cfg.min_loop_inliers:
        return None
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = pose[:3, :3], pose[:3, 3]
    return T


def _gravity_align(poses_w2c, up_cam, valid):
    """Rotate the whole reconstruction so the averaged floor-normal is world +Z.

    up_cam[i] is frame i's up in its camera frame; R_wc_i @ up_cam_i is it in world.
    Average the valid ones, then rotate the gauge so that mean up = +Z (a horizontal
    floor / near-planar trajectory). Purely a global re-gauge — leaves residuals to BA."""
    ups = [np.linalg.inv(poses_w2c[i])[:3, :3] @ up_cam[i]
           for i in range(len(poses_w2c)) if valid[i]]
    if not ups:
        return poses_w2c
    g = np.mean(ups, 0)
    g /= np.linalg.norm(g) + 1e-9
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(g, z)
    s = np.linalg.norm(axis)
    if s < 1e-6:
        return poses_w2c
    axis /= s
    ang = np.arctan2(s, np.dot(g, z))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K   # world rotation g->+Z
    Rw = np.eye(4); Rw[:3, :3] = R
    return [w2c @ np.linalg.inv(Rw) for w2c in poses_w2c]         # cam_from_world' = w2c @ inv(Rworld)


def _icp_posegraph(image_paths, kpts, matches_h5, pairs, name_to_idx, cfg: MoGe3SfMConfig):
    """MoGe-3 inference + ICP pose graph (odometry + loop closure) + global optimization +
    gravity align. Returns (geoms, poses_w2c (N,4,4))."""
    import torch
    import open3d as o3d
    from hloc.utils.io import get_matches
    from moge.model import import_model_class_by_version
    import utils3d_moge as u3d
    r = o3d.pipelines.registration

    device = torch.device(cfg.device)
    model = (import_model_class_by_version(cfg.moge_version)
             .from_pretrained(_resolve_moge_pretrained(cfg)).to(device).eval())
    rng = np.random.default_rng(0)
    n = len(image_paths)

    # 1) per-frame: MoGe inference -> compact geom + downsampled cloud + camera-frame up.
    geoms: list[_FrameGeom] = []
    pcds, up_cam, depths = [], [], []
    for i, q in enumerate(image_paths):
        rgb = np.array(Image.open(q).convert("RGB"))
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            pred = model.infer(t)
        points = pred["points"].float().cpu().numpy()
        mask = (pred["mask"].cpu().numpy().astype(bool) if "mask" in pred
                else np.isfinite(points).all(-1))
        K = u3d.np.denormalize_intrinsics(pred["intrinsics"].cpu().numpy(),
                                          (rgb.shape[0], rgb.shape[1]))
        kp3d, kp_valid, kp_rgb, fuse_xyz, fuse_rgb = _lift_frame(points, mask, rgb, kpts[i], cfg, rng)
        geoms.append(_FrameGeom(q.name, (rgb.shape[1], rgb.shape[0]), K,
                                kp3d, kp_valid, kp_rgb, fuse_xyz, fuse_rgb))
        pcds.append(_to_pcd(points, mask, cfg.icp_voxel))
        up_cam.append(_up_from_normals(pred["normal"].cpu().numpy(), mask)
                      if "normal" in pred else None)
        vp = points[mask & np.isfinite(points).all(-1)]
        depths.append(float(np.median(np.linalg.norm(vp, axis=1))) if vp.size else np.nan)
        if (i + 1) % 50 == 0:
            print(f"[moge3-sfm] MoGe inference + cloud {i + 1}/{n}")

    # 1b) scale normalization: MoGe's per-frame metric scale drifts, so rigid ICP + pose graph
    # can't fix it and the reconstruction balloons (~4x, tens of metres tall). Pin every frame to a
    # common scale via its median point distance (robust, ~constant across a room walk) -> a
    # scale-consistent, ~metric reconstruction that the SE(3) graph CAN keep coherent.
    if cfg.scale_normalize:
        d = np.array(depths); ref = float(np.nanmedian(d))
        for i in range(n):
            if not np.isfinite(d[i]) or d[i] < 1e-6:
                continue
            s = ref / d[i]
            pcds[i].scale(s, center=(0.0, 0.0, 0.0))
            geoms[i].kp3d = geoms[i].kp3d * s
            geoms[i].fuse_xyz = (geoms[i].fuse_xyz * s).astype(np.float32)
        sc = ref / d[np.isfinite(d) & (d > 1e-6)]
        print(f"[moge3-sfm] scale-normalized to median depth {ref:.2f}m "
              f"(per-frame scale {sc.min():.2f}-{sc.max():.2f})")

    # 2) odometry edges (consecutive ICP), accumulate node init (cam->world = inv of w2c chain).
    pg = r.PoseGraph()
    pg.nodes.append(r.PoseGraphNode(np.eye(4)))
    w2c_chain = np.eye(4)
    prev_rel = np.eye(4)          # last accepted consecutive motion (constant-velocity prior)
    n_weak = 0
    for i in range(1, n):
        # source=i-1, target=i so T = T_{cam_i <- cam_{i-1}} matches PoseGraphEdge(i-1, i, T)
        # (Open3D reads edge.transformation as source->target). Init ICP from the last accepted motion.
        T, info, fit, _ = _pairwise_icp(pcds[i - 1], pcds[i], prev_rel,
                                        cfg.icp_max_corr_dist, cfg.icp_max_iter)
        # Robust odometry: a bad consecutive ICP (low overlap / implausible jump) is still an
        # uncertain=False backbone edge the line process can't reject, so it would enforce a jump.
        # Coast on the constant-velocity prior and down-weight the edge so loops position the node.
        if fit < cfg.icp_min_fitness or np.linalg.norm(T[:3, 3]) > cfg.icp_max_motion:
            T = prev_rel; info = info * 0.05; n_weak += 1
        else:
            prev_rel = T
        w2c_chain = T @ w2c_chain
        pg.nodes.append(r.PoseGraphNode(np.linalg.inv(w2c_chain)))
        pg.edges.append(r.PoseGraphEdge(i - 1, i, T, info, uncertain=False))
    print(f"[moge3-sfm] {n - 1} odometry edges ({n_weak} weak/coasted)")

    # 3) loop-closure edges from retrieval pairs (|i-j| beyond the sequential window).
    n_loop = 0
    for a, b in pairs:
        ia, ib = name_to_idx[a], name_to_idx[b]
        if abs(ia - ib) <= cfg.sequential_window:
            continue
        m, _ = get_matches(matches_h5, a, b)
        if m.shape[0] < cfg.min_loop_inliers:
            continue
        init = _loop_init_3d3d(geoms[ia], geoms[ib], m, cfg, rng)
        if init is None:
            continue
        T, info, fit, rmse = _pairwise_icp(pcds[ia], pcds[ib], init,
                                           cfg.icp_max_corr_dist, cfg.icp_max_iter)
        if fit < cfg.icp_loop_min_fitness or rmse > cfg.icp_loop_max_rmse:
            continue
        pg.edges.append(r.PoseGraphEdge(ia, ib, T, info, uncertain=True))
        n_loop += 1
    print(f"[moge3-sfm] {n_loop} loop-closure edges kept")

    # 4) global optimization (robust line process closes loops + rejects bad edges).
    # max_correspondence_distance must scale with the trajectory: too small (e.g. the ICP fine
    # radius ~0.05m) makes the line process reject the large residuals of drifted loop edges, so
    # NOTHING closes. Validated on synthetic loops (0.15*diag closes loops across 6-50m scales
    # while still rejecting inconsistent loops). Diagonal from a robust (2-98%) node bbox.
    init_C = np.stack([np.asarray(node.pose)[:3, 3] for node in pg.nodes])
    diag = float(np.linalg.norm(np.percentile(init_C, 98, 0) - np.percentile(init_C, 2, 0)))
    mcd = float(np.clip(0.15 * diag, 0.3, 3.0))
    opt = r.GlobalOptimizationOption(
        max_correspondence_distance=mcd,
        edge_prune_threshold=cfg.posegraph_prune_threshold,
        preference_loop_closure=1.0, reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        r.global_optimization(pg, r.GlobalOptimizationLevenbergMarquardt(),
                              r.GlobalOptimizationConvergenceCriteria(), opt)
    poses_w2c = [np.linalg.inv(np.asarray(node.pose)) for node in pg.nodes]

    # 5) gravity align (rotate scene so averaged MoGe floor-normal = up -> planar room).
    if cfg.gravity_align and any(u is not None for u in up_cam):
        valid = [u is not None for u in up_cam]
        safe_up = [u if u is not None else np.array([0.0, -1.0, 0.0]) for u in up_cam]
        poses_w2c = _gravity_align(poses_w2c, safe_up, valid)

    print(f"[moge3-sfm] pose graph optimized: {n} frames, {n_loop} loops")
    return geoms, np.stack(poses_w2c)
