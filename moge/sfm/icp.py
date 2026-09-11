"""Pose engine 1: Open3D RANSAC feature-based global registration + ICP refine -> SE(3) pose graph.

MoGe-3's per-frame metric point clouds are accurate, so poses come from directly REGISTERING the
clouds. The rough relative pose is the Open3D global-registration tutorial's PRIMARY method:
FPFH features + RANSAC feature matching (registration_ransac_based_on_feature_matching). Unlike Fast
Global Registration (FGR), RANSAC applies edge-length + distance correspondence checkers that reject
the geometrically-inconsistent matches FGR accepted on the repetitive room surfaces (near-identical
floor/walls) — the wrong-wall matching that produced a mixed-direction blob. Per pair:

  1. subsample each cloud (coarse voxel ~1-2k pts) + FPFH features,
  2. RANSAC feature matching (with checkers, no pose prior) -> a rough transform,
  3. point-to-plane ICP on the finer clouds from that rough transform -> exact relative pose.

REJECT WRONG PAIRS: an odometry edge is coasted when its ICP overlap (fitness) or motion is
implausible; a loop edge is kept only when BOTH the RANSAC rough registration is confident
(fitness >= ransac_min_loop_fitness) AND the ICP agrees (fitness/rmse). Open3D global_optimization
(robust line process) then closes loops + rejects the residual bad edges, and gravity-align rotates
the scene so the averaged MoGe floor-normal points up (planar-room prior). Rigid SE(3): MoGe-3 is
metric, so per-frame scale is consistent and no scale DOF is needed."""

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
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    return pcd


def _fpfh(pcd, voxel):
    import open3d as o3d
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))


def _up_from_normals(normal, mask):
    """Per-frame 'up' (opposite gravity) in the CAMERA frame from MoGe normals.

    Horizontal surfaces (floor/ceiling/tables) dominate an indoor scene and share one axis
    (vertical), so the top eigenvector of sum(n nᵀ) is that axis. Sign is ambiguous; orient
    it upward (OpenCV camera y points down, so world-up has a negative y component)."""
    n = normal[mask & np.isfinite(normal).all(-1)].astype(np.float64)
    if n.shape[0] < 50:
        return None
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    _, V = np.linalg.eigh(n.T @ n)
    v = V[:, -1]
    return v if v[1] < 0 else -v


def _register_pair(cs, fs, ct, ft, fine_s, fine_t, cfg):
    """Open3D RANSAC feature-based global registration + ICP refine, source->target.
    Returns (T, info, fitness, rmse, rough_fit): `fitness`/`rmse` are the ICP overlap, `rough_fit`
    is the RANSAC feature-matching fitness (used to reject unconfident loop pairs). RANSAC needs no
    pose prior, so it also handles wide-baseline loop pairs; consecutive frames that RANSAC can't
    register still align by ICP from identity (small motion)."""
    import open3d as o3d
    r = o3d.pipelines.registration
    dist = cfg.coarse_voxel * 1.5
    fine_dist = cfg.icp_max_corr_dist / 2.0
    Tinit, rough_fit = np.eye(4), 0.0
    if len(cs.points) >= 20 and len(ct.points) >= 20:
        try:
            res = r.registration_ransac_based_on_feature_matching(
                cs, ct, fs, ft, True, dist,
                r.TransformationEstimationPointToPoint(False), 3,
                [r.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 r.CorrespondenceCheckerBasedOnDistance(dist)],
                r.RANSACConvergenceCriteria(cfg.ransac_max_iter, cfg.ransac_confidence))
            Tf = np.asarray(res.transformation)
            if Tf.shape == (4, 4) and np.isfinite(Tf).all():
                Tinit, rough_fit = Tf, float(res.fitness)
        except Exception:
            pass
    if len(fine_s.points) < 10 or len(fine_t.points) < 10:
        return Tinit, np.eye(6), 0.0, 1e9, rough_fit
    reg = r.registration_icp(fine_s, fine_t, fine_dist, Tinit,
                             r.TransformationEstimationPointToPlane(),
                             r.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iter))
    T = np.asarray(reg.transformation)
    info = r.get_information_matrix_from_point_clouds(fine_s, fine_t, fine_dist, T)
    return T, info, reg.fitness, reg.inlier_rmse, rough_fit


def _gravity_align(poses_w2c, up_cam, valid):
    """Rotate the whole reconstruction so the averaged floor-normal is world +Z."""
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
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    Rw = np.eye(4); Rw[:3, :3] = R
    return [w2c @ np.linalg.inv(Rw) for w2c in poses_w2c]


def _icp_posegraph(image_paths, kpts, matches_h5, pairs, name_to_idx, cfg: MoGe3SfMConfig):
    """MoGe-3 inference + RANSAC/ICP pairwise registration -> SE(3) pose graph + gravity align.
    Returns (geoms, poses_w2c (N,4,4)). `matches_h5` is unused for poses (kept for the BA caller);
    loop candidates come from `pairs` (retrieval), registered by cloud geometry."""
    import torch
    import open3d as o3d
    from moge.model import import_model_class_by_version
    import utils3d_moge as u3d
    r = o3d.pipelines.registration

    device = torch.device(cfg.device)
    model = (import_model_class_by_version(cfg.moge_version)
             .from_pretrained(_resolve_moge_pretrained(cfg)).to(device).eval())
    rng = np.random.default_rng(0)
    n = len(image_paths)

    # 1) per-frame: MoGe inference -> geom + fine cloud (ICP) + coarse cloud & FPFH (RANSAC) + up.
    geoms: list[_FrameGeom] = []
    fine, coarse, fpfh, up_cam = [], [], [], []
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
        cf = _to_pcd(points, mask, cfg.coarse_voxel)
        fine.append(_to_pcd(points, mask, cfg.icp_voxel))
        coarse.append(cf)
        fpfh.append(_fpfh(cf, cfg.coarse_voxel))
        up_cam.append(_up_from_normals(pred["normal"].cpu().numpy(), mask)
                      if "normal" in pred else None)
        if (i + 1) % 50 == 0:
            print(f"[moge3-sfm] MoGe inference + cloud {i + 1}/{n}")

    def register(a, b):
        return _register_pair(coarse[a], fpfh[a], coarse[b], fpfh[b], fine[a], fine[b], cfg)

    # 2) odometry edges (consecutive), accumulate node init. source=i-1, target=i so
    #    T = cam_i <- cam_{i-1} matches PoseGraphEdge(i-1, i, T). REJECT a wrong edge: a blurry /
    #    low-overlap / wrong-wall registration yields a bad ICP that, trusted, offsets the whole
    #    downstream node-init chain. On failure (ICP overlap below icp_min_fitness or motion above
    #    icp_max_motion), COAST the init through it (reuse the last accepted relative motion) and add
    #    the edge uncertain + heavily down-weighted, letting loop closure recover the pose.
    pg = r.PoseGraph()
    pg.nodes.append(r.PoseGraphNode(np.eye(4)))
    w2c = np.eye(4)
    prev_T = np.eye(4)          # last accepted relative motion (constant-velocity coast)
    n_bad_odo = 0
    for i in range(1, n):
        T, info, fit, _, _ = register(i - 1, i)
        good = fit >= cfg.icp_min_fitness and float(np.linalg.norm(T[:3, 3])) <= cfg.icp_max_motion
        if good:
            prev_T = T
        else:
            T, info, n_bad_odo = prev_T, info * 1e-2, n_bad_odo + 1
        w2c = T @ w2c
        pg.nodes.append(r.PoseGraphNode(np.linalg.inv(w2c)))
        pg.edges.append(r.PoseGraphEdge(i - 1, i, T, info, uncertain=not good))
    print(f"[moge3-sfm] {n - 1} odometry edges ({n_bad_odo} coasted — low overlap/motion)")

    # 3) loop-closure edges from retrieval pairs, kept only when BOTH the RANSAC rough registration
    #    is confident (rough_fit) AND the ICP agrees (fitness/rmse) — rejecting the wrong-wall loops.
    n_loop = 0
    for a, b in pairs:
        ia, ib = name_to_idx[a], name_to_idx[b]
        if abs(ia - ib) <= cfg.sequential_window:
            continue
        T, info, fit, rmse, rough_fit = register(ia, ib)
        if (rough_fit < cfg.ransac_min_loop_fitness
                or fit < cfg.icp_loop_min_fitness or rmse > cfg.icp_loop_max_rmse):
            continue
        pg.edges.append(r.PoseGraphEdge(ia, ib, T, info, uncertain=True))
        n_loop += 1
    print(f"[moge3-sfm] {n_loop} loop-closure edges kept")

    # 4) global optimization (robust line process closes loops + rejects bad edges). With reliable
    #    RANSAC+ICP edges a single MODERATE mcd both closes loops and pulls surfaces to cm scale — no
    #    scene-scaled looseness (which left ~1 m-thick surfaces) and no two-stage tightening (which
    #    pruned real loops whose initial residual exceeds the tight mcd under drift).
    mcd = float(max(3.0 * cfg.icp_voxel, 0.20))
    opt = r.GlobalOptimizationOption(max_correspondence_distance=mcd,
                                     edge_prune_threshold=cfg.posegraph_prune_threshold,
                                     preference_loop_closure=1.0, reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        r.global_optimization(pg, r.GlobalOptimizationLevenbergMarquardt(),
                              r.GlobalOptimizationConvergenceCriteria(), opt)
    poses_w2c = [np.linalg.inv(np.asarray(nd.pose)) for nd in pg.nodes]

    # 5) gravity align (rotate scene so averaged MoGe floor-normal = up -> planar room).
    if cfg.gravity_align and any(u is not None for u in up_cam):
        valid = [u is not None for u in up_cam]
        safe_up = [u if u is not None else np.array([0.0, -1.0, 0.0]) for u in up_cam]
        poses_w2c = _gravity_align(poses_w2c, safe_up, valid)

    print(f"[moge3-sfm] pose graph optimized: {n} frames, {n_loop} loops")
    return geoms, np.stack(poses_w2c)
