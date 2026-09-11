"""Pose engine 1: match-based pairwise registration + ICP refine -> SE(3) pose graph.

MoGe-3's per-frame metric point clouds are accurate, so poses come from directly REGISTERING pairs.
The rough relative pose is recovered from the LightGlue feature MATCHES (the same ones hloc computes
for BA) lifted to each frame's MoGe metric 3D and solved with RANSAC 3D-3D. Feature correspondences
stay reliable at small overlap and never confuse the repetitive room surfaces (near-identical
floor/walls) that blind FPFH-FGR matches to the WRONG wall — the failure that produced a
mixed-direction blob. Per pair:

  1. LightGlue matches for the pair -> lift matched keypoints to MoGe metric 3D,
  2. RANSAC 3D-3D (rigid) on those correspondences -> rough relative pose (rejected if too few inliers),
  3. point-to-plane ICP on the fine clouds from that rough transform -> exact relative pose.

Edges: odometry (consecutive; coast only when a pair has too few matches) + loop closure (retrieval
pairs, kept when confident). Open3D global_optimization (robust line process) closes loops + rejects
bad edges, then gravity-align (rotate so the averaged MoGe floor-normal points up) enforces the
planar-room prior. Rigid SE(3): MoGe-3 is metric, so per-frame scale is consistent (no scale DOF)."""

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
    """MoGe-3 inference + MATCH-based pairwise registration -> SE(3) pose graph + gravity align.
    Returns (geoms, poses_w2c (N,4,4)). The rough relative pose comes from the LightGlue matches
    (matches_h5) lifted to each frame's MoGe metric 3D (RANSAC 3D-3D), then point-to-plane ICP
    refines it — robust to the small-overlap / repetitive-surface pairs where FPFH-FGR mismatches
    the wrong wall. Loop candidates come from `pairs` (retrieval)."""
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

    # 1) per-frame: MoGe inference -> geom (keypoint 3D for the match-based pose) + fine cloud (ICP) + up.
    geoms: list[_FrameGeom] = []
    fine, up_cam = [], []
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
        fine.append(_to_pcd(points, mask, cfg.icp_voxel))
        up_cam.append(_up_from_normals(pred["normal"].cpu().numpy(), mask)
                      if "normal" in pred else None)
        if (i + 1) % 50 == 0:
            print(f"[moge3-sfm] MoGe inference + cloud {i + 1}/{n}")

    from hloc.utils.io import get_matches

    def _rough_from_matches(a, b):
        """Rough a->b pose from the LightGlue matches lifted to each frame's MoGe metric 3D
        (RANSAC 3D-3D). Feature correspondences stay reliable at small overlap and don't confuse
        the repetitive room surfaces that FPFH-FGR matches to the wrong wall. (T_a2b or None, n_inl)."""
        m, _ = get_matches(matches_h5, geoms[a].name, geoms[b].name)   # (M,2) kpt indices [a, b]
        if m.shape[0] < cfg.min_pair_inliers:
            return None, int(m.shape[0])
        v = geoms[a].kp_valid[m[:, 0]] & geoms[b].kp_valid[m[:, 1]]
        pa = geoms[a].kp3d[m[v, 0]].astype(np.float64)
        pb = geoms[b].kp3d[m[v, 1]].astype(np.float64)
        if len(pa) < cfg.min_pair_inliers:
            return None, int(len(pa))
        T, inl = u3d.np.solve_pose_ransac(pa, pb, mode="rigid",   # transform p(a)->q(b) = a->b
                                          threshold=cfg.ransac_threshold, rng=rng)
        if int(inl.sum()) < cfg.min_pair_inliers:
            return None, int(inl.sum())
        return np.asarray(T, dtype=np.float64), int(inl.sum())

    def register(a, b):
        """Match-based rough pose (a->b) then point-to-plane ICP refine on the fine clouds.
        Returns (T, info, fitness, rmse, n_match_inliers); T is None if the rough step fails."""
        Tinit, n_inl = _rough_from_matches(a, b)
        if Tinit is None:
            return None, np.eye(6), 0.0, 1e9, n_inl
        fine_dist = cfg.icp_max_corr_dist / 2.0
        if len(fine[a].points) < 10 or len(fine[b].points) < 10:
            return Tinit, np.eye(6), 1.0, 0.0, n_inl        # trust the match pose; too few pts to ICP
        reg = r.registration_icp(fine[a], fine[b], fine_dist, Tinit,
                                 r.TransformationEstimationPointToPlane(),
                                 r.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iter))
        T = np.asarray(reg.transformation)
        info = r.get_information_matrix_from_point_clouds(fine[a], fine[b], fine_dist, T)
        return T, info, reg.fitness, reg.inlier_rmse, n_inl

    # 2) odometry edges (consecutive). source=i-1, target=i so T = cam_i <- cam_{i-1} matches
    #    PoseGraphEdge(i-1, i, T). The match-based rough pose is reliable, so trust it; only when a
    #    consecutive pair has too few feature matches (fast motion / genuine no-overlap) COAST the init
    #    with the last accepted relative motion and add the edge down-weighted, letting loop closure
    #    recover it — never a wrong-but-confident FPFH pose.
    pg = r.PoseGraph()
    pg.nodes.append(r.PoseGraphNode(np.eye(4)))
    w2c = np.eye(4)
    prev_T = np.eye(4)          # last accepted relative motion (constant-velocity coast)
    n_bad_odo = 0
    for i in range(1, n):
        T, info, fit, rmse, n_inl = register(i - 1, i)
        good = T is not None
        if good:
            prev_T = T
        else:
            T, info, n_bad_odo = prev_T, np.eye(6) * 1e-2, n_bad_odo + 1
        w2c = T @ w2c
        pg.nodes.append(r.PoseGraphNode(np.linalg.inv(w2c)))
        pg.edges.append(r.PoseGraphEdge(i - 1, i, T, info, uncertain=not good))
    print(f"[moge3-sfm] {n - 1} odometry edges ({n_bad_odo} coasted — too few matches)")

    # 3) loop-closure edges from retrieval pairs, kept only when the registration is confident.
    n_loop = 0
    for a, b in pairs:
        ia, ib = name_to_idx[a], name_to_idx[b]
        if abs(ia - ib) <= cfg.sequential_window:
            continue
        T, info, fit, rmse, n_inl = register(ia, ib)
        if T is None or rmse > cfg.icp_loop_max_rmse:   # confident = enough match inliers + ICP agrees
            continue
        pg.edges.append(r.PoseGraphEdge(ia, ib, T, info, uncertain=True))
        n_loop += 1
    print(f"[moge3-sfm] {n_loop} loop-closure edges kept")

    # 4) global optimization (robust line process closes loops + rejects the few bad edges). With the
    #    reliable match-based init the odometry drift is small, so a single MODERATE mcd both closes
    #    loops and pulls surfaces to cm scale — no scene-scaled looseness (which left the FGR run ~1 m
    #    thick) and no two-stage tightening (which pruned real loops).
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
